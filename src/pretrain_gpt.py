# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""Pretrain and SFT GPT."""

import random
from functools import partial
from typing import Optional, Tuple

import torch
from datasets import concatenate_datasets, load_dataset
from megatron.core import parallel_state
from megatron.core.enums import ModelType
from megatron.core.models.gpt import GPTModel
from megatron.core.rerun_state_machine import RerunDataIterator, get_rerun_state_machine
from megatron.core.utils import StragglerDetector, get_attr_wrapped_model
from megatron.training import get_args, get_timers, inprocess_restart, pretrain, print_rank_0
from megatron.training.utils import is_first_or_last_pipeline_stage
from transformers import AutoTokenizer

from gpt_builders import gpt_builder
from model_provider import model_provider

try:
    from megatron.post_training.arguments import add_modelopt_args
    from megatron.post_training.loss_func import loss_func as loss_func_modelopt

    has_nvidia_modelopt = True
except ImportError:
    has_nvidia_modelopt = False

stimer = StragglerDetector()


def get_batch(data_iterator: RerunDataIterator, vp_stage=None):
    """Generate a batch."""
    # TODO: this is pretty hacky, find a better way
    if not is_first_or_last_pipeline_stage(vp_stage):
        return None, None, None, None, None

    tokens, labels, loss_mask, attention_mask, position_ids = next(data_iterator)

    return tokens.cuda(), labels.cuda(), loss_mask.cuda(), attention_mask.cuda(), position_ids.cuda()


# define spiky loss as a loss that's 10x the max loss observed
SPIKY_LOSS_FACTOR = 10


def loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor, model: Optional[GPTModel] = None):
    """Loss function.

    Args:
        loss_mask (torch.Tensor): Used to mask out some portions of the loss
        output_tensor (torch.Tensor): The tensor with the losses
        model (GPTModel, optional): The model (can be wrapped)

    Returns:
        the loss scalar for this micro-batch
        the number of non-padded tokens in this microbatch
        a dict containing reporting metrics on the loss and number of tokens across
            the data parallel ranks
    """
    args = get_args()

    if has_nvidia_modelopt and getattr(args, "modelopt_enabled", False):  # [ModelOpt]
        return loss_func_modelopt(loss_mask, output_tensor, model=model)

    losses = output_tensor.view(-1).float()
    loss_mask = loss_mask.view(-1).float()
    loss = torch.sum(losses * loss_mask)

    # Check individual rank losses are not NaN prior to DP all-reduce.
    rerun_state_machine = get_rerun_state_machine()
    if args.check_for_nan_in_loss_and_grad:
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=torch.isnan,
            message="found NaN in local forward loss calculation",
            tolerance=0.0,  # forward pass calculations are determinisic
            fatal=True,
        )
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=torch.isinf,
            message="found Inf in local forward loss calculation",
            tolerance=0.0,  # forward pass calculations are determinisic
            fatal=True,
        )
    # Check for spiky loss
    if args.check_for_spiky_loss:
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=partial(
                rerun_state_machine.is_unexpectedly_large,
                threshold=SPIKY_LOSS_FACTOR,
                context="loss",
            ),
            message="Spiky loss",
            tolerance=0.0,  # forward pass calculations are determinisic
            fatal=False,
        )

    num_tokens = loss_mask.sum().clone().detach().to(torch.int)
    reporting_loss = torch.cat([loss.clone().detach().view(1), num_tokens.view(1)])

    return (loss, num_tokens, {"lm loss": reporting_loss})


def forward_step(data_iterator, model: GPTModel, return_schedule_plan: bool = False):
    """Forward training step.

    Args:
        data_iterator : Input data iterator
        model (GPTModel): The GPT Model
        return_schedule_plan (bool): Whether to return the schedule plan instead of the output tensor
    """
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers("batch-generator", log_level=2).start()
    global stimer
    with stimer(bdata=True):
        vp_stage = get_attr_wrapped_model(model, "vp_stage")
        tokens, labels, loss_mask, attention_mask, position_ids = get_batch(data_iterator, vp_stage)
    timers("batch-generator").stop()

    with stimer:
        if args.use_legacy_models:
            output_tensor = model(tokens, position_ids, attention_mask, labels=labels)
        else:
            if return_schedule_plan:
                assert args.overlap_moe_expert_parallel_comm, (
                    "overlap_moe_expert_parallel_comm must be enabled to return the schedule plan"
                )
                schedule_plan = model.build_schedule_plan(
                    tokens, position_ids, attention_mask, labels=labels, loss_mask=loss_mask
                )
                return schedule_plan, partial(loss_func, loss_mask, model=model)
            else:
                output_tensor = model(tokens, position_ids, attention_mask, labels=labels, loss_mask=loss_mask)

    # [ModelOpt]: model is needed to access ModelOpt distillation losses
    return output_tensor, partial(loss_func, loss_mask, model=model)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset, tokenizer, seq_length: int = 128):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.seq_length = seq_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        item = self.dataset[index]

        item = "".join(f"<{unit}>" for unit in item["units"])
        item = item + self.tokenizer.eos_token

        inputs = self.tokenizer(item, padding=False)

        # random truncation
        length = len(inputs.input_ids)
        if length > self.seq_length + 1:
            start = random.randrange(0, length - self.seq_length - 1)
            input_ids = inputs.input_ids[start : start + self.seq_length + 1]
        else:
            input_ids = inputs.input_ids + [self.tokenizer.pad_token_id] * (self.seq_length + 1 - length)

        inputs = self.tokenizer.pad({"input_ids": input_ids}, return_tensors="pt")

        tokens = inputs.input_ids[:-1]
        labels = inputs.input_ids[1:]
        loss_mask = inputs.attention_mask.float()[1:]
        attention_mask = torch.triu(
            torch.ones((self.seq_length, self.seq_length), dtype=torch.bool), diagonal=1
        ).unsqueeze(0)
        position_ids = torch.arange(self.seq_length, dtype=torch.long)

        return tokens, labels, loss_mask, attention_mask, position_ids


def train_valid_test_datasets_provider(train_val_test_num_samples, vp_stage=None):
    """Build the train test and validation datasets.

    Args:
        train_val_test_num_samples : A list containing the number of samples in train test and validation.
    """
    args = get_args()

    print_rank_0("> building train, validation, and test datasets for GPT ...")

    librilight = load_dataset(args.data_path[0], "Libri-Light", split="train", keep_in_memory=True)
    libriheavy = load_dataset(args.data_path[0], "libriheavy", split="train", keep_in_memory=True)
    librispeech = load_dataset(args.data_path[0], "LibriSpeech", split="train", keep_in_memory=True)
    tinystories = load_dataset(args.data_path[0], "TinyStories", split="train", keep_in_memory=True)
    peoples_speech = load_dataset(args.data_path[0], "peoples_speech", split="train", keep_in_memory=True)
    voxpopuli = load_dataset(args.data_path[0], "voxpopuli", split="train", keep_in_memory=True)

    train_dataset = concatenate_datasets(
        [
            libriheavy,
            librispeech,
            tinystories,
            peoples_speech,
            voxpopuli,
            librilight,
            librispeech.remove_columns("aligned_units"),
            tinystories.remove_columns("aligned_units"),
            peoples_speech.remove_columns("aligned_units"),
            voxpopuli.remove_columns("aligned_units"),
        ]
    )
    train_dataset = train_dataset.shard(
        parallel_state.get_data_parallel_world_size(), parallel_state.get_data_parallel_rank()
    )

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = Dataset(train_dataset, tokenizer, args.seq_length)

    print_rank_0("> finished creating GPT datasets ...")

    return train_dataset, None, None


if __name__ == "__main__":
    # Temporary for transition to core datasets
    train_valid_test_datasets_provider.is_distributed = True

    # Optionally enable inprocess restart on pretrain
    pretrain, store = inprocess_restart.maybe_wrap_for_inprocess_restart(pretrain)

    pretrain(
        train_valid_test_datasets_provider,
        partial(model_provider, gpt_builder),
        ModelType.encoder_or_decoder,
        forward_step,
        args_defaults={"tokenizer_type": "HuggingFaceTokenizer"},
        extra_args_provider=add_modelopt_args if has_nvidia_modelopt else None,
        store=store,
    )
