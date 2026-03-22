import argparse
from pathlib import Path

import torch
from megatron.bridge import AutoBridge
from megatron.bridge.models.decorators import torchrun_main
from transformers import AutoModelForCausalLM, AutoTokenizer


def expand_vocab(args):
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model_id, cache_dir=args.cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    vocab = tokenizer.get_vocab()
    units = [f"<{unit}>" for unit in range(args.vocab_size)]
    for unit in units:
        assert unit not in vocab
    tokenizer.add_tokens(units)

    # Model
    model = AutoModelForCausalLM.from_pretrained(args.hf_model_id, cache_dir=args.cache_dir)
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)

    # Save
    Path(args.save_dir).parent.mkdir(parents=True, exist_ok=True)

    tokenizer.save_pretrained(args.save_dir)
    model.save_pretrained(args.save_dir)


@torchrun_main
def main(args):
    Path(args.megatron_save_path).parent.mkdir(parents=True, exist_ok=True)

    bridge = AutoBridge.from_hf_pretrained(args.save_dir, trust_remote_code=True, torch_dtype=torch.bfloat16)
    model_provider = bridge.to_megatron_provider(load_weights=True)
    model_provider.pipeline_dtype = torch.bfloat16
    model_provider.params_dtype = torch.bfloat16
    model_provider.expert_model_parallel_size = args.ep
    model_provider.finalize()
    model_provider.initialize_model_parallel(seed=0)
    megatron_model = model_provider.provide_distributed_model(wrap_with_ddp=False)
    bridge.save_megatron_model(megatron_model, args.megatron_save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-model-id", type=str, default="", help="HuggingFace model ID")
    parser.add_argument("--save-dir", type=str, default="")
    parser.add_argument(
        "--megatron-save-path", type=str, default="", help="Path to save the model in Megatron checkpoint format"
    )
    parser.add_argument("--cache-dir", type=str, default="~/.cache/huggingface/hub", help="HuggingFace cache dir")
    parser.add_argument("--vocab-size", type=int, default=8192, help="Speech vocab size")
    parser.add_argument("--ep", type=int, default=4, help="Expert parallelism size")

    args = parser.parse_args()

    main(args)

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
