#!/bin/bash

#$ -cwd                      ## Execute a job in the current directory
#$ -l node_f=1               ## Use number of node
#$ -l h_rt=24:00:00          ## Running job time
#$ -j y                      ## Integrate standard error output into a standard output
#$ -p -5
#$ -m abe
#$ -M EMAIL_ADDRESS

module load openmpi/5.0.7-nvhpc
module load cudnn/9.8.0
module load nccl/2.20.5
module load miniconda

TOKENIZER_MODEL=$1
DATA_PATH=$2
CKPT_PATH=$3
HOSTFILE=${4:-hostfile}

eval "$(/apps/t4/rhel9/free/miniconda/24.1.2/bin/conda shell.bash hook)"
conda activate py312
torchrun --nproc_per_node=4 src/convert.py

export CUDA_DEVICE_MAX_CONNECTIONS=1
export MASTER_ADDR=$(head -n 1 $PE_HOSTFILE | awk '{print $1}')
export MASTER_PORT=29500
export HF_HOME=~/.cache/huggingface

GPUS_PER_NODE=4
NNODES=1
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

awk '{print $1 " slots=4"}' "$PE_HOSTFILE" > $HOSTFILE

mpirun \
    --hostfile $HOSTFILE \
    -npernode $GPUS_PER_NODE \
    -n $WORLD_SIZE \
    --bind-to none \
    -x MASTER_ADDR=$MASTER_ADDR \
    -x MASTER_PORT=$MASTER_PORT \
    -x LD_LIBRARY_PATH \
    -x HF_HOME \
    -x TOKENIZER_MODEL \
    -x DATA_PATH \
    -x CKPT_PATH \
    bash -c '
        export RANK=$OMPI_COMM_WORLD_RANK
        export WORLD_SIZE=$OMPI_COMM_WORLD_SIZE
        export LOCAL_RANK=$OMPI_COMM_WORLD_LOCAL_RANK
        eval "$(/apps/t4/rhel9/free/miniconda/24.1.2/bin/conda shell.bash hook)"
        conda activate py312
        python src/pretrain_gpt.py \
            --tokenizer-model $TOKENIZER_MODEL \
            --data-path $DATA_PATH \
            --save $CKPT_PATH \
            --load $CKPT_PATH \
            --use-mcore-models \
            --disable-bias-linear \
            --seq-length 128 \
            --max-position-embeddings 40960 \
            --num-layers 48 \
            --hidden-size 2048 \
            --ffn-hidden-size 6144 \
            --num-attention-heads 32 \
            --kv-channels 128 \
            --group-query-attention \
            --num-query-groups 4 \
            --qk-layernorm \
            --init-method-std 0.02 \
            --attention-dropout 0.0 \
            --hidden-dropout 0.0 \
            --norm-epsilon 1e-6 \
            --normalization RMSNorm \
            --position-embedding-type rope \
            --swiglu \
            --untie-embeddings-and-output-weights \
            --no-masked-softmax-fusion \
            --no-position-embedding \
            --rotary-base 1000000 \
            --num-experts 128 \
            --moe-ffn-hidden-size 768 \
            --moe-router-topk 8 \
            --moe-router-load-balancing-type global_aux_loss \
            --moe-aux-loss-coeff 1e-2 \
            --moe-grouped-gemm \
            --moe-token-dispatcher-type alltoall \
            --moe-router-dtype fp32 \
            --moe-z-loss-coeff 1e-3 \
            --overlap-param-gather \
            --overlap-grad-reduce \
            --tokenizer-type HuggingFaceTokenizer \
            --num-workers 16 \
            --split 100,0,0 \
            --micro-batch-size 64 \
            --global-batch-size 16384 \
            --lr 3e-4 \
            --min-lr 3.0e-5 \
            --train-iters 15000 \
            --lr-decay-style WSD \
            --lr-wsd-decay-style linear \
            --lr-wsd-decay-iters 5000 \
            --lr-warmup-iters 100 \
            --weight-decay 0.1 \
            --adam-beta1 0.9 \
            --adam-beta2 0.95 \
            --clip-grad 0.5 \
            --bf16 \
            --expert-model-parallel-size 4 \
            --use-distributed-optimizer \
            --log-interval 10 \
            --save-interval 1000 \
            --eval-interval 1000 \
            --eval-iters 1000 \
            --no-load-optim \
            --no-load-rng \
            --no-gradient-accumulation-fusion
    '