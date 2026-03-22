#!/bin/bash

#$ -cwd                      ## Execute a job in the current directory
#$ -l node_q=1               ## Use number of node
#$ -l h_rt=00:30:00          ## Running job time
#$ -j y                      ## Integrate standard error output into a standard output
#$ -p -5

module load cuda/12.8.0
module load cudnn/9.8.0
module load nccl/2.20.5
module load miniconda

export CUDA_HOME=/apps/t4/rhel9/cuda/12.8.0
export PATH=$CUDA_HOME/bin:$PATH
export CPATH=$CUDA_HOME/include:$CPATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

eval "$(/apps/t4/rhel9/free/miniconda/24.1.2/bin/conda shell.bash hook)"
conda create -n py312 --file spec-file.txt
conda activate py312
pip install --no-build-isolation --no-cache-dir -r requirements.txt