export CUDA_HOME=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Core/cudacore/11.8.0
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

#!/bin/sh
#SBATCH --job-name=miniLLM
#SBATCH --gres=gpu:v100l:1
#SBATCH --qos=m3
#SBATCH --time=3:00:00
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --output=../out/llava_lr5e-5.out 
#SBATCH --error=../out/error/llava_lr5e-5.out

BASE_PATH=${1}

export TF_CPP_MIN_LOG_LEVEL=3

# only prompt for MiniLLM train
PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/tools/process_data_dolly.py \
    --data-dir ${BASE_PATH}/data/dolly/ \
    --processed-data-dir ${BASE_PATH}/processed_data/dolly/prompt \
    --model-path ${BASE_PATH}/checkpoints/gpt2-large \
    --data-process-workers 32 \
    --max-prompt-length 256 \
    --dev-num 1000 \
    --only-prompt \
    --model-type gpt2

# prompt and response for baselines
PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/tools/process_data_dolly.py \
    --data-dir ${BASE_PATH}/data/dolly/ \
    --processed-data-dir ${BASE_PATH}/processed_data/dolly/full \
    --model-path ${BASE_PATH}/checkpoints/gpt2-large \
    --data-process-workers 32 \
    --max-prompt-length 256 \
    --dev-num 1000 \
    --model-type gpt2
