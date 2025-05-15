#!/bin/sh
#SBATCH --job-name=miniLLM
#SBATCH --gres=gpu:v100l:4
#SBATCH --qos=m3
#SBATCH --time=10:00:00
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH --output=/scratch/gaeunyim/minillm/results/gpt2/train/logs_760_to_760_3.out 
#SBATCH --error=/scratch/gaeunyim/minillm/results/gpt2/train/errors_760_to_760_3.out 

#! /bin/bash

export CUDA_HOME=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Core/cudacore/11.8.0
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

MASTER_ADDR=localhost
MASTER_PORT=${2-2012}
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=${3-4}

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT \
                  --tee 3"

# model
BASE_PATH=${1-"/home/gaeunyim/LMOps/minillm"}
TEACHER_SIZE="760M"
STUDENT_SIZE="760M"
INDEX_EXP="3"
CKPT_NAME="init-gpt2-${STUDENT_SIZE}"
CKPT="/scratch/gaeunyim/minillm/checkpoints/${CKPT_NAME}"
TEACHER_CKPT_NAME="minillm-gpt2-${TEACHER_SIZE}"
TEACHER_CKPT="/scratch/gaeunyim/minillm/checkpoints/${TEACHER_CKPT_NAME}"
# data
PROMPT_DATA_DIR="/scratch/gaeunyim/minillm/processed_data/dolly/prompt/gpt2/"
LM_DATA_DIR="/scratch/gaeunyim/minillm/processed_data/openwebtext/gpt2/512/10M/"
# runtime
DISTILLED_NAME="distilled-${TEACHER_SIZE}-to-${STUDENT_SIZE}_${INDEX_EXP}"
SAVE_PATH="/scratch/gaeunyim/minillm/results/gpt2/ckpt/${DISTILLED_NAME}"
# hp
GRAD_ACC=2
BATCH_SIZE=2
CHUNK_SIZE=8


OPTS=""
# model
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-path ${CKPT}"
OPTS+=" --teacher-model-path ${TEACHER_CKPT}"
OPTS+=" --ckpt-name ${CKPT_NAME}"
OPTS+=" --teacher-ckpt-name ${TEACHER_CKPT_NAME}"
OPTS+=" --n-gpu ${GPUS_PER_NODE}"
OPTS+=" --n-nodes ${NNODES}"
OPTS+=" --teacher-model-fp16"
# OPTS+=" --gradient-checkpointing"
# data
OPTS+=" --prompt-data-dir ${PROMPT_DATA_DIR}"
OPTS+=" --lm-data-dir ${LM_DATA_DIR}"
OPTS+=" --dev-num 1000"
OPTS+=" --num-workers 0"
# hp
OPTS+=" --epochs 10"
OPTS+=" --total-iters 5000"
OPTS+=" --kd-ratio 0.5"
OPTS+=" --batch-size ${BATCH_SIZE}"
OPTS+=" --lr 5e-6"
OPTS+=" --lr-min 5e-6"
OPTS+=" --gradient-accumulation-steps ${GRAD_ACC}"
OPTS+=" --max-length 512"
OPTS+=" --max-prompt-length 256"
OPTS+=" --warmup-iters 100"
# runtime
OPTS+=" --save ${SAVE_PATH}"
OPTS+=" --seed 10"
OPTS+=" --seed-ppo 42"
OPTS+=" --seed-lm 7"
OPTS+=" --save-interval 1000"
OPTS+=" --eval-interval 500"
OPTS+=" --log-interval 16"
OPTS+=" --mid-log-num 1"
# ppo
OPTS+=" --type minillm"
OPTS+=" --ppo-epochs 4"
OPTS+=" --num-rollouts 256"
OPTS+=" --chunk-size ${CHUNK_SIZE}"
# minillm
OPTS+=" --length-norm"
OPTS+=" --single-step-reg"
OPTS+=" --teacher-mixed-alpha 0.2"
# reward
OPTS+=" --reward-scaling 0.5"
OPTS+=" --cliprange-reward 100"
# gen
OPTS+=" --do-sample"
OPTS+=" --top-k 0"
OPTS+=" --top-p 1.0"
OPTS+=" --temperature 1.0"
# deepspeed
OPTS+=" --deepspeed"
OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config_zero1_fp16.json"

export NCCL_DEBUG=""
export WANDB_DISABLED=True
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH=${BASE_PATH}
CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/train_minillm.py ${OPTS} $@"

echo ${CMD}
echo "PYTHONPATH=${PYTHONPATH}"
mkdir -p ${SAVE_PATH}
${CMD}
