#!/bin/sh
#SBATCH --job-name=MiniLLM
#SBATCH --gres=gpu:a100:4
#SBATCH --qos=m3
#SBATCH --time=1:00:00
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --output=/scratch/gaeunyim/LMOps-TA-Distillation/MiniLLM/minillm/results/opt/eval_main/dolly-512/distilled-opt-2.7B-to-1.3B_3/eval_1.3B.out 
#SBATCH --error=/scratch/gaeunyim/LMOps-TA-Distillation/MiniLLM/minillm/results/opt/eval_main/dolly-512/distilled-opt-2.7B-to-1.3B_3/errors/eval_1.3B.out

#! /bin/bash
source /home/gaeunyim/GAEUN/bin/activate

export CUDA_HOME=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Core/cudacore/11.8.0
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

MASTER_ADDR=localhost
MASTER_PORT=${2-2113}
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=${3-4}

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

# model
BASE_PATH=${1-"/scratch/gaeunyim/LMOps-TA-Distillation/MiniLLM/minillm"}
TEACHER_SIZE=2.7B
STUDENT_SIZE=1.3B
EXP_ITER=1
CKPT_NAME=${4-"distilled-opt-${TEACHER_SIZE}-${STUDENT_SIZE}_${EXP_ITER}"}
CKPT="${BASE_PATH}/checkpoints/${CKPT_NAME}"
# data
DATA_NAMES="dolly"
DATA_DIR="${BASE_PATH}/data/dolly"
# hp
EVAL_BATCH_SIZE=16
# runtime
SAVE_PATH="${BASE_PATH}/results/gpt2/eval_main/"
TYPE="eval_main"

OPTS=""
# model
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-path ${CKPT}"
OPTS+=" --ckpt-name ${CKPT_NAME}"
OPTS+=" --n-gpu ${GPUS_PER_NODE}"
OPTS+=" --model-type gpt2"
# data
OPTS+=" --data-dir ${DATA_DIR}"
OPTS+=" --data-names ${DATA_NAMES}"
OPTS+=" --num-workers 0"
OPTS+=" --dev-num -1"
OPTS+=" --data-process-workers -1"
OPTS+=" --json-data"
# hp
OPTS+=" --eval-batch-size ${EVAL_BATCH_SIZE}"
OPTS+=" --max-length 512"
OPTS+=" --max-prompt-length 256"
# runtime
OPTS+=" --do-eval"
OPTS+=" --save ${SAVE_PATH}"
OPTS+=" --seed 10"
# deepspeed
OPTS+=" --deepspeed"
OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config_zero1_fp16.json"
OPTS+=" --type ${TYPE}"
# gen
OPTS+=" --do-sample"
OPTS+=" --top-k 0"
OPTS+=" --top-p 1.0"
OPTS+=" --temperature 1.0"

export NCCL_DEBUG=""
export TOKENIZERS_PARALLELISM=false
export PYTHONIOENCODING=utf-8
export PYTHONPATH=${BASE_PATH}

echo ">>> WHICH PYTHON:"
which python
python -c "import pyarrow as pa; print('✅ pyarrow version:', pa.__version__)"
CMD="/home/gaeunyim/GAEUN/bin/torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/perplexity_evaluate.py ${OPTS} $@"

echo ${CMD}
echo "PYTHONPATH=${PYTHONPATH}"
mkdir -p ${SAVE_PATH}
${CMD}
