# LMOps-TA-Knowledge-Distillation

## 0. Project Overview
This project is a replication of the [MiniLLM](https://arxiv.org/abs/2306.08543) paper, which presents methodologies for maintaining model accuracy during the process of compressing a large language model into a smaller one. MiniLLM addresses the issues arising in the knowledge distillation process of sentence-generating large language models by introducing a novel objective function. Typically, when performing knowledge distillation from a teacher model to a student model, the student tends to overestimate the low-probability regions of the teacher's distribution. By adopting reverse knowledge distillation instead of the conventional forward approach, MiniLLM enables the student model to generate more accurate and higher-quality sentences.

In addition to replicating the training process of MiniLLM, this project further improves its performance by incorporating the [TAKD]((https://arxiv.org/abs/1902.03393)) (Teacher Assistant Knowledge Distillation) framework. One of the most critical challenges in knowledge distillation from a teacher model to a student model is the significant performance degradation due to the limited number of parameters in the student model compared to the teacher. To address this, the TAKD approach introduces an intermediate model, known as the Teacher Assistant (TA), which has fewer parameters than the teacher but more than the student. By using the TA model as a bridge, TAKD performs multi-stage distillation, which helps transfer knowledge more effectively and mitigates the capacity gap between teacher and student models.

## 1. Environmental Setup
You need to use DRAC (Digital Alliance Canada Cluster) to run this experiment. 
Usage of Digital Research Alliance Canada cluster, slurm script.
Download the dataset, and pre-existing checkpoints or original code from the paper, from the authors themselves of hugging face.


## 2. Replication of MiniLLM
1. Clone this repository in your own path.
```
git clone https://github.com/GAEUNYIM/LMOps-TA-Distillation.git
```
2. Train the student model using the following scripts. 
```
sbatch {BASE_PATH}/LMOps-TA-Distillation/MiniLLM/scripts/gpt2/ours/ours_train_gpt2_{TEACHER_SIZE}_to_{STUDENT_SIZE}.sh
```
It will take at most 10 hours for each student models by using the following resources. Make sure that your script has proper option at the top of the bash file.
```
#!/bin/sh
#SBATCH --job-name=miniLLM
#SBATCH --gres=gpu:v100l:4
#SBATCH --qos=m3
#SBATCH --time=10:00:00
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH --output={BASE_PATH}/LMOps-TA-Distillation/MiniLLM/results/gpt2/train/gpt2_logs/{TEACHER_SIZE}_to_{STUDENT_SIZE}.out 
#SBATCH --error=/home/gaeunyim/LMOps/minillm/results/gpt2/train/gpt2_logs/error/{TEACHER_SIZE}_to_{STUDENT_SIZE}.out
```
3. You can see the checkpoint file after finishing the training
```
cd {BASE_PATH}/results/gpt2/train/minillm
```
4. Evaluate the student model 
```
sbatch {BASE_PATH}/LMOps-TA-Distillation/MiniLLM/scripts/gpt2/eval/eval_main_dolly_{TEACHER_SIZE}_to_{STUDENT_SIZE}.sh
```

## 3. Application of TAKD on MiniLLM

## 4. Results
Replication results

## 5. Analysis
Provide insightful explanations, any differences

## 6. Discussion
