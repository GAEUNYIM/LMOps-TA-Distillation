# LMOps-TA-Distillation

## 0. Project Overview
This project is a replication of the [MiniLLM](https://arxiv.org/abs/2306.08543) paper, which presents methodologies for maintaining model accuracy during the process of compressing a large language model into a smaller one. MiniLLM addresses the issues arising in the knowledge distillation process of sentence-generating large language models by introducing a novel objective function. Typically, when performing knowledge distillation from a teacher model to a student model, the student tends to overestimate the low-probability regions of the teacher's distribution. By adopting reverse knowledge distillation instead of the conventional forward approach, MiniLLM enables the student model to generate more accurate and higher-quality sentences.

In addition to replicating the training process of MiniLLM, this project further improves its performance by incorporating the [TAKD](https://arxiv.org/abs/1902.03393) (Teacher Assistant Knowledge Distillation) framework. One of the most critical challenges in knowledge distillation from a teacher model to a student model is the significant performance degradation due to the limited number of parameters in the student model compared to the teacher. To address this, the TAKD approach introduces an intermediate model, known as the Teacher Assistant (TA), which has fewer parameters than the teacher but more than the student. By using the TA model as a bridge, TAKD performs multi-stage distillation, which helps transfer knowledge more effectively and mitigates the capacity gap between teacher and student models.

## Part 1. Environmental Setup
1. Connect to the server with GPUs.
We will use [DRAC](https://alliancecan.ca/en/search?keywords=ssh) (Digital Alliance Canada Cluster) to run this experiment. Create your account, and connect to server with ssh to use GPU by following the instruction of the website.
2. Create your own virtual environment in the server.
```
module load python/3.10
virtualenv --no-download {ENV_NAME}
source ~/{ENV_NAME}/bin/activate # ✅ TODO everyday: Always activate this env whenever you use the server!!
pip install --no-index --upgrade pip
```

## Part 2. Replication of MiniLLM (Teacher -> Student)
1. Clone this repository in your own path.
```
git clone https://github.com/GAEUNYIM/LMOps-TA-Distillation.git
```
(Prerequisite) Download the dataset, and pre-existing checkpoints ([hugging face](https://huggingface.co/MiniLLM)).
```
# dataset
huggingface-cli download MiniLLM/dolly --repo-type dataset --local-dir {BASE_PATH}/LMOps-TA-Distillation/data/dolly/

# ckpt of teacher model (1.2B)
huggingface-cli download gpt2-large --repo-type model --local-dir {BASE_PATH}/LMOps-TA-Distillation/checkpoints/gpt2-large

# init of student model (120M)
huggingface-cli download MiniLLM/init-gpt2-120M --repo-type model --local-dir {BASE_PATH}/LMOps-TA-Distillation/checkpoints/init-gpt2-120M
```
2. Train the student model using the following scripts. 
```
sbatch {BASE_PATH}/LMOps-TA-Distillation/MiniLLM/scripts/gpt2/ours/ours_train_gpt2_{TEACHER_SIZE}_to_{STUDENT_SIZE}.sh
```
Note: Make sure that your script has proper option at the top of the bash file. It will take at most 10 hours for each student models by using the following resources. 
```
#!/bin/sh
#SBATCH --job-name=miniLLM
#SBATCH --gres=gpu:v100l:4
#SBATCH --qos=m3
#SBATCH --time=10:00:00
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH --output={BASE_PATH}/LMOps-TA-Distillation/MiniLLM/results/gpt2/train/gpt2_logs/{TEACHER_SIZE}_to_{STUDENT_SIZE}.out 
#SBATCH --error={BASE_PATH}/LMOps-TA-Distillation/MiniLLM/results/gpt2/train/gpt2_logs/error/{TEACHER_SIZE}_to_{STUDENT_SIZE}.out
```
3. You can see the checkpoint file after finishing the training.
```
cd {BASE_PATH}/results/gpt2/train/minillm
```
4. Evaluate the student model.
```
sbatch {BASE_PATH}/LMOps-TA-Distillation/MiniLLM/scripts/gpt2/eval/eval_main_dolly_{TEACHER_SIZE}_to_{STUDENT_SIZE}.sh
```

## Part 3. Application of MiniLLM with TAKD (Teacher -> TA -> Student)
In this part, you will train student models with TA models, but not teacher model. TA model will also be trained from teacher model.
(Prerequisite) Download the pre-existing checkpoints.
```
# ckpt of TA model (760M, 340M)
huggingface-cli download MiniLLM/MiniLLM-gpt2-760M --repo-type model --local-dir {BASE_PATH}/LMOps-TA-Distillation/checkpoints/gpt2-760M
huggingface-cli download MiniLLM/MiniLLM-gpt2-340M --repo-type model --local-dir {BASE_PATH}/LMOps-TA-Distillation/checkpoints/gpt2-340M
```
1. Train the TA model using the following scripts. 
```
sbatch {BASE_PATH}/LMOps-TA-Distillation/MiniLLM/scripts/gpt2/ours/ours_train_gpt2_{TEACHER_SIZE}_to_{TA_SIZE}.sh
```
2. You can see the checkpoint file after finishing the training.
```
cd {BASE_PATH}/results/gpt2/train/minillm
```
3. Train the student model using the following scripts. 
```
sbatch {BASE_PATH}/LMOps-TA-Distillation/MiniLLM/scripts/gpt2/ours/ours_train_gpt2_{TA_SIZE}_to_{STUDENT_SIZE}.sh
``` 
4. Evaluate the student model.
```
sbatch {BASE_PATH}/LMOps-TA-Distillation/MiniLLM/scripts/gpt2/eval/eval_main_dolly_{TA_SIZE}_to_{STUDENT_SIZE}.sh
```
## Part 4. Results
The table below shows the experiment results of Part 2 and Part 3. 
- Experiment 2.A ~ 2.D showed similar results with the original paper ( +,- 0.2 of R-L Scores).
- Experiment 3.A ~ 3.D showed applying TA model between teacher model and student model can results better R-L socre than without applyint TA model.
  - 3.A showed that the student model of 340M results 26.3 R-L Score by distilling knowledge through one TA (760M) model, which is greater than the student model of 340M from 2.C showed 25.4 R-L Score with direct distillation.
  - 3.B showed that the student model of 340M results 27.2 R-L Score by distilling knowledge through one TA (760M) model, which is greater than the student model of 120M from 2.D showed 24.6 R-L Score with direct distillation.
  - 3.C showed that the student model of 340M results 27.0 R-L Score by distilling knowledge through one TA (340M) model, which is greater than the student model of 120M from 2.D showed 24.6 R-L Score with direct distillation.
  - 3.D showed that the student model of 120M results 24.1 R-L Score by distilling knowledge through two TA (760M, and 340M) models, which is less than the student model of 120M from 2.D showed 24.6 R-L Score with direct distillation.
```
| ExpID | Teacher Size | TA 1 Size | TA 2 Size | Student Size | R-L Score (paper) | R-L Score (experiment) |
|-------|--------------|-----------|-----------|--------------|-------------------| -----------------------|
| 2.A   | 1.5B         | -         | -         | -            | 27.6              | 27.4                   |
| 2.B   | 1.5B         | -         | -         | 760M         | 26.4              | 26.5                   |
| 2.C   | 1.5B         | -         | -         | 340M         | 25.4              | 25.2                   |
| 2.D   | 1.5B         | -         | -         | 120M         | 24.6              | 24.6                   |
| 3.A   | 1.5B         | 760M      | -         | 340M         | None              | 26.3                   |
| 3.B   | 1.5B         | 760M      | -         | 120M         | None              | 27.2                   |
| 3.C   | 1.5B         | 340M      | -         | 120M         | None              | 27.0                   |
| 3.D   | 1.5B         | 760M      | 340M      | 120M         | None              | 24.1                   |
```
## Part 5. Analysis
Provide insightful explanations, any differences

## Part 6. Discussion
