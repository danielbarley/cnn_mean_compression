#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=rivulet
#SBATCH --output=jobout18
#SBATCH --job-name=CompCNN18
#SBATCH --cpus-per-task=32
#SBATCH --mem=24G

# load appropriate conda paths, because we are not in a login shell
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate torch

python train_cosine.py --silent --data_dir=/local/dbarley/imagenet --arch=resnet18 --batch_size=64 --num_epochs=90 --weight_decay=0.0001 --lr=0.1 --warmup_epochs=0 --cooldown_epochs=0 --sched=step --decay_epochs=30 --momentum=0.9 --num_workers=32
