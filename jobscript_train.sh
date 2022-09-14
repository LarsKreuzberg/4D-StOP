#!/bin/sh

#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1

python -u train_SemanticKitti.py