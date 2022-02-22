#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100l:1
#SBATCH --mem=12700M
#SBATCH --account=def-granger
#SBATCH --time=00-40:00            # time (DD-HH:MM)
#SBATCH --job-name=resume_training
#SBATCH --output=outs/%x-%j.out
module load python/3.6
source ~/kd_reid_env/bin/activate

PYTHONPATH=$PYTHONPATH:$PWD python resume_training.py
