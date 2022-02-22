#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100l:1
#SBATCH --mem=12700M
#SBATCH --account=def-ebrahimi
#SBATCH --time=00-40:00            # time (DD-HH:MM)
#SBATCH --job-name=cuhk_to_all3
#SBATCH --output=outs/%x-%j.out
module load python/3.7.9
source ~/KD_REID_ENV/bin/activate

# Prepare data
DATA_DIR=${SLURM_TMPDIR}/data
mkdir $DATA_DIR
tar xf ~/reid-data.tar.gz -C $DATA_DIR

PYTHONPATH=$PYTHONPATH:$PWD python cuhk_to_all3.py
