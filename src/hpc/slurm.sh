#!/bin/bash

#SBATCH --partition=gpu-a30 # Request a specific partition
#SBATCH --ntasks=1 # Number of tasks (see below)
#SBATCH --cpus-per-task=24 # Number of CPU cores per task
#SBATCH --nodes=1 # Ensure that all cores are on one machine
#SBATCH --time=0-06:00 # Runtime in D-HH:MM
#SBATCH --gres=gpu:2 # Optionally type and number of gpus
#SBATCH --mem=150G # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --output=hostname_%j.out # File to which STDOUT will be written
#SBATCH --error=hostname_%j.err # File to which STDERR will be written
#SBATCH --mail-type=END # Type of email notification - BEGIN,END,FAIL,ALL
#SBATCH --mail-user=username@student.uni-tuebingen.de # Email to which notifications will be sent

# Print info about current job - makes debug easy
scontrol show job $SLURM_JOB_ID

module purge
eval "$(conda shell.bash hook)"
conda activate env

# command
