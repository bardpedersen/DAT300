#!/bin/bash
#SBATCH --ntasks=8           # 8 cores (CPU)
#SBATCH --nodes=1            # Use 1 node
#SBATCH --job-name=dt3_test  # Name of job
#SBATCH --partition=gpu      # Use GPU partition
#SBATCH --gres=gpu:1         # Use one GPUs
#SBATCH --mem=64G            # Default memory per CPU is 3GB
#SBATCH --output=../output_logs/training_prints_%j.out # Stdout file   

## Script commands
module load singularity

SIFFILE="$HOME/singularity_container/container_dat300_h23.sif" 

## RUN THE PYTHON SCRIPT
# Using a singularity container named container_u_net.sif
singularity exec --nv $SIFFILE python test_script.py        

# Send this job into the slurm queue with the following command: 
# >> sbatch test_script_slurm.sh
