#!/bin/bash
#SBATCH --job-name=julia_serial
#SBATCH --output=array_job_out_%A_%a.txt
#SBATCH --error=array_job_err_%A_%a.txt
#SBATCH --account=project_2007347
#SBATCH --partition=small
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=6G
#SBATCH --array=1-300

module load julia
srun julia FDE_CSC.jl ${SLURM_ARRAY_TASK_ID}
