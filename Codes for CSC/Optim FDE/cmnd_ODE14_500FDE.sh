#!/bin/bash
#SBATCH --job-name=julia_serial
#SBATCH --output=array_job_out_%A_%a.txt
#SBATCH --error=array_job_err_%A_%a.txt
#SBATCH --account=project_2007347
#SBATCH --partition=small
#SBATCH --time=6:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=8G
#SBATCH --array=1-500

module load julia
srun julia Optim_FDE_CSC_Birth_ODE14_E0_500.jl ${SLURM_ARRAY_TASK_ID}
