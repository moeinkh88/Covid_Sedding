#!/bin/bash
#SBATCH --job-name=julia_serial
#SBATCH --account=project_2007347
#SBATCH --partition=small
#SBATCH --time=15:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=20G

module load julia
srun julia Turing_ODE_CSC_SAfrica12.jl
