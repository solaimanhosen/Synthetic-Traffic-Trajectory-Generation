#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=0-08:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --account=meisam-lab
#SBATCH --job-name=ngsim_preprocess
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# ACTIVATE VENV
source /lustre/hdd/LAS/meisam-lab/hosen/python_venvs/ngsim_venv/bin/activate

# DIAGNOSTICS (optional but useful)
echo "Running on node(s): $SLURM_NODELIST"
echo "Number of GPUs allocated: $SLURM_GPUS_ON_NODE"
nvidia-smi
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Working directory: $(pwd)"

# RUN PROGRAM
python preprocess.py > logs/ngsim_preprocess_$(date +%Y%m%d_%H%M%S).log

echo "Job completed successfully"
