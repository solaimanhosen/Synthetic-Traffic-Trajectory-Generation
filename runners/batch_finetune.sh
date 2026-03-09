#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=3-00:00:00
#SBATCH --mem=128G
#SBATCH --gres=gpu:a100:1
#SBATCH --account=meisam-lab
#SBATCH --job-name=ngsim_finetune
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# ACTIVATE VENV
source /lustre/hdd/LAS/meisam-lab/hosen/python_venvs/ngsim_venv/bin/activate

# DIAGNOSTICS
echo "Running on node(s): $SLURM_NODELIST"
echo "Number of GPUs allocated: $SLURM_GPUS_ON_NODE"
nvidia-smi
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Working directory: $(pwd)"
echo "Job started at: $(date)"

# RUN PROGRAM
python finetune.py \
    --train_file data/ngsim_rowbyrow_train.jsonl \
    --eval_file  data/ngsim_rowbyrow_eval.jsonl \
    --output_dir ./results \
    --model_save_dir ./ngsim_model \
    --num_train_epochs 1 \
    --learning_rate 2e-5 \
    --warmup_steps 1500 \
    --eval_steps 1000 \
    --save_steps 1000 \
    --max_eval_samples 100 \
    --resume_from_checkpoint

echo "Job completed at: $(date)"