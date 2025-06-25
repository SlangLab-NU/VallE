#!/bin/bash

# SLURM job parameters
#SBATCH --job-name=prepare_libritts
#SBATCH --output=/scratch/lewis.jor/VallE/egs/libritts/logs/%j_output.log
#SBATCH --error=/scratch/lewis.jor/VallE/egs/libritts/logs/%j_error.log
#SBATCH --constraint=ib
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100-sxm2
#SBATCH --mem=15G
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=lewis.jor@northeastern.edu

# Load required modules
module load singularity

# Set up environment variables
valle_root=/scratch/lewis.jor/VallE
cd $valle_root/egs/libritts
singularity_image=/work/van-speech-nlp/valle_container/valle.sif

export SINGULARITYENV_PYTHONPATH="/workspace/icefall:$PYTHONPATH"

# Run training script within Singularity container
singularity run --nv --bind $valle_root:$valle_root $singularity_image \
    bash prepare.sh --stage -1 --stop-stage 3

#    python3 bin/trainer.py --max-duration 80 --filter-min-duration 0.5 --filter-max-duration 14 --train-stage 1 \
#      --num-buckets 6 --dtype "float16" --save-every-n 10000 --valid-interval 20000 \
#      --model-name valle --share-embedding true --norm-first true --add-prenet false \
#      --decoder-dim 1024 --nhead 16 --num-decoder-layers 12 --prefix-mode 1 \
#      --base-lr 0.05 --warmup-steps 200 --average-period 0 \
#      --num-epochs 20 --start-epoch 1 --start-batch 0 --accumulate-grad-steps 4 \
#      --exp-dir $exp_dir
