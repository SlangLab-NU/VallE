#!/bin/bash

# SLURM job parameters
#SBATCH --job-name=infer_libritts
#SBATCH --output=/scratch/lewis.jor/VallE/egs/libritts/exp/libritts_train_4_16_0_1/logs/infer_%j_output.log
#SBATCH --error=/scratch/lewis.jor/VallE/egs/libritts/exp/libritts_train_4_16_0_1/logs/infer_%j_error.log
#SBATCH --constraint=ib
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100-sxm2
#SBATCH --mem=15G
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=aa.mohan@northeastern.edu

# Load required modules
module load singularity

# Set up environment variables
valle_root=/scratch/lewis.jor/VallE
output_dir=$valle_root/egs/libritts/output-libritts_16_8
# output_dir=/work/van-speech-nlp/expts/6_decoders/outputs
mkdir -p $output_dir
# checkpoint_dir=/work/van-speech-nlp/expts/6_decoders
checkpoint_dir=$valle_root/egs/libritts/exp/libritts_16_8
#/work/van-speech-nlp/aanchan/vall-e/egs/libritts/exp/tr_stable_orig_8_0_1
cd $valle_root/egs/libritts
singularity_image=/work/van-speech-nlp/valle_container/valle.sif
export SINGULARITYENV_PYTHONPATH="/workspace/icefall:$PYTHONPATH"


singularity run --nv --bind $valle_root:$valle_root $singularity_image \
    python3 bin/infer.py --output-dir $output_dir --checkpoint $checkpoint_dir/best-valid-loss.pt \
    --text-prompts "ADJACENT" \
    --audio-prompts ./prompts/prompts/CF02_B2_UW20_M3.wav \
    --text "Adjacent"
   # --text-prompts "KNOT one point one five miles per hour." \
   # --audio-prompts ./prompts/8463_294825_000043_000000.wav \
   # --text "To get up and running quickly just follow the steps below."

# Run training script within Singularity container
#singularity run --nv --bind $valle_root:$valle_root $singularity_image \
#    bash /work/van-speech-nlp/aanchan/vall-e/egs/libritts/prepare.sh --stage -1 --stop-stage 3

#    python3 bin/trainer.py --max-duration 80 --filter-min-duration 0.5 --filter-max-duration 14 --train-stage 1 \
#      --num-buckets 6 --dtype "float16" --save-every-n 10000 --valid-interval 20000 \
#      --model-name valle --share-embedding true --norm-first true --add-prenet false \
#      --decoder-dim 1024 --nhead 16 --num-decoder-layers 12 --prefix-mode 1 \
#      --base-lr 0.05 --warmup-steps 200 --average-period 0 \
#      --num-epochs 20 --start-epoch 1 --start-batch 0 --accumulate-grad-steps 4 \
#      --exp-dir $exp_dir
