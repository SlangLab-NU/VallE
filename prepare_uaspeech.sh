#!/bin/bash

# SLURM job parameters
#SBATCH --job-name=prepare_uaspeech
#SBATCH --output=/scratch/lewis.jor/logs/prepare_uaspeech_tts_%j_output.log
#SBATCH --error=/scratch/lewis.jor/logs/prepare_uaspeech_tts_%j_error.log
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
cd $valle_root/egs/uaspeech
singularity_image=concat_speakers_on_dev_set.sif


#echo "Copying UASpeech"
#mkdir -p $valle_root/egs/uspeech/UASpeech
#rsync -av --progress /work/van-speech-nlp/UASpeech $valle_root/egs/uspeech/UASpeech

echo "Running data preparation"
export SINGULARITYENV_PYTHONPATH="/workspace/icefall:$PYTHONPATH"

# Run training script within Singularity container
singularity exec --nv \
  --bind $valle_root:$valle_root \
  --bind /work/van-speech-nlp/UASpeech:/scratch/lewis.jor/UASpeech \
  $singularity_image \
    bash prepare.sh --stage -1 --stop-stage 2 --prep-tts 1 --control-tts 0 --atypical-tts 1

#    python3 bin/trainer.py --max-duration 80 --filter-min-duration 0.5 --filter-max-duration 14 --train-stage 1 \
#      --num-buckets 6 --dtype "float16" --save-every-n 10000 --valid-interval 20000 \
#      --model-name valle --share-embedding true --norm-first true --add-prenet false \
#      --decoder-dim 1024 --nhead 16 --num-decoder-layers 12 --prefix-mode 1 \
#      --base-lr 0.05 --warmup-steps 200 --average-period 0 \
#      --num-epochs 20 --start-epoch 1 --start-batch 0 --accumulate-grad-steps 4 \
#      --exp-dir $exp_dir
