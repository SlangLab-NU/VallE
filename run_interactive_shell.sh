module load singularity/3.10.3

export VALLE_ROOT=/scratch/lewis.jor
export VALLE_REPO_ROOT=$VALLE_ROOT/VallE
export SINGULARITYENV_PYTHONPATH="$VALLE_REPO_ROOT:/workspace/icefall:$PYTHONPATH"
export singularity_image=$VALLE_REPO_ROOT/concat_speakers_on_dev_set.sif

singularity shell --nv --bind $VALLE_REPO_ROOT:$VALLE_REPO_ROOT \
                        --bind /work/van-speech-nlp/UASpeech:/scratch/lewis.jor/UASpeech $singularity_image
