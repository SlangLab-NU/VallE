# module load apptainer

export PYTHONPATH=""
export VALLE_ROOT=/scratch/lewis.jor
export VALLE_REPO_ROOT=$VALLE_ROOT/VallE
export APPTAINERENV_PYTHONPATH="$VALLE_REPO_ROOT:/workspace/icefall:$PYTHONPATH"
export apptainer_image=$VALLE_REPO_ROOT/valle_v100.sif

apptainer shell --nv --bind $VALLE_REPO_ROOT:$VALLE_REPO_ROOT \
                        --bind /projects/van-speech-nlp/UASpeech:/scratch/lewis.jor/UASpeech $apptainer_image
