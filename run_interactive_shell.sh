# module load apptainer

export PYTHONPATH=""
export VALLE_ROOT=/scratch/lewis.jor
export VALLE_REPO_ROOT=$VALLE_ROOT/VallE
export APPTAINERENV_PYTHONPATH="$VALLE_REPO_ROOT:/workspace/icefall:$PYTHONPATH"
export apptainer_image=$VALLE_REPO_ROOT/valle_container.sif

apptainer shell --nv --bind $VALLE_REPO_ROOT:$VALLE_REPO_ROOT \
                        --bind /projects/van-speech-nlp:/scratch/lewis.jor $apptainer_image
