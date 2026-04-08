# module load apptainer

export PYTHONPATH=""
export VALLE_ROOT=/scratch/lewis.jor
export VALLE_REPO_ROOT=$VALLE_ROOT/VallE
export APPTAINERENV_PYTHONPATH="$VALLE_REPO_ROOT:/workspace/icefall:$PYTHONPATH"
export apptainer_image=$VALLE_REPO_ROOT/valle_container2.sif

apptainer shell --nv \
        --bind /scratch/lewis.jor:/scratch/lewis.jor \
       	$apptainer_image
