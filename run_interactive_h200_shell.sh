# module load apptainer

export PYTHONPATH=""
export VALLE_ROOT=/scratch/lewis.jor
export VALLE_REPO_ROOT=$VALLE_ROOT/VallE
export APPTAINERENV_PYTHONPATH="$VALLE_REPO_ROOT:/workspace/icefall:$PYTHONPATH"
export apptainer_image=$VALLE_REPO_ROOT/valle_h200_container.sif

# Point all cache dirs to writable scratch space
export APPTAINERENV_NUMBA_CACHE_DIR=$VALLE_ROOT/cache/numba
export APPTAINERENV_TRANSFORMERS_CACHE=$VALLE_ROOT/cache/huggingface
export APPTAINERENV_HF_HOME=$VALLE_ROOT/cache/huggingface
export APPTAINERENV_LHOTSE_CACHE_DIR=$VALLE_ROOT/cache/lhotse
export APPTAINERENV_TORCH_HOME=$VALLE_ROOT/cache/torch
export APPTAINERENV_MPLCONFIGDIR=$VALLE_ROOT/cache/matplotlib

mkdir -p $VALLE_ROOT/cache/numba
mkdir -p $VALLE_ROOT/cache/huggingface
mkdir -p $VALLE_ROOT/cache/lhotse
mkdir -p $VALLE_ROOT/cache/torch
mkdir -p $VALLE_ROOT/cache/matplotlib

apptainer shell --nv \
    --no-home \
    --bind /scratch/lewis.jor:/scratch/lewis.jor \
    $apptainer_image