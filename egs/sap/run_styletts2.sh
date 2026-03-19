
#!/bin/bash


export PYTHONPATH=""
export apptainer_image=sap_analysis_styletts2.sif

apptainer shell --nv \
        --bind /scratch/lewis.jor:/scratch/lewis.jor \
       	$apptainer_image

