#!/bin/bash

# This script train the TotalSegMRI nnUNet models.

# BASH SETTINGS
# ======================================================================================================================

# Uncomment for full verbose
# set -v

# Immediately exit if error
set -e

# Exit if user presses CTRL+C (Linux) or CMD+C (OSX)
trap "echo Caught Keyboard Interrupt within script. Exiting now.; exit" INT

# SCRIPT STARTS HERE
# ======================================================================================================================

# RAM requirement in GB
RAM_REQUIREMENT=8
# Get the number of CPUs, subtract some for system processes
LEAVE_CPUS=1
JOBS_FOR_CPUS=$(( $(($(lscpu -p | egrep -v '^#' | wc -l) - $LEAVE_CPUS < 1 ? 1 : $(lscpu -p | egrep -v '^#' | wc -l) - $LEAVE_CPUS )) ))
# Get the total memory in GB divided by RAM_REQUIREMENT, rounded down to nearest integer, and ensure the value is at least 1
JOBS_FOR_RAMGB=$(( $(awk -v ram_req="$RAM_REQUIREMENT" '/MemTotal/ {print int($2/1024/1024/ram_req < 1 ? 1 : $2/1024/1024/ram_req)}' /proc/meminfo) ))
# Get the minimum of JOBS_FOR_CPUS and JOBS_FOR_RAMGB
JOBS=$(( JOBS_FOR_CPUS < JOBS_FOR_RAMGB ? JOBS_FOR_CPUS : JOBS_FOR_RAMGB ))

export nnUNet_def_n_proc=$JOBS
export nnUNet_n_proc_DA=$JOBS

# Set nnunet params
export nnUNet_raw=data/nnUNet/raw
export nnUNet_preprocessed=data/nnUNet/preprocessed
export nnUNet_results=data/nnUNet/results
export nnUNet_tests=data/nnUNet/tests
export nnUNet_exports=data/nnUNet/exports

nnUNetTrainer=nnUNetTrainer_8000epochs

# Set the datasets to work with - default is 101 102 103
DATASETS=${1:-101 102 103}

# Set the fold to work with - default is 0
FOLD=${2:-0}

echo "Working with datasets: ${DATASETS[@]}, fold: $FOLD"

for d in ${DATASETS[@]}; do

    # Get the dataset name
    d_name=$(basename $(ls -d $nnUNet_raw/Dataset${d}_TotalSegMRI*))
    
    if [ ! -d $nnUNet_preprocessed/$d_name ]; then
        echo "Preprocess dataset $d_name"
        nnUNetv2_plan_and_preprocess -d $d -c 3d_fullres -npfp $JOBS -np $JOBS --verify_dataset_integrity
    fi
    
    if [ ! -d $nnUNet_results/$d_name/${nnUNetTrainer}__nnUNetPlans__3d_fullres/fold_$FOLD ]; then
        echo "Train nnUNet model for dataset $d_name"
        nnUNetv2_train $d 3d_fullres $FOLD -tr $nnUNetTrainer --npz
    else
        echo "Continue training nnUNet model for dataset $d_name"
        nnUNetv2_train $d 3d_fullres $FOLD -tr $nnUNetTrainer --npz --c
    fi

    echo "Export the model for dataset $d_name in $nnUNet_exports"
    mkdir -p $nnUNet_exports
    mkdir -p $nnUNet_results/$d_name/ensembles
    nnUNetv2_export_model_to_zip -d $d -o $nnUNet_exports/${d_name}_fold_$FOLD.zip -c 3d_fullres -f $FOLD -tr $nnUNetTrainer

    echo "Testing nnUNet model for dataset $d_name"
    mkdir -p $nnUNet_tests/${d_name}_fold_$FOLD
    nnUNetv2_predict -d $d -i $nnUNet_raw/$d_name/imagesTs -o $nnUNet_tests/${d_name}_fold_$FOLD -f $FOLD -c 3d_fullres -tr $nnUNetTrainer -npp $JOBS -nps $JOBS
    nnUNetv2_evaluate_folder $nnUNet_raw/$d_name/labelsTs $nnUNet_tests/${d_name}_fold_$FOLD -djfile $nnUNet_results/$d_name/${nnUNetTrainer}__nnUNetPlans__3d_fullres/dataset.json -pfile $nnUNet_results/$d_name/${nnUNetTrainer}__nnUNetPlans__3d_fullres/plans.json -np $JOBS

done
