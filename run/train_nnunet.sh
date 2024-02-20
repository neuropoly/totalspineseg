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
# Get the number of CPUs, subtract 1, and ensure the value is at least 1
JOBS_FOR_CPUS=$(( $(($(nproc) - 1 < 1 ? 1 : $(nproc) - 1 )) ))
# Get the total memory in GB divided by RAM_REQUIREMENT, rounded down to nearest integer, and ensure the value is at least 1
JOBS_FOR_RAMGB=$(( $(awk -v ram_req="$RAM_REQUIREMENT" '/MemTotal/ {print int($2/1024/1024/ram_req < 1 ? 1 : $2/1024/1024/ram_req)}' /proc/meminfo) ))
# Get the minimum of JOBS_FOR_CPUS and JOBS_FOR_RAMGB
JOBS=$(( JOBS_FOR_CPUS < JOBS_FOR_RAMGB ? JOBS_FOR_CPUS : JOBS_FOR_RAMGB ))

export nnUNet_def_n_proc=$JOBS
export nnUNet_n_proc_DA=$JOBS

# Set nnunet params
export nnUNet_raw=data/nnUNet/nnUNet_raw
export nnUNet_preprocessed=data/nnUNet/nnUNet_preprocessed
export nnUNet_results=data/nnUNet/nnUNet_results

for d in 206 210; do
    # Preprocess
    nnUNetv2_plan_and_preprocess -d $d -c 3d_fullres -npfp $JOBS -np $JOBS --verify_dataset_integrity
    
    # Train with 5 fold cross-validation
    for f in {0..4}; do
        nnUNetv2_train $d 3d_fullres $f --npz
    done

    # Find best configuration
    nnUNetv2_find_best_configuration $d -np $JOBS -c 3d_fullres -f $f
done
