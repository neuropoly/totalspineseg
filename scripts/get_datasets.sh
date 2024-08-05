#!/bin/bash

# This script get the datasets require to train the model from:
#   git@data.neuro.polymtl.ca:datasets/whole-spine.git
#   git@data.neuro.polymtl.ca:datasets/spider-challenge-2023.git
#   git@github.com:spine-generic/data-multi-subject.git
#   git@github.com:spine-generic/data-single-subject.git

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

# set TOTALSPINESEG and TOTALSPINESEG_DATA if not set
TOTALSPINESEG="$(realpath "${TOTALSPINESEG:-totalspineseg}")"
TOTALSPINESEG_DATA="$(realpath "${TOTALSPINESEG_DATA:-data}")"

# Set the paths to the BIDS data folders
bids="$TOTALSPINESEG_DATA"/bids

# Make sure $TOTALSPINESEG_DATA/bids exists and enter it
mkdir -p "$bids"
CURR_DIR="$(realpath .)"
cd "$bids"

datasets=(
    git@data.neuro.polymtl.ca:datasets/whole-spine.git
    git@data.neuro.polymtl.ca:datasets/spider-challenge-2023.git
    git@github.com:spine-generic/data-multi-subject.git
    git@github.com:spine-generic/data-single-subject.git
)

# Loop over datasets and download them
for ds in ${datasets[@]}; do
    dsn=$(basename $ds .git)

    # Clone the dataset from the specified repository
    git clone $ds

    # Enter the dataset directory
    cd $dsn

    # Remove all files and folders not in this formats:
    #   .*
    #   sub-*/anat/sub-*_{T1,T2,T2star,MTS}.nii.gz
    #   derivatives/labels_iso/sub-*/anat/sub-*_{T1w,T2w,T2star,MTS}_space-resampled_{label-spine_dseg,label-SC_seg,label-canal_seg}.nii.gz
    find . ! -path '.' ! -path './.*' \
        ! -regex '^\./sub-[^/]*\(/anat\(/sub-[^/]*_\(T1w\|T2w\|T2star\|MTS\)\.nii\.gz\)?\)?$' \
        ! -regex '^\./derivatives\(/labels_iso\(/sub-[^/]*\(/anat\(/sub-[^/]*_\(T1w\|T2w\|T2star\|MTS\)_space-resampled_\(label-spine_dseg\|label-SC_seg\|label-canal_seg\)\.nii\.gz\)?\)?\)?\)?$' \
        -delete

    # Ddownload the necessary files from git-annex
    git annex get

    # Move back to the parent directory to process the next dataset
    cd ..
done

# Return to the original working directory
cd "$CURR_DIR"
