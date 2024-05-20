#!/bin/bash

# This script get the datasets require to train the model from:
#   git@data.neuro.polymtl.ca:datasets/whole-spine.git
#   git@data.neuro.polymtl.ca:datasets/spider-challenge-2023.git
#   git@github.com:spine-generic/data-multi-subject.git
#   git@github.com:spine-generic/data-single-subject

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


# Set the paths to the BIDS data folders
bids=data/bids

# Make sure data/bids exists and enter it
mkdir -p $bids
cd $bids

datasets=(
    git@data.neuro.polymtl.ca:datasets/whole-spine.git
    git@data.neuro.polymtl.ca:datasets/spider-challenge-2023.git
    git@github.com:spine-generic/data-multi-subject.git
    git@github.com:spine-generic/data-single-subject
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
    #   sub-*/anat/sub-*_{T1,T2}w.nii.gz
    #   derivatives/labels/sub-*/anat/sub-*_{T1,T2}w_{label-spine_dseg,label-SC_seg,label-canal_seg}.nii.gz
    find . ! -path '.' ! -path './.*' \
        ! -regex '^\./sub-[^/]*\(/anat\(/sub-[^/]*_\(T1\|T2\)w\.nii\.gz\)?\)?$' \
        ! -regex '^\./derivatives\(/labels\(/sub-[^/]*\(/anat\(/sub-[^/]*_\(T1\|T2\)w_\(label-spine_dseg\|label-SC_seg\|label-canal_seg\)\.nii\.gz\)?\)?\)?\)?$' \
        -delete

    # Initialize the current dataset directory as a git-annex repository and download the necessary files
    git annex init
    git annex get

    # Move back to the parent directory to process the next dataset
    cd ..
done

# Return to the original working directory
cd ../..
