#!/bin/bash

# This script get the data require to train the model from https://github.com/spine-generic/ repository.

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

# Loop over two dataset directories: data-multi-subject and data-single-subject
for ds in data-multi-subject data-single-subject; do
    # Clone the dataset from the specified GitHub repository
    git clone https://github.com/spine-generic/$ds
    cd $ds

    # Remove code and derivatives directories which are not needed for the current task
    rm -rf code derivatives

    # Delete all files in the current directory (root of the dataset) except for those named 'sub-*'
    find . -mindepth 1 -maxdepth 1 -type f ! -name 'sub-*' -delete

    # Delete all non-anat directories inside each subject's folder
    find . -type d -regex '\./sub-[^/]+/.*' ! -regex '\./sub-[^/]+/anat' -exec rm -rf {} \; -prune
    
    # Within each subject's anat directory, delete all files except T1 and T2 weighted images (both .nii.gz and .json)
    find . -type f -regex '\./sub-[^/]+/anat/.*' ! -regex '\./sub-[^/]+/anat/sub-[^_]+_\(T1\|T2\)w\.\(nii\.gz\|json\)' -delete

    # Initialize the current dataset directory as a git-annex repository and download the necessary files
    git annex init
    git annex get

    # Move back to the parent directory to process the next dataset
    cd ..
done

# Return to the original working directory
cd ../..
