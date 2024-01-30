#!/bin/bash

# This script prepares datasets for the TotalSegMRI model in BIDS structure.

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

# Check if the TotalSegmentator MRI folder exists
if [ ! -d totalsegmentator-mri ]; then
    echo "totalsegmentator-mri folder not found in current working directory. Exiting."
    echo "Please make sure you clone the repository into the current working directory."
    exit 0
fi

# Check if the SPIDER dataset images and masks are available
if [ ! -d data/src/spider/images ] || [ ! -d data/src/spider/masks ]; then
    echo "SPIDER dataset images or masks not found. Please make sure they are available in data/src/spider."
else
    echo "Preparing SPIDER dataset"

    echo "Convert images and masks from .mha to .nii.gz format"
    python totalsegmentator-mri/src/totalsegmri/utils/mha2nii.py -i data/src/spider/images -o data/bids/spider
    python totalsegmentator-mri/src/totalsegmri/utils/mha2nii.py -i data/src/spider/masks -o data/bids/spider/derivatives/labels

    echo "Map SPIDER labels to the labels used in this project"
    python totalsegmentator-mri/src/totalsegmri/utils/map_labels.py -m totalsegmentator-mri/src/totalsegmri/resources/labels_maps/spider.json -i data/bids/spider/derivatives/labels -o data/bids/spider/derivatives/labels

    echo "Rename files to follow BIDS naming conventions"
    # Add 'sub-' prefix to filenames
    for f in data/bids/spider/*.nii.gz; do mv $f ${f/spider\//spider\/sub-}; done
    for f in data/bids/spider/derivatives/labels/*.nii.gz; do mv $f ${f/labels\//labels\/sub-}; done

    # Replace 't1' with 'T1w' in filenames
    for f in data/bids/spider/*t1.nii.gz; do mv $f ${f/t1/T1w}; done
    for f in data/bids/spider/derivatives/labels/*t1.nii.gz; do mv $f ${f/t1/T1w}; done

    # Replace 't2' with 'T2w' in filenames
    for f in data/bids/spider/*t2.nii.gz; do mv $f ${f/t2/T2w}; done
    for f in data/bids/spider/derivatives/labels/*t2.nii.gz; do mv $f ${f/t2/T2w}; done

    # Replace 't2_SPACE' with 'T2Sw' in filenames
    for f in data/bids/spider/*t2_SPACE.nii.gz; do mv $f ${f/t2_SPACE/T2Sw}; done
    for f in data/bids/spider/derivatives/labels/*t2_SPACE.nii.gz; do mv $f ${f/t2_SPACE/T2Sw}; done

    # Create anat directories
    for f in data/bids/spider/*_*.nii.gz; do mkdir -p ${f/_*.nii.gz/}/anat; done
    for f in data/bids/spider/derivatives/labels/*_*.nii.gz; do mkdir -p ${f/_*.nii.gz/}/anat; done

    # Move files into anat directories
    for f in data/bids/spider/*_*.nii.gz; do mv $f ${f/_*.nii.gz/}/anat; done
    for f in data/bids/spider/derivatives/labels/*_*.nii.gz; do mv $f ${f/_*.nii.gz/}/anat; done

    # Rename segmentation files with '_seg' suffix
    for f in data/bids/spider/derivatives/labels/sub-*/anat/sub-*_*.nii.gz; do mv $f ${f/.nii.gz/_seg.nii.gz}; done
fi

# Make sure data/bids exists and enter it
mkdir -p data/bids
cd data/bids

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
