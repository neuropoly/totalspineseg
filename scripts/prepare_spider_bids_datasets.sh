#!/bin/bash

# This script prepares SPIDER datasets in BIDS structure.

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

# Set the path to the resources folder
resources="$TOTALSPINESEG"/totalspineseg/resources

# Set the paths to the BIDS and raw data folders
bids="$TOTALSPINESEG_DATA"/bids
raw="$TOTALSPINESEG_DATA"/raw

# Check if the TotalSpineSeg folder exists
if [ ! -d totalspineseg ]; then
    echo "totalspineseg folder not found in current working directory. Exiting."
    echo "Please make sure you clone the repository into the current working directory."
    exit 0
fi

# Check if the SPIDER dataset images and masks are available
if [ ! -d "$raw"/spider/images ] || [ ! -d "$raw"/spider/masks ]; then
    echo "SPIDER dataset images or masks not found. Please make sure they are available in "$raw"/spider."
else
    echo "Preparing SPIDER dataset"

    echo "Convert images and masks from .mha to .nii.gz format"
    totalspineseg_mha2nii -i "$raw"/spider/images -o "$bids"/spider
    totalspineseg_mha2nii -i "$raw"/spider/masks -o "$bids"/spider/derivatives/labels

    echo "Map SPIDER labels to the labels used in this project"
    totalspineseg_map_labels -m "$resources"/labels_maps/spider.json -s "$bids"/spider/derivatives/labels -o "$bids"/spider/derivatives/labels

    echo "Rename files to follow BIDS naming conventions"
    # Add 'sub-' prefix to filenames
    for f in "$bids"/spider/*.nii.gz; do mv $f ${f/spider\//spider\/sub-}; done
    for f in "$bids"/spider/derivatives/labels/*.nii.gz; do mv $f ${f/labels\//labels\/sub-}; done

    # Replace 't1' with 'T1w' in filenames
    for f in "$bids"/spider/*t1.nii.gz; do mv $f ${f/t1/T1w}; done
    for f in "$bids"/spider/derivatives/labels/*t1.nii.gz; do mv $f ${f/t1/T1w}; done

    # Replace 't2' with 'T2w' in filenames
    for f in "$bids"/spider/*t2.nii.gz; do mv $f ${f/t2/T2w}; done
    for f in "$bids"/spider/derivatives/labels/*t2.nii.gz; do mv $f ${f/t2/T2w}; done

    # Replace 't2_SPACE' with 'T2Sw' in filenames
    for f in "$bids"/spider/*t2_SPACE.nii.gz; do mv $f ${f/t2_SPACE/T2Sw}; done
    for f in "$bids"/spider/derivatives/labels/*t2_SPACE.nii.gz; do mv $f ${f/t2_SPACE/T2Sw}; done

    # Create anat directories
    for f in "$bids"/spider/*_*.nii.gz; do mkdir -p ${f/_*.nii.gz/}/anat; done
    for f in "$bids"/spider/derivatives/labels/*_*.nii.gz; do mkdir -p ${f/_*.nii.gz/}/anat; done

    # Move files into anat directories
    for f in "$bids"/spider/*_*.nii.gz; do mv $f ${f/_*.nii.gz/}/anat; done
    for f in "$bids"/spider/derivatives/labels/*_*.nii.gz; do mv $f ${f/_*.nii.gz/}/anat; done

    # Rename segmentation files with '_totalspineseg' suffix
    for f in "$bids"/spider/derivatives/labels/sub-*/anat/sub-*_*.nii.gz; do mv $f ${f/.nii.gz/_totalspineseg.nii.gz}; done

    # Add the spinal cord from _seg to _totalspineseg
    totalspineseg_map_labels -m 1:200 --add-input --seg-suffix "_seg" --output-seg-suffix "_totalspineseg" -d "sub-" -u "anat" -s "$bids"/spider/derivatives/labels -o "$bids"/spider/derivatives/labels

    # Add the sacrum from _sacrum to _totalspineseg
    totalspineseg_map_labels -m 1:92 --add-input --seg-suffix "_sacrum" --output-seg-suffix "_totalspineseg" -d "sub-" -u "anat" -s "$bids"/spider/derivatives/labels -o "$bids"/spider/derivatives/labels
fi
