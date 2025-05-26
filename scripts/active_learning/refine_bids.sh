#!/bin/bash

# Uncomment for full verbose
# set -x

# Immediately exit if error
set -e -o pipefail

# Exit if user presses CTRL+C (Linux) or CMD+C (OSX)
trap "echo Caught Keyboard Interrupt within script. Exiting now.; exit" INT

# GET PARAMS
# ======================================================================================================================
# SET DEFAULT VALUES FOR PARAMETERS.
# ----------------------------------------------------------------------------------------------------------------------
BIDS_FOLDER="/home/GRAMES.POLYMTL.CA/p118739/data_nvme_p118739/data/datasets/whole-spine" # -----> Update environment variables
TOTALSPINESEG="/home/GRAMES.POLYMTL.CA/p118739/data_nvme_p118739/code/totalspineseg"      # -----> Update environment variables

IMG_FOLDER="derivatives/img"
PRED_FOLDER="derivatives/pred"
LABEL_FOLDER="derivatives/labels_nnInteractive"

# ======================================================================================================================
# SCRIPT STARTS HERE
# ======================================================================================================================

cd "$BIDS_FOLDER"

# Create folders
mkdir -p "$IMG_FOLDER"
mkdir -p "$PRED_FOLDER"
mkdir -p "$LABEL_FOLDER"

# Copy images in same folder
cp $(find sub* -type f -name '*.nii.gz' | grep -v .git) "$IMG_FOLDER"

# Run totalspineseg on BIDS dataset
source /usr/local/miniforge3/etc/profile.d/conda.sh # -----> Update environment variables
conda activate tss_env                              # -----> Update environment variables
echo Running TotalSpineSeg
echo 
totalspineseg "$IMG_FOLDER" "$PRED_FOLDER" -k step2_output
conda deactivate                                    # -----> Update environment variables

# Run nnInteractive using the predictions
for file in $(ls "$IMG_FOLDER");do
    sub=$(echo "$file" | cut -d _ -f 1)
    file_noext=$(echo "$file" | cut -d . -f 1)

    # Create directory
    mkdir -p "$LABEL_FOLDER"/"$sub"/anat

    # Run nnInteractive
    conda activate nnInteractive                    # -----> Update environment variables
    python "$TOTALSPINESEG"/scripts/active_learning/nnInteractive_refine.py -i "$IMG_FOLDER"/"$file" -s "$PRED_FOLDER"/step2_output/"$file" -o "$LABEL_FOLDER"/"$sub"/anat
    conda deactivate                                # -----> Update environment variables
done
    
