#!/bin/bash

# This script prepares datasets for the TotalSpineSeg model in nnUNetv2 structure.
# The script execpt DATASET as the first positional argument to specify the dataset to prepare.
# It can be either 101, 102, 103 or all. If all is specified, it will prepare all datasets (101, 102, 103).
# By default, it will prepare datasets 101 and 102.
# The script also exepct -noaug parameter to not generate augmentations.

# The script excpects the following environment variables to be set:
#   TOTALSPINESEG: The path to the TotalSpineSeg repository.
#   TOTALSPINESEG_DATA: The path to the TotalSpineSeg data folder.
#   TOTALSPINESEG_JOBS: The number of CPU cores to use. Default is the number of CPU cores available.

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

# Set the datasets to work with - default is 101 102
if [[ -z $1 || $1 == 101 || $1 == all || $1 == -* ]]; then PREP_101=1; else PREP_101=0; fi
if [[ -z $1 || $1 == 102 || $1 == all || $1 == -* ]]; then PREP_102=1; else PREP_102=0; fi

# set TOTALSPINESEG and TOTALSPINESEG_DATA if not set
TOTALSPINESEG="$(realpath "${TOTALSPINESEG:-totalspineseg}")"
TOTALSPINESEG_DATA="$(realpath "${TOTALSPINESEG_DATA:-data}")"

# Set the path to the resources folder
resources="$TOTALSPINESEG"/totalspineseg/resources

# Fetch path to data list
data_json="$resources/data/training_data.json"

# Get the number of CPUs
CORES=${SLURM_JOB_CPUS_PER_NODE:-$(lscpu -p | egrep -v '^#' | wc -l)}

# Set the number of jobs
JOBS=${TOTALSPINESEG_JOBS:-$CORES}

# Set nnunet params
nnUNet_raw="$TOTALSPINESEG_DATA"/nnUNet/raw

# Set the paths to the BIDS data folders
bids="$TOTALSPINESEG_DATA"/bids

SRC_DATASET=Dataset99_TotalSpineSeg

### Prepare TRAIN set

echo "Make nnUNet raw folders"
mkdir -p "$nnUNet_raw"/$SRC_DATASET/imagesTr
mkdir -p "$nnUNet_raw"/$SRC_DATASET/labelsTr

# Move to bids directory
CURR_DIR="$(realpath .)"
cd "$bids"

# Copy label data in nnUNet_raw folder
cp $(jq -r ".TRAINING | .[].LABEL_SPINE" "$data_json") "$nnUNet_raw"/$SRC_DATASET/labelsTr
cp $(jq -r ".TRAINING | .[].LABEL_CORD" "$data_json") "$nnUNet_raw"/$SRC_DATASET/labelsTr
cp $(jq -r ".TRAINING | .[].LABEL_CANAL" "$data_json") "$nnUNet_raw"/$SRC_DATASET/labelsTr

# Copy images and add nnUNet suffix _0000
for img in $(jq -r ".TRAINING | .[].IMAGE" "$data_json");do img_name=$(basename ${img/.nii.gz/_0000.nii.gz}); cp "$img" "$nnUNet_raw"/$SRC_DATASET/imagesTr/"$img_name";done

# Reorient images to canonical space
echo "Transform images to canonical space"
totalspineseg_reorient_canonical -i "$nnUNet_raw"/$SRC_DATASET/imagesTr -o "$nnUNet_raw"/$SRC_DATASET/imagesTr -r -w $JOBS
totalspineseg_reorient_canonical -i "$nnUNet_raw"/$SRC_DATASET/labelsTr -o "$nnUNet_raw"/$SRC_DATASET/labelsTr -r -w $JOBS

# Map canal and SC onto label-spine_dseg file
echo "Adding label-canal_seg and label-SC_seg to label-spine_dseg"
totalspineseg_map_labels -m 1:2 -s "$nnUNet_raw"/$SRC_DATASET/labelsTr -o "$nnUNet_raw"/$SRC_DATASET/labelsTr  --update-segs-dir "$nnUNet_raw"/$SRC_DATASET/labelsTr --seg-suffix "_label-canal_seg" --output-seg-suffix "_label-spine_dseg" --update-seg-suffix "_label-spine_dseg" -r -w $JOBS
totalspineseg_map_labels -m 1:1 -s "$nnUNet_raw"/$SRC_DATASET/labelsTr -o "$nnUNet_raw"/$SRC_DATASET/labelsTr  --update-segs-dir "$nnUNet_raw"/$SRC_DATASET/labelsTr --seg-suffix "_label-SC_seg" --output-seg-suffix "_label-spine_dseg" --update-seg-suffix "_label-spine_dseg" -r -w $JOBS

echo "Removing SC and canal label from folder"
rm "$nnUNet_raw"/$SRC_DATASET/labelsTr/*_label-SC_seg.nii.gz
rm "$nnUNet_raw"/$SRC_DATASET/labelsTr/*_label-canal_seg.nii.gz

echo "Renaming labels"
for label in "$nnUNet_raw"/$SRC_DATASET/labelsTr/*;do mv "$label" ${label/_label-spine_dseg/};done

echo "Resample images to 1x1x1mm"
totalspineseg_resample -i "$nnUNet_raw"/$SRC_DATASET/imagesTr -o "$nnUNet_raw"/$SRC_DATASET/imagesTr -r -w $JOBS

echo "Transform labels to images space"
totalspineseg_transform_seg2image -i "$nnUNet_raw"/$SRC_DATASET/imagesTr -s "$nnUNet_raw"/$SRC_DATASET/labelsTr -o "$nnUNet_raw"/$SRC_DATASET/labelsTr -r -w $JOBS

### Prepare TEST set

echo "Creating test folders"
mkdir -p "$nnUNet_raw"/$SRC_DATASET/imagesTs
mkdir -p "$nnUNet_raw"/$SRC_DATASET/labelsTs

# Copy test data in nnUNet_raw folder
cp $(jq -r ".TESTING | .[].LABEL_SPINE" "$data_json") "$nnUNet_raw"/$SRC_DATASET/labelsTs
cp $(jq -r ".TESTING | .[].LABEL_CORD" "$data_json") "$nnUNet_raw"/$SRC_DATASET/labelsTs
cp $(jq -r ".TESTING | .[].LABEL_CANAL" "$data_json") "$nnUNet_raw"/$SRC_DATASET/labelsTs

# Copy images and add nnUNet suffix _0000
for img in $(jq -r ".TESTING | .[].IMAGE" "$data_json");do img_name=$(basename ${img/.nii.gz/_0000.nii.gz}); cp "$img" "$nnUNet_raw"/$SRC_DATASET/imagesTs/"$img_name";done

# Map canal and SC onto label-spine_dseg file for testing
echo "Adding label-canal_seg and label-SC_seg to label-spine_dseg"
totalspineseg_map_labels -m 1:2 -s "$nnUNet_raw"/$SRC_DATASET/labelsTs -o "$nnUNet_raw"/$SRC_DATASET/labelsTs  --update-segs-dir "$nnUNet_raw"/$SRC_DATASET/labelsTs --seg-suffix "_label-canal_seg" --output-seg-suffix "_label-spine_dseg" --update-seg-suffix "_label-spine_dseg" -r -w $JOBS
totalspineseg_map_labels -m 1:1 -s "$nnUNet_raw"/$SRC_DATASET/labelsTs -o "$nnUNet_raw"/$SRC_DATASET/labelsTs  --update-segs-dir "$nnUNet_raw"/$SRC_DATASET/labelsTs --seg-suffix "_label-SC_seg" --output-seg-suffix "_label-spine_dseg" --update-seg-suffix "_label-spine_dseg" -r -w $JOBS

echo "Removing SC and canal label from folder"
rm "$nnUNet_raw"/$SRC_DATASET/labelsTs/*_label-SC_seg.nii.gz
rm "$nnUNet_raw"/$SRC_DATASET/labelsTs/*_label-canal_seg.nii.gz

echo "Renaming labels"
for label in "$nnUNet_raw"/$SRC_DATASET/labelsTs/*;do mv "$label" ${label/_label-spine_dseg/};done

echo "Resample images to 1x1x1mm"
totalspineseg_resample -i "$nnUNet_raw"/$SRC_DATASET/imagesTs -o "$nnUNet_raw"/$SRC_DATASET/imagesTs -r -w $JOBS

echo "Transform labels to images space"
totalspineseg_transform_seg2image -i "$nnUNet_raw"/$SRC_DATASET/imagesTs -s "$nnUNet_raw"/$SRC_DATASET/labelsTs -o "$nnUNet_raw"/$SRC_DATASET/labelsTs -r -w $JOBS

### Remap label for TRAINING and TESTING

if [ $PREP_101 -eq 1 ]; then
    echo "Generate nnUNet dataset 101 (step 1)"
    totalspineseg_cpdir "$nnUNet_raw"/$SRC_DATASET "$nnUNet_raw"/Dataset101_TotalSpineSeg_step1 -p "imagesT*/*.nii.gz" -r -w $JOBS
    totalspineseg_map_labels -m "$resources"/labels_maps/nnunet_step1.json -s "$nnUNet_raw"/$SRC_DATASET/labelsTr -o "$nnUNet_raw"/Dataset101_TotalSpineSeg_step1/labelsTr -r -w $JOBS
    totalspineseg_map_labels -m "$resources"/labels_maps/nnunet_step1.json -s "$nnUNet_raw"/$SRC_DATASET/labelsTs -o "$nnUNet_raw"/Dataset101_TotalSpineSeg_step1/labelsTs -r -w $JOBS
    # Copy the dataset.json file and update the number of training samples
    jq --arg numTraining "$(ls "$nnUNet_raw"/Dataset101_TotalSpineSeg_step1/labelsTr | wc -l)" '.numTraining = ($numTraining|tonumber)' "$resources"/datasets/dataset_step1.json > "$nnUNet_raw"/Dataset101_TotalSpineSeg_step1/dataset.json
fi

if [ $PREP_102 -eq 1 ]; then
    echo "Generate nnUNet dataset 102 (step 2)"
    totalspineseg_cpdir "$nnUNet_raw"/$SRC_DATASET "$nnUNet_raw"/Dataset102_TotalSpineSeg_step2 -p "imagesT*/*.nii.gz" -r -w $JOBS
    # This make a copy of the labelsTr then later we will map the labels so the odds and evens IVDs are switched
    totalspineseg_cpdir "$nnUNet_raw"/$SRC_DATASET "$nnUNet_raw"/Dataset102_TotalSpineSeg_step2 -p "imagesTr/*.nii.gz" -t "_0000.nii.gz:_o2e_0000.nii.gz" -r -w $JOBS
    # This will map the labels to the second input channel
    totalspineseg_extract_alternate -s "$nnUNet_raw"/$SRC_DATASET/labelsTr -o "$nnUNet_raw"/Dataset102_TotalSpineSeg_step2/imagesTr --labels 63-100 --prioratize-labels 63 65 67 72 74 76 78 80 82 92 94 --output-seg-suffix _0001 -r -w $JOBS -r
    # This will map the labels to the extra images second input channel so the odd and even IVDs are switched
    totalspineseg_extract_alternate -s "$nnUNet_raw"/$SRC_DATASET/labelsTr -o "$nnUNet_raw"/Dataset102_TotalSpineSeg_step2/imagesTr --labels 63-100 --prioratize-labels 64 66 71 73 75 77 79 81 91 93 95 --output-seg-suffix _o2e_0001 -r -w $JOBS -r
    # This will map the labels to the second input channel for the test set
    totalspineseg_extract_alternate -s "$nnUNet_raw"/$SRC_DATASET/labelsTs -o "$nnUNet_raw"/Dataset102_TotalSpineSeg_step2/imagesTs --labels 63-100 --prioratize-labels 63 65 67 72 74 76 78 80 82 92 94 --output-seg-suffix _0001 -r -w $JOBS -r
    totalspineseg_map_labels -m "$resources"/labels_maps/nnunet_step2.json -s "$nnUNet_raw"/$SRC_DATASET/labelsTr -o "$nnUNet_raw"/Dataset102_TotalSpineSeg_step2/labelsTr -r -w $JOBS
    # This will map the extra images labels so the odd and even IVDs are switched
    totalspineseg_map_labels -m "$resources"/labels_maps/nnunet_step2_o2e.json -s "$nnUNet_raw"/$SRC_DATASET/labelsTr -o "$nnUNet_raw"/Dataset102_TotalSpineSeg_step2/labelsTr --output-seg-suffix _o2e -r -w $JOBS
    totalspineseg_map_labels -m "$resources"/labels_maps/nnunet_step2.json -s "$nnUNet_raw"/$SRC_DATASET/labelsTs -o "$nnUNet_raw"/Dataset102_TotalSpineSeg_step2/labelsTs -r -w $JOBS
    # Copy the dataset.json file and update the number of training samples
    jq --arg numTraining "$(ls "$nnUNet_raw"/Dataset102_TotalSpineSeg_step2/labelsTr | wc -l)" '.numTraining = ($numTraining|tonumber)' "$resources"/datasets/dataset_step2.json > "$nnUNet_raw"/Dataset102_TotalSpineSeg_step2/dataset.json
fi

# Move back
cd "$CURR_DIR"