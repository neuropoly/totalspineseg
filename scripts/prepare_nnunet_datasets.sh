#!/bin/bash

# This script prepares datasets for the TotalSpineSeg model in nnUNetv2 structure.
# The script execpt DATASET as the first positional argument to specify the dataset to prepare.
# It can be either 101, 102, 103 or all. If all is specified, it will prepare all datasets (101, 102, 103).
# By default, it will prepare datasets 101 and 102.
# The script also exepct -noaug parameter to not generate augmentations.

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
if [[ $1 == 103 || $1 == all ]]; then PREP_103=1; else PREP_103=0; fi

# Set the augmentations to generate - default is to generate augmentations
if [[ $1 == -noaug || $2 == -noaug ]]; then NOAUG=1; else NOAUG=0; fi

# set TOTALSPINESEG and TOTALSPINESEG_DATA if not set
TOTALSPINESEG="$(realpath ${TOTALSPINESEG:-totalspineseg})"
TOTALSPINESEG_DATA="$(realpath ${TOTALSPINESEG_DATA:-data})"

# Set the path to the resources folder
resources="$TOTALSPINESEG"/totalspineseg/resources

# Set nnunet params
nnUNet_raw="$TOTALSPINESEG_DATA"/nnUNet/raw

# Set the paths to the BIDS data folders
bids="$TOTALSPINESEG_DATA"/bids

SRC_DATASET=Dataset99_TotalSpineSeg

echo "Make nnUNet raw folders"
mkdir -p "$nnUNet_raw"/$SRC_DATASET/imagesTr
mkdir -p "$nnUNet_raw"/$SRC_DATASET/labelsTr

# Init a list of dataset words
datasets_words=()

# Convert from BIDS to nnUNet dataset, loop over each dataset
for dsp in "$bids"/*; do
    # Get the dataset name
    dsn=$(basename $dsp);
    # Get dataset word from the dataset name
    dsw=${dsn#data-}; dsw=${dsw%%-*}; dsw=${dsw^^};

    # Add the dataset word to the list of dataset words
    datasets_words+=($dsw)

    echo "Working on $dsn"

    echo "Adding label-canal_seg and label-SC_seg to label-spine_dseg"
    totalspineseg_map_labels -m 1:201 --add-input -s "$bids"/$dsn/derivatives/labels_iso -o "$bids"/$dsn/derivatives/labels_iso --seg-suffix "_label-canal_seg" --output-seg-suffix "_label-spine_dseg" -d "sub-" -u "anat"
    totalspineseg_map_labels -m 1:200 --add-input -s "$bids"/$dsn/derivatives/labels_iso -o "$bids"/$dsn/derivatives/labels_iso --seg-suffix "_label-SC_seg" --output-seg-suffix "_label-spine_dseg" -d "sub-" -u "anat"

    echo "Copy images and labels into the nnUNet dataset folder"
    totalspineseg_cpdir "$bids"/$dsn "$nnUNet_raw"/$SRC_DATASET/imagesTr -p "sub-*/anat/sub-*.nii.gz" -f -r sub-:sub-${dsw} .nii.gz:_0000.nii.gz
    totalspineseg_cpdir "$bids"/$dsn/derivatives/labels_iso "$nnUNet_raw"/$SRC_DATASET/labelsTr -p "sub-*/anat/sub-*_label-spine_dseg.nii.gz" -f -r sub-:sub-${dsw} _space-resampled_label-spine_dseg.nii.gz:.nii.gz
done

echo "Remove images withot segmentation and segmentation without images"
for f in "$nnUNet_raw"/$SRC_DATASET/imagesTr/*.nii.gz; do if [ ! -f "$nnUNet_raw"/$SRC_DATASET/labelsTr/$(basename ${f/_0000.nii.gz/.nii.gz}) ]; then rm $f; fi; done
for f in "$nnUNet_raw"/$SRC_DATASET/labelsTr/*.nii.gz; do if [ ! -f "$nnUNet_raw"/$SRC_DATASET/imagesTr/$(basename ${f/.nii.gz/_0000.nii.gz}) ]; then rm $f; fi; done

echo "Convert 4D images to 3D"
totalspineseg_generate_averaged4d -i "$nnUNet_raw"/$SRC_DATASET/imagesTr -o "$nnUNet_raw"/$SRC_DATASET/imagesTr

echo "Transform images to canonical space"
totalspineseg_transform_norm -i "$nnUNet_raw"/$SRC_DATASET/imagesTr -o "$nnUNet_raw"/$SRC_DATASET/imagesTr

echo "Resample images to 1x1x1mm"
totalspineseg_generate_resampled_images -i "$nnUNet_raw"/$SRC_DATASET/imagesTr -o "$nnUNet_raw"/$SRC_DATASET/imagesTr

echo "Transform labels to images space"
totalspineseg_transform_labels2images -i "$nnUNet_raw"/$SRC_DATASET/imagesTr -s "$nnUNet_raw"/$SRC_DATASET/labelsTr -o "$nnUNet_raw"/$SRC_DATASET/labelsTr

echo "Making test folders and moving 10% of the data to test folders"
mkdir -p "$nnUNet_raw"/$SRC_DATASET/imagesTs
mkdir -p "$nnUNet_raw"/$SRC_DATASET/labelsTs

# Make sure each dataset and contrast has 10% of the data in the test folder
for d in ${datasets_words[@]}; do
    contrasts=(T1w T2w T2star flip-1_mt-on_MTS flip-2_mt-off_MTS)
    if [ -n "$(ls "$nnUNet_raw"/$SRC_DATASET/labelsTr/sub-${d}*_acq-highresSag_T2w.nii.gz 2>/dev/null)" ]; then
        contrasts=(acq-lowresSag_T1w acq-lowresSag_T2w acq-highresSag_T2w)
    fi
    for c in ${contrasts[@]}; do
        if [ -n "$(ls "$nnUNet_raw"/$SRC_DATASET/labelsTr/sub-${d}*_${c}.nii.gz 2>/dev/null)" ]; then
            files=($(for f in "$nnUNet_raw"/$SRC_DATASET/labelsTr/sub-${d}*_${c}.nii.gz; do basename "${f/.nii.gz/}"; done))
            files_shuf=($(shuf -e "${files[@]}"))
            files_10p=(${files_shuf[@]:0:$((${#files_shuf[@]} * 10 / 100))})
            for f in ${files_10p[@]}; do
                mv "$nnUNet_raw"/$SRC_DATASET/imagesTr/${f}_0000.nii.gz "$nnUNet_raw"/$SRC_DATASET/imagesTs;
                mv "$nnUNet_raw"/$SRC_DATASET/labelsTr/${f}.nii.gz "$nnUNet_raw"/$SRC_DATASET/labelsTs;
            done
        fi
    done
done

if [ $NOAUG -eq 0 ]; then
    echo "Generate augmentations"
    totalspineseg_generate_augmentations -i "$nnUNet_raw"/$SRC_DATASET/imagesTr -s "$nnUNet_raw"/$SRC_DATASET/labelsTr -o "$nnUNet_raw"/Dataset100_TotalSpineSeg_Aug/imagesTr -g "$nnUNet_raw"/Dataset100_TotalSpineSeg_Aug/labelsTr --labels2image --seg-classes 202-224 18-41,92 200 201
    totalspineseg_cpdir "$nnUNet_raw"/$SRC_DATASET "$nnUNet_raw"/Dataset100_TotalSpineSeg_Aug -p "*Ts/*.nii.gz"
    totalspineseg_transform_labels2images -i "$nnUNet_raw"/Dataset100_TotalSpineSeg_Aug/imagesTr -s "$nnUNet_raw"/Dataset100_TotalSpineSeg_Aug/labelsTr -o "$nnUNet_raw"/Dataset100_TotalSpineSeg_Aug/labelsTr
    SRC_DATASET=Dataset100_TotalSpineSeg_Aug
fi

if [ $PREP_101 -eq 1 ]; then
    echo "Generate nnUNet dataset 101 (step 1)"
    totalspineseg_cpdir "$nnUNet_raw"/$SRC_DATASET "$nnUNet_raw"/Dataset101_TotalSpineSeg_step1 -p "imagesT*/*.nii.gz"
    totalspineseg_map_labels -m "$resources"/labels_maps/nnunet_step1.json -s "$nnUNet_raw"/$SRC_DATASET/labelsTr -o "$nnUNet_raw"/Dataset101_TotalSpineSeg_step1/labelsTr
    totalspineseg_map_labels -m "$resources"/labels_maps/nnunet_step1.json -s "$nnUNet_raw"/$SRC_DATASET/labelsTs -o "$nnUNet_raw"/Dataset101_TotalSpineSeg_step1/labelsTs
    # Copy the dataset.json file and update the number of training samples
    jq --arg numTraining "$(ls "$nnUNet_raw"/Dataset101_TotalSpineSeg_step1/labelsTr | wc -l)" '.numTraining = ($numTraining|tonumber)' "$resources"/datasets/dataset_step1.json > "$nnUNet_raw"/Dataset101_TotalSpineSeg_step1/dataset.json
fi

if [ $PREP_102 -eq 1 ]; then
    echo "Generate nnUNet dataset 102 (step 2)"
    totalspineseg_cpdir "$nnUNet_raw"/$SRC_DATASET "$nnUNet_raw"/Dataset102_TotalSpineSeg_step2 -p "imagesT*/*.nii.gz"
    # This make a copy of the labelsTr then later we will map the labels so the odds and evens IVDs are switched
    totalspineseg_cpdir "$nnUNet_raw"/$SRC_DATASET "$nnUNet_raw"/Dataset102_TotalSpineSeg_step2 -p "imagesTr/*.nii.gz" -r "_0000.nii.gz:_o2e_0000.nii.gz"
    # This will map the labels to the second input channel
    totalspineseg_map_labels -m "$resources"/labels_maps/nnunet_step2_input.json -s "$nnUNet_raw"/$SRC_DATASET/labelsTr -o "$nnUNet_raw"/Dataset102_TotalSpineSeg_step2/imagesTr --output-seg-suffix _0001
    # This will map the labels to the extra images second input channel so the odd and even IVDs are switched
    totalspineseg_map_labels -m "$resources"/labels_maps/nnunet_step2_input_o2e.json -s "$nnUNet_raw"/$SRC_DATASET/labelsTr -o "$nnUNet_raw"/Dataset102_TotalSpineSeg_step2/imagesTr --output-seg-suffix _o2e_0001
    # This will map the labels to the second input channel for the test set
    totalspineseg_map_labels -m "$resources"/labels_maps/nnunet_step2_input.json -s "$nnUNet_raw"/$SRC_DATASET/labelsTs -o "$nnUNet_raw"/Dataset102_TotalSpineSeg_step2/imagesTs --output-seg-suffix _0001
    totalspineseg_map_labels -m "$resources"/labels_maps/nnunet_step2.json -s "$nnUNet_raw"/$SRC_DATASET/labelsTr -o "$nnUNet_raw"/Dataset102_TotalSpineSeg_step2/labelsTr
    # This will map the extra images labels so the odd and even IVDs are switched
    totalspineseg_map_labels -m "$resources"/labels_maps/nnunet_step2_o2e.json -s "$nnUNet_raw"/$SRC_DATASET/labelsTr -o "$nnUNet_raw"/Dataset102_TotalSpineSeg_step2/labelsTr --output-seg-suffix _o2e
    totalspineseg_map_labels -m "$resources"/labels_maps/nnunet_step2.json -s "$nnUNet_raw"/$SRC_DATASET/labelsTs -o "$nnUNet_raw"/Dataset102_TotalSpineSeg_step2/labelsTs
    # Copy the dataset.json file and update the number of training samples
    jq --arg numTraining "$(ls "$nnUNet_raw"/Dataset102_TotalSpineSeg_step2/labelsTr | wc -l)" '.numTraining = ($numTraining|tonumber)' "$resources"/datasets/dataset_step2.json > "$nnUNet_raw"/Dataset102_TotalSpineSeg_step2/dataset.json
fi

if [ $PREP_103 -eq 1 ]; then
    echo "Generate nnUNet dataset 103 (full)"
    totalspineseg_cpdir "$nnUNet_raw"/$SRC_DATASET "$nnUNet_raw"/Dataset103_TotalSpineSeg_full -p "imagesT*/*.nii.gz"
    totalspineseg_map_labels -m "$resources"/labels_maps/nnunet_full.json -s "$nnUNet_raw"/$SRC_DATASET/labelsTr -o "$nnUNet_raw"/Dataset103_TotalSpineSeg_full/labelsTr
    totalspineseg_map_labels -m "$resources"/labels_maps/nnunet_full.json -s "$nnUNet_raw"/$SRC_DATASET/labelsTs -o "$nnUNet_raw"/Dataset103_TotalSpineSeg_full/labelsTs
    # Copy the dataset.json file and update the number of training samples
    jq --arg numTraining "$(ls "$nnUNet_raw"/Dataset103_TotalSpineSeg_full/labelsTr | wc -l)" '.numTraining = ($numTraining|tonumber)' "$resources"/datasets/dataset_full.json > "$nnUNet_raw"/Dataset103_TotalSpineSeg_full/dataset.json
fi