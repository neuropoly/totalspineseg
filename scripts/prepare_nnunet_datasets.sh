#!/bin/bash

# This script prepares datasets for the TotalSegMRI model in nnUNetv2 structure.

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

# Set the path to the utils folder and resources
utils=totalsegmentator-mri/src/totalsegmri/utils
resources=totalsegmentator-mri/src/totalsegmri/resources

# Set nnunet params
nnUNet_raw=data/nnUNet/raw

# Set the paths to the BIDS data folders
bids=data/bids

echo "Make nnUNet raw folders"
mkdir -p $nnUNet_raw/Dataset100_TotalSegMRI/imagesTr
mkdir -p $nnUNet_raw/Dataset100_TotalSegMRI/labelsTr

# Init a list of dataset words
datasets_words=()

# Convert from BIDS to nnUNet dataset, loop over each dataset
for dsp in $bids/*; do
    # Get the dataset name
    dsn=$(basename $dsp);
    # Get dataset word from the dataset name
    dsw=${dsn#data-}; dsw=${dsw%-*};

    # Add the dataset word to the list of dataset words
    datasets_words+=($dsw)

    echo "Working on $dsn"
    #label-spine_dseg\|label-SC_seg\|label-canal_seg

    echo "Adding label-canal_seg and label-SC_seg to label-spine_dseg"
    python $utils/map_labels.py -m 1:201 --add-input -s $bids/$dsn/derivatives/labels -o $bids/$dsn/derivatives/labels --seg-suffix "_label-canal_seg" --output-seg-suffix "_label-spine_dseg" -d "sub-" -u "anat"
    python $utils/map_labels.py -m 1:200 --add-input -s $bids/$dsn/derivatives/labels -o $bids/$dsn/derivatives/labels --seg-suffix "_label-SC_seg" --output-seg-suffix "_label-spine_dseg" -d "sub-" -u "anat"

    echo "Copy images and labels into the nnUNet dataset folder"
    python $utils/cpdir.py $bids/$dsn $nnUNet_raw/Dataset100_TotalSegMRI/imagesTr -p "sub-*/anat/sub-*.nii.gz" -f -r sub-:${dsw}_ .nii.gz:_0000.nii.gz
    python $utils/cpdir.py $bids/$dsn/derivatives/labels $nnUNet_raw/Dataset100_TotalSegMRI/labelsTr -p "sub-*/anat/sub-*_label-spine_dseg.nii.gz" -f -r sub-:${dsw}_ _label-spine_dseg.nii.gz:.nii.gz
done

echo "Transform images to canonical space and fix data type mismatch and sform qform mismatch"
python $utils/transform_norm.py -i $nnUNet_raw/Dataset100_TotalSegMRI/imagesTr -o $nnUNet_raw/Dataset100_TotalSegMRI/imagesTr

echo "Resample images to 1x1x1mm"
python $utils/generate_resampled_images.py -i $nnUNet_raw/Dataset100_TotalSegMRI/imagesTr -o $nnUNet_raw/Dataset100_TotalSegMRI/imagesTr

echo "Transform labels to images space"
python $utils/transform_labels2images.py -i $nnUNet_raw/Dataset100_TotalSegMRI/imagesTr -s $nnUNet_raw/Dataset100_TotalSegMRI/labelsTr -o $nnUNet_raw/Dataset100_TotalSegMRI/labelsTr

echo "Making test folders and moving 10% of the data to test folders"
mkdir -p $nnUNet_raw/Dataset100_TotalSegMRI/imagesTs
mkdir -p $nnUNet_raw/Dataset100_TotalSegMRI/labelsTs

# Make sure each dataset and contrast has 10% of the data in the test folder
for d in ${datasets_words[@]}; do
    for c in acq-lowresSag_T1w acq-lowresSag_T2w acq-highresSag_T2w; do
        files=($(for f in $nnUNet_raw/Dataset100_TotalSegMRI/labelsTr/${d}_*${c}.nii.gz; do basename "${f/.nii.gz/}"; done))
        files_shuf=($(shuf -e "${files[@]}"))
        files_10p=(${files_shuf[@]:0:$((${#files_shuf[@]} * 10 / 100))})
        for f in ${files_10p[@]}; do
            mv $nnUNet_raw/Dataset100_TotalSegMRI/imagesTr/${f}_0000.nii.gz $nnUNet_raw/Dataset100_TotalSegMRI/imagesTs;
            mv $nnUNet_raw/Dataset100_TotalSegMRI/labelsTr/${f}.nii.gz $nnUNet_raw/Dataset100_TotalSegMRI/labelsTs;
        done
    done
done

echo "Generate augmentations"
python $utils/generate_augmentations.py -i $nnUNet_raw/Dataset100_TotalSegMRI/imagesTr -s $nnUNet_raw/Dataset100_TotalSegMRI/labelsTr -o $nnUNet_raw/Dataset100_TotalSegMRI/imagesTr -g $nnUNet_raw/Dataset100_TotalSegMRI/labelsTr --labels2image --seg-classes 202-224 18-41,92 200 201

echo "Map labels form TotalSegMRI labels to the datasets specific label"
python $utils/cpdir.py $nnUNet_raw/Dataset100_TotalSegMRI $nnUNet_raw/Dataset101_TotalSegMRI_step1 -p "imagesT*/*.nii.gz"
python $utils/cpdir.py $nnUNet_raw/Dataset100_TotalSegMRI $nnUNet_raw/Dataset102_TotalSegMRI_step2 -p "imagesT*/*.nii.gz"
# This make a copy of the labelsTr then later we will map the labels so the odds and evens IVDs are switched
python $utils/cpdir.py $nnUNet_raw/Dataset100_TotalSegMRI $nnUNet_raw/Dataset102_TotalSegMRI_step2 -p "imagesTr/*.nii.gz" -r "_0000.nii.gz:_o2e_0000.nii.gz"
python $utils/cpdir.py $nnUNet_raw/Dataset100_TotalSegMRI $nnUNet_raw/Dataset103_TotalSegMRI_full -p "imagesT*/*.nii.gz"

python $utils/map_labels.py -m $resources/labels_maps/nnunet_step1.json -s $nnUNet_raw/Dataset100_TotalSegMRI/labelsTr -o $nnUNet_raw/Dataset101_TotalSegMRI_step1/labelsTr
python $utils/map_labels.py -m $resources/labels_maps/nnunet_step1.json -s $nnUNet_raw/Dataset100_TotalSegMRI/labelsTs -o $nnUNet_raw/Dataset101_TotalSegMRI_step1/labelsTs
# This will map the labels to the second input channel
python $utils/map_labels.py -m $resources/labels_maps/nnunet_step2_input.json -s $nnUNet_raw/Dataset100_TotalSegMRI/labelsTr -o $nnUNet_raw/Dataset102_TotalSegMRI_step2/imagesTr --output-seg-suffix _0001
# This will map the labels to the extra images second input channel so the odd and even IVDs are switched
python $utils/map_labels.py -m $resources/labels_maps/nnunet_step2_input_o2e.json -s $nnUNet_raw/Dataset100_TotalSegMRI/labelsTr -o $nnUNet_raw/Dataset102_TotalSegMRI_step2/imagesTr --output-seg-suffix _o2e_0001
# This will map the labels to the second input channel for the test set
python $utils/map_labels.py -m $resources/labels_maps/nnunet_step2_input.json -s $nnUNet_raw/Dataset100_TotalSegMRI/labelsTs -o $nnUNet_raw/Dataset102_TotalSegMRI_step2/imagesTs --output-seg-suffix _0001
python $utils/map_labels.py -m $resources/labels_maps/nnunet_step2.json -s $nnUNet_raw/Dataset100_TotalSegMRI/labelsTr -o $nnUNet_raw/Dataset102_TotalSegMRI_step2/labelsTr
# This will map the extra images labels so the odd and even IVDs are switched
python $utils/map_labels.py -m $resources/labels_maps/nnunet_step2_o2e.json -s $nnUNet_raw/Dataset100_TotalSegMRI/labelsTr -o $nnUNet_raw/Dataset102_TotalSegMRI_step2/labelsTr --output-seg-suffix _o2e
python $utils/map_labels.py -m $resources/labels_maps/nnunet_step2.json -s $nnUNet_raw/Dataset100_TotalSegMRI/labelsTs -o $nnUNet_raw/Dataset102_TotalSegMRI_step2/labelsTs
python $utils/map_labels.py -m $resources/labels_maps/nnunet_full.json -s $nnUNet_raw/Dataset100_TotalSegMRI/labelsTr -o $nnUNet_raw/Dataset103_TotalSegMRI_full/labelsTr
python $utils/map_labels.py -m $resources/labels_maps/nnunet_full.json -s $nnUNet_raw/Dataset100_TotalSegMRI/labelsTs -o $nnUNet_raw/Dataset103_TotalSegMRI_full/labelsTs

echo "Copy dataset files"
jq --arg numTraining "$(ls $nnUNet_raw/Dataset101_TotalSegMRI_step1/labelsTr | wc -l)" '.numTraining = ($numTraining|tonumber)' $resources/datasets/dataset_step1.json > $nnUNet_raw/Dataset101_TotalSegMRI_step1/dataset.json
jq --arg numTraining "$(ls $nnUNet_raw/Dataset102_TotalSegMRI_step2/labelsTr | wc -l)" '.numTraining = ($numTraining|tonumber)' $resources/datasets/dataset_step2.json > $nnUNet_raw/Dataset102_TotalSegMRI_step2/dataset.json
jq --arg numTraining "$(ls $nnUNet_raw/Dataset103_TotalSegMRI_full/labelsTr | wc -l)" '.numTraining = ($numTraining|tonumber)' $resources/datasets/dataset_full.json > $nnUNet_raw/Dataset103_TotalSegMRI_full/dataset.json