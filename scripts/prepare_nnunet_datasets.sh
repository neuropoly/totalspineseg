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
mkdir -p $nnUNet_raw/Dataset99_TotalSegMRI/imagesTr
mkdir -p $nnUNet_raw/Dataset99_TotalSegMRI/labelsTr

# Init a list of dataset words
datasets_words=()

# Convert from BIDS to nnUNet dataset, loop over each dataset
for dsp in $bids/*; do
    # Get the dataset name
    dsn=$(basename $dsp);
    # Get dataset word from the dataset name
    dsw=${dsn#data-}; dsw=${dsw%%-*}; dsw=${dsw^^};

    # Add the dataset word to the list of dataset words
    datasets_words+=($dsw)

    echo "Working on $dsn"

    echo "Adding label-canal_seg and label-SC_seg to label-spine_dseg"
    python $utils/map_labels.py -m 1:201 --add-input -s $bids/$dsn/derivatives/labels -o $bids/$dsn/derivatives/labels --seg-suffix "_label-canal_seg" --output-seg-suffix "_label-spine_dseg" -d "sub-" -u "anat"
    python $utils/map_labels.py -m 1:200 --add-input -s $bids/$dsn/derivatives/labels -o $bids/$dsn/derivatives/labels --seg-suffix "_label-SC_seg" --output-seg-suffix "_label-spine_dseg" -d "sub-" -u "anat"

    echo "Copy images and labels into the nnUNet dataset folder"
    python $utils/cpdir.py $bids/$dsn $nnUNet_raw/Dataset99_TotalSegMRI/imagesTr -p "sub-*/anat/sub-*.nii.gz" -f -r sub-:sub-${dsw} .nii.gz:_0000.nii.gz
    python $utils/cpdir.py $bids/$dsn/derivatives/labels $nnUNet_raw/Dataset99_TotalSegMRI/labelsTr -p "sub-*/anat/sub-*_label-spine_dseg.nii.gz" -f -r sub-:sub-${dsw} _label-spine_dseg.nii.gz:.nii.gz
done

echo "Remove images withot segmentation and segmentation without images"
for f in $nnUNet_raw/Dataset99_TotalSegMRI/imagesTr/*.nii.gz; do if [ ! -f $nnUNet_raw/Dataset99_TotalSegMRI/labelsTr/$(basename ${f/_0000.nii.gz/.nii.gz}) ]; then rm $f; fi; done
for f in $nnUNet_raw/Dataset99_TotalSegMRI/labelsTr/*.nii.gz; do if [ ! -f $nnUNet_raw/Dataset99_TotalSegMRI/imagesTr/$(basename ${f/.nii.gz/_0000.nii.gz}) ]; then rm $f; fi; done

echo "Transform images to canonical space and fix data type mismatch and sform qform mismatch"
python $utils/transform_norm.py -i $nnUNet_raw/Dataset99_TotalSegMRI/imagesTr -o $nnUNet_raw/Dataset99_TotalSegMRI/imagesTr

echo "Resample images to 1x1x1mm"
python $utils/generate_resampled_images.py -i $nnUNet_raw/Dataset99_TotalSegMRI/imagesTr -o $nnUNet_raw/Dataset99_TotalSegMRI/imagesTr

echo "Transform labels to images space"
python $utils/transform_labels2images.py -i $nnUNet_raw/Dataset99_TotalSegMRI/imagesTr -s $nnUNet_raw/Dataset99_TotalSegMRI/labelsTr -o $nnUNet_raw/Dataset99_TotalSegMRI/labelsTr

echo "Making test folders and moving 10% of the data to test folders"
mkdir -p $nnUNet_raw/Dataset99_TotalSegMRI/imagesTs
mkdir -p $nnUNet_raw/Dataset99_TotalSegMRI/labelsTs

# Make sure each dataset and contrast has 10% of the data in the test folder
for d in ${datasets_words[@]}; do
    contrasts=(T1w T2w T2star flip-1_mt-on_MTS flip-2_mt-off_MTS)
    if [ -n "$(ls $nnUNet_raw/Dataset99_TotalSegMRI/labelsTr/sub-${d}*_acq-highresSag_T2w.nii.gz 2>/dev/null)" ]; then
        contrasts=(acq-lowresSag_T1w acq-lowresSag_T2w acq-highresSag_T2w)
    fi
    for c in ${contrasts[@]}; do
        if [ -n "$(ls $nnUNet_raw/Dataset99_TotalSegMRI/labelsTr/sub-${d}*_${c}.nii.gz 2>/dev/null)" ]; then
            files=($(for f in $nnUNet_raw/Dataset99_TotalSegMRI/labelsTr/sub-${d}*_${c}.nii.gz; do basename "${f/.nii.gz/}"; done))
            files_shuf=($(shuf -e "${files[@]}"))
            files_10p=(${files_shuf[@]:0:$((${#files_shuf[@]} * 10 / 100))})
            for f in ${files_10p[@]}; do
                mv $nnUNet_raw/Dataset99_TotalSegMRI/imagesTr/${f}_0000.nii.gz $nnUNet_raw/Dataset99_TotalSegMRI/imagesTs;
                mv $nnUNet_raw/Dataset99_TotalSegMRI/labelsTr/${f}.nii.gz $nnUNet_raw/Dataset99_TotalSegMRI/labelsTs;
            done
        fi
    done
done

echo "Generate augmentations"
python $utils/generate_augmentations.py -i $nnUNet_raw/Dataset99_TotalSegMRI/imagesTr -s $nnUNet_raw/Dataset99_TotalSegMRI/labelsTr -o $nnUNet_raw/Dataset100_TotalSegMRI_Aug/imagesTr -g $nnUNet_raw/Dataset100_TotalSegMRI_Aug/labelsTr --labels2image --seg-classes 202-224 18-41,92 200 201
python $utils/cpdir.py $nnUNet_raw/Dataset99_TotalSegMRI $nnUNet_raw/Dataset100_TotalSegMRI_Aug -p "*Ts/*.nii.gz"

echo "Transform labels to images space"
python $utils/transform_labels2images.py -i $nnUNet_raw/Dataset100_TotalSegMRI_Aug/imagesTr -s $nnUNet_raw/Dataset100_TotalSegMRI_Aug/labelsTr -o $nnUNet_raw/Dataset100_TotalSegMRI_Aug/labelsTr

echo "Map labels form TotalSegMRI labels to the datasets specific label"
python $utils/cpdir.py $nnUNet_raw/Dataset100_TotalSegMRI_Aug $nnUNet_raw/Dataset101_TotalSegMRI_step1 -p "imagesT*/*.nii.gz"
python $utils/cpdir.py $nnUNet_raw/Dataset100_TotalSegMRI_Aug $nnUNet_raw/Dataset102_TotalSegMRI_step2 -p "imagesT*/*.nii.gz"
# This make a copy of the labelsTr then later we will map the labels so the odds and evens IVDs are switched
python $utils/cpdir.py $nnUNet_raw/Dataset100_TotalSegMRI_Aug $nnUNet_raw/Dataset102_TotalSegMRI_step2 -p "imagesTr/*.nii.gz" -r "_0000.nii.gz:_o2e_0000.nii.gz"
python $utils/cpdir.py $nnUNet_raw/Dataset100_TotalSegMRI_Aug $nnUNet_raw/Dataset103_TotalSegMRI_full -p "imagesT*/*.nii.gz"

python $utils/map_labels.py -m $resources/labels_maps/nnunet_step1.json -s $nnUNet_raw/Dataset100_TotalSegMRI_Aug/labelsTr -o $nnUNet_raw/Dataset101_TotalSegMRI_step1/labelsTr
python $utils/map_labels.py -m $resources/labels_maps/nnunet_step1.json -s $nnUNet_raw/Dataset100_TotalSegMRI_Aug/labelsTs -o $nnUNet_raw/Dataset101_TotalSegMRI_step1/labelsTs
# This will map the labels to the second input channel
python $utils/map_labels.py -m $resources/labels_maps/nnunet_step2_input.json -s $nnUNet_raw/Dataset100_TotalSegMRI_Aug/labelsTr -o $nnUNet_raw/Dataset102_TotalSegMRI_step2/imagesTr --output-seg-suffix _0001
# This will map the labels to the extra images second input channel so the odd and even IVDs are switched
python $utils/map_labels.py -m $resources/labels_maps/nnunet_step2_input_o2e.json -s $nnUNet_raw/Dataset100_TotalSegMRI_Aug/labelsTr -o $nnUNet_raw/Dataset102_TotalSegMRI_step2/imagesTr --output-seg-suffix _o2e_0001
# This will map the labels to the second input channel for the test set
python $utils/map_labels.py -m $resources/labels_maps/nnunet_step2_input.json -s $nnUNet_raw/Dataset100_TotalSegMRI_Aug/labelsTs -o $nnUNet_raw/Dataset102_TotalSegMRI_step2/imagesTs --output-seg-suffix _0001
python $utils/map_labels.py -m $resources/labels_maps/nnunet_step2.json -s $nnUNet_raw/Dataset100_TotalSegMRI_Aug/labelsTr -o $nnUNet_raw/Dataset102_TotalSegMRI_step2/labelsTr
# This will map the extra images labels so the odd and even IVDs are switched
python $utils/map_labels.py -m $resources/labels_maps/nnunet_step2_o2e.json -s $nnUNet_raw/Dataset100_TotalSegMRI_Aug/labelsTr -o $nnUNet_raw/Dataset102_TotalSegMRI_step2/labelsTr --output-seg-suffix _o2e
python $utils/map_labels.py -m $resources/labels_maps/nnunet_step2.json -s $nnUNet_raw/Dataset100_TotalSegMRI_Aug/labelsTs -o $nnUNet_raw/Dataset102_TotalSegMRI_step2/labelsTs
python $utils/map_labels.py -m $resources/labels_maps/nnunet_full.json -s $nnUNet_raw/Dataset100_TotalSegMRI_Aug/labelsTr -o $nnUNet_raw/Dataset103_TotalSegMRI_full/labelsTr
python $utils/map_labels.py -m $resources/labels_maps/nnunet_full.json -s $nnUNet_raw/Dataset100_TotalSegMRI_Aug/labelsTs -o $nnUNet_raw/Dataset103_TotalSegMRI_full/labelsTs

echo "Copy dataset files"
jq --arg numTraining "$(ls $nnUNet_raw/Dataset101_TotalSegMRI_step1/labelsTr | wc -l)" '.numTraining = ($numTraining|tonumber)' $resources/datasets/dataset_step1.json > $nnUNet_raw/Dataset101_TotalSegMRI_step1/dataset.json
jq --arg numTraining "$(ls $nnUNet_raw/Dataset102_TotalSegMRI_step2/labelsTr | wc -l)" '.numTraining = ($numTraining|tonumber)' $resources/datasets/dataset_step2.json > $nnUNet_raw/Dataset102_TotalSegMRI_step2/dataset.json
jq --arg numTraining "$(ls $nnUNet_raw/Dataset103_TotalSegMRI_full/labelsTr | wc -l)" '.numTraining = ($numTraining|tonumber)' $resources/datasets/dataset_full.json > $nnUNet_raw/Dataset103_TotalSegMRI_full/dataset.json
