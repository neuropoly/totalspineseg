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

for d in 206 210; do
    echo "Make nnUNet raw folders ($d)"
    mkdir -p data/nnUNet/nnUNet_raw/Dataset${d}_TotalSegMRI/imagesTr
    mkdir -p data/nnUNet/nnUNet_raw/Dataset${d}_TotalSegMRI/labelsTr
    
    echo "Copy dataset file"
    cp totalsegmentator-mri/src/totalsegmri/resources/datasets/dataset_${d}.json data/nnUNet/nnUNet_raw/Dataset${d}_TotalSegMRI/dataset.json
done

# Convert from BIDS to nnUNet dataset, loop over each dataset
for ds in spider data-single-subject data-multi-subject whole-spine; do
    echo "Working on $ds"

    # Set segmentation file name based on dataset, 'seg' for 'spider', 'PAM50_seg' otherwise.
    [ "$ds" = "spider" ] && seg="seg" || seg="PAM50_seg"

    echo "Copy images and labels into the nnUNet dataset folder"
    cp data/bids/$ds/sub-*/anat/sub-*.nii.gz data/nnUNet/nnUNet_raw/Dataset206_TotalSegMRI/imagesTr
    cp data/bids/$ds/derivatives/labels/sub-*/anat/sub-*_$seg.nii.gz data/nnUNet/nnUNet_raw/Dataset206_TotalSegMRI/labelsTr
    
    # Format dataset name by removing 'data-' and '-subject' or '-spine'
    dsn=${ds/data-/}; dsn=${dsn/-subject/}; dsn=${dsn/-spine/}

    echo "Replace 'sub-' with dataset name"
    for f in data/nnUNet/nnUNet_raw/Dataset206_TotalSegMRI/*Tr/sub-*.nii.gz; do mv $f ${f/sub-/${dsn}_}; done

    echo "Remove _$seg from files name"
    for f in data/nnUNet/nnUNet_raw/Dataset206_TotalSegMRI/labelsTr/*_$seg.nii.gz; do mv $f ${f/_$seg/}; done
done

echo "Append '_0000' to the images names"
for f in data/nnUNet/nnUNet_raw/Dataset206_TotalSegMRI/imagesTr/*.nii.gz; do mv $f ${f/.nii.gz/_0000.nii.gz}; done

echo "Remove images withot segmentation"
for f in data/nnUNet/nnUNet_raw/Dataset206_TotalSegMRI/imagesTr/*.nii.gz; do if [ ! -f data/nnUNet/nnUNet_raw/Dataset206_TotalSegMRI/labelsTr/$(basename ${f/_0000.nii.gz/.nii.gz}) ]; then rm $f; fi; done

echo "Duplicate spider T2Sw X 7, whole X 5 to balance the dataset."
for f in data/nnUNet/nnUNet_raw/Dataset206_TotalSegMRI/imagesTr/spider*_T2Sw_0000.nii.gz; do for i in {1..6}; do cp $f ${f/_0000.nii.gz/_${i}_0000.nii.gz}; done; done
for f in data/nnUNet/nnUNet_raw/Dataset206_TotalSegMRI/labelsTr/spider*_T2Sw.nii.gz; do for i in {1..6}; do cp $f ${f/.nii.gz/_${i}.nii.gz}; done; done
for f in data/nnUNet/nnUNet_raw/Dataset206_TotalSegMRI/imagesTr/whole*_0000.nii.gz; do for i in {1..4}; do cp $f ${f/_0000.nii.gz/_${i}_0000.nii.gz}; done; done
for f in data/nnUNet/nnUNet_raw/Dataset206_TotalSegMRI/labelsTr/whole*.nii.gz; do for i in {1..4}; do cp $f ${f/.nii.gz/_${i}.nii.gz}; done; done

echo "Fix csf label to include all non cord spinal canal, this will put the spinal canal label in all the voxels (labeled as a backgroupn) between the spinal canal and the spinal cord."
python totalsegmentator-mri/src/totalsegmri/utils/fix_csf_label.py -s data/nnUNet/nnUNet_raw/Dataset206_TotalSegMRI/labelsTr -o data/nnUNet/nnUNet_raw/Dataset206_TotalSegMRI/labelsTr

echo "Crop images and segmentations in the most anteior voxel of the lowest vertebrae in the image or at the lowest voxel of T12-L1 IVD (for SPIDER dataset)."
for dsn in single multi whole; do
    python totalsegmentator-mri/src/totalsegmri/utils/generate_croped_images.py -p ${dsn}_ -i data/nnUNet/nnUNet_raw/Dataset206_TotalSegMRI/imagesTr -s data/nnUNet/nnUNet_raw/Dataset206_TotalSegMRI/labelsTr -o data/nnUNet/nnUNet_raw/Dataset206_TotalSegMRI/imagesTr -g data/nnUNet/nnUNet_raw/Dataset206_TotalSegMRI/labelsTr
done
python totalsegmentator-mri/src/totalsegmri/utils/generate_croped_images.py -p spider_ -i data/nnUNet/nnUNet_raw/Dataset206_TotalSegMRI/imagesTr -s data/nnUNet/nnUNet_raw/Dataset206_TotalSegMRI/labelsTr -o data/nnUNet/nnUNet_raw/Dataset206_TotalSegMRI/imagesTr -g data/nnUNet/nnUNet_raw/Dataset206_TotalSegMRI/labelsTr --from-bottom

echo "Copy from 206 to 210 dataset"
for f in data/nnUNet/nnUNet_raw/Dataset206_TotalSegMRI/*Tr/*.nii.gz; do cp $f ${f/206/210}; done

echo "Map labels to 210 2'nd channel"
python totalsegmentator-mri/src/totalsegmri/utils/map_labels.py -m totalsegmentator-mri/src/totalsegmri/resources/labels_maps/nnunet_210_0001.json -s data/nnUNet/nnUNet_raw/Dataset210_TotalSegMRI/labelsTr -o data/nnUNet/nnUNet_raw/Dataset210_TotalSegMRI/imagesTr --output-seg-suffix _0001

for d in 206 210; do
    echo "Map original labels to the dataset specific labels"
    python totalsegmentator-mri/src/totalsegmri/utils/map_labels.py -m totalsegmentator-mri/src/totalsegmri/resources/labels_maps/nnunet_${d}.json -s data/nnUNet/nnUNet_raw/Dataset${d}_TotalSegMRI/labelsTr -o data/nnUNet/nnUNet_raw/Dataset${d}_TotalSegMRI/labelsTr
done
