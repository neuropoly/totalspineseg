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

# RAM requirement in GB
RAM_REQUIREMENT=8
# Get the number of CPUs, subtract 1, and ensure the value is at least 1
JOBS_FOR_CPUS=$(( $(($(nproc) - 1 < 1 ? 1 : $(nproc) - 1 )) ))
# Get the total memory in GB divided by RAM_REQUIREMENT, rounded down to nearest integer, and ensure the value is at least 1
JOBS_FOR_RAMGB=$(( $(awk -v ram_req="$RAM_REQUIREMENT" '/MemTotal/ {print int($2/1024/1024/ram_req < 1 ? 1 : $2/1024/1024/ram_req)}' /proc/meminfo) ))
# Get the minimum of JOBS_FOR_CPUS and JOBS_FOR_RAMGB
JOBS=$(( JOBS_FOR_CPUS < JOBS_FOR_RAMGB ? JOBS_FOR_CPUS : JOBS_FOR_RAMGB ))

export nnUNet_def_n_proc=$JOBS
export nnUNet_n_proc_DA=$JOBS

# Set nnunet params
export nnUNet_raw=data/nnUNet/nnUNet_raw
export nnUNet_preprocessed=data/nnUNet/nnUNet_preprocessed
export nnUNet_results=data/nnUNet/nnUNet_results

for d in 206 210; do
    # Make nnUNet raw folders
    mkdir -p data/nnUNet/nnUNet_raw/Dataset${d}_TotalSegMRI/imagesTr
    mkdir -p data/nnUNet/nnUNet_raw/Dataset${d}_TotalSegMRI/labelsTr
    
    # Copy dataset file
    cp totalsegmentator-mri/src/totalsegmri/resources/datasets/dataset_${d}.json data/nnUNet/nnUNet_raw/Dataset${d}_TotalSegMRI/dataset.json
done

# Convert from BIDS to nnUNet dataset, loop over each dataset
for ds in spider data-single-subject data-multi-subject whole-spine; do
    # Set segmentation file name based on dataset, 'seg' for 'spider', 'PAM50_seg' otherwise.
    [ "$ds" = "spider" ] && seg="seg" || seg="PAM50_seg"

    # Copy image and label files to the nnUNet dataset structure
    cp data/bids/$ds/sub-*/anat/sub-*.nii.gz data/nnUNet/nnUNet_raw/Dataset206_TotalSegMRI/imagesTr
    cp data/bids/$ds/derivatives/labels/sub-*/anat/sub-*_$seg.nii.gz data/nnUNet/nnUNet_raw/Dataset206_TotalSegMRI/labelsTr
    
    # Format dataset name by removing 'data-' and '-subject' or '-spine'
    dsn=${ds/data-/}; dsn=${dsn/-subject/}; dsn=${dsn/-spine/}

    # Rename nnUNet dataset: replace 'sub-' with dataset name, remove segmentation identifier.
    for f in data/nnUNet/nnUNet_raw/Dataset206_TotalSegMRI/*Tr/sub-*.nii.gz; do mv $f ${f/sub-/${dsn}_}; done
    for f in data/nnUNet/nnUNet_raw/Dataset206_TotalSegMRI/labelsTr/*_$seg.nii.gz; do mv $f ${f/_$seg/}; done
done
# Append '_0000' to the images
for f in data/nnUNet/nnUNet_raw/Dataset206_TotalSegMRI/imagesTr/*.nii.gz; do mv $f ${f/.nii.gz/_0000.nii.gz}; done

# Fix csf label to include all non cord spinal canal, this will put the spinal canal label in all the voxels (labeled as a backgroupn) between the spinal canal and the spinal cord.
python totalsegmentator-mri/src/totalsegmri/utils/fix_csf_label.py -i data/nnUNet/nnUNet_raw/Dataset206_TotalSegMRI/labelsTr -o data/nnUNet/nnUNet_raw/Dataset206_TotalSegMRI/labelsTr

# Crop images and segmentations in the most anteior voxel of the lowest vertebrae in the image or at the lowest voxel of T12-L1 IVD (for SPIDER dataset).
for dsn in single multi whole; do
    python totalsegmentator-mri/src/totalsegmri/utils/generate_croped_images.py -p ${dsn}_ -i data/nnUNet/nnUNet_raw/Dataset206_TotalSegMRI/imagesTr -s data/nnUNet/nnUNet_raw/Dataset206_TotalSegMRI/labelsTr -o data/nnUNet/nnUNet_raw/Dataset206_TotalSegMRI/imagesTr -g data/nnUNet/nnUNet_raw/Dataset206_TotalSegMRI/labelsTr
done
python totalsegmentator-mri/src/totalsegmri/utils/generate_croped_images.py -p spider_ -i data/nnUNet/nnUNet_raw/Dataset206_TotalSegMRI/imagesTr -s data/nnUNet/nnUNet_raw/Dataset206_TotalSegMRI/labelsTr -o data/nnUNet/nnUNet_raw/Dataset206_TotalSegMRI/imagesTr -g data/nnUNet/nnUNet_raw/Dataset206_TotalSegMRI/labelsTr --from-bottom

# Copy from 206 to 210 dataset
for f in data/nnUNet/nnUNet_raw/Dataset206_TotalSegMRI/*Tr/*.nii.gz; do cp $f ${f/206/210}; done

# Map labels to 210 2'nd channel
python totalsegmentator-mri/src/totalsegmri/utils/map_labels.py -m totalsegmentator-mri/src/totalsegmri/resources/labels_maps/nnunet_210_0001.json -i data/nnUNet/nnUNet_raw/Dataset210_TotalSegMRI/labelsTr -o data/nnUNet/nnUNet_raw/Dataset${d}_TotalSegMRI/imagesTr --output-seg-suffix _0001

for d in 206 210; do
    # Map original labels to the dataset specific labels
    python totalsegmentator-mri/src/totalsegmri/utils/map_labels.py -m totalsegmentator-mri/src/totalsegmri/resources/labels_maps/nnunet_${d}.json -i data/nnUNet/nnUNet_raw/Dataset${d}_TotalSegMRI/labelsTr -o data/nnUNet/nnUNet_raw/Dataset${d}_TotalSegMRI/labelsTr
    
    # Preprocess
    nnUNetv2_plan_and_preprocess -d $d -c 3d_fullres -npfp $JOBS -np $JOBS --verify_dataset_integrity
    
    # Train with 5 fold cross-validation
    for f in {0..4}; do
        nnUNetv2_train $d 3d_fullres $f --npz
    done

    # Find best configuration
    nnUNetv2_find_best_configuration $d -np $JOBS -c 3d_fullres -f $f
done
