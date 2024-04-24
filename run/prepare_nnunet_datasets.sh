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
mkdir -p $nnUNet_raw/Dataset100_TotalSegMRI/imagesTr_PAM50
mkdir -p $nnUNet_raw/Dataset100_TotalSegMRI/labelsTr_PAM50

# Convert from BIDS to nnUNet dataset, loop over each dataset
for ds in spider data-single-subject data-multi-subject whole-spine; do
    echo "Working on $ds"
    # Format dataset name by removing 'data-' and '-subject' or '-spine'
    dsn=${ds/data-/}; dsn=${dsn/-subject/}; dsn=${dsn/-spine/}

    echo "Copy images and labels into the nnUNet dataset folder"
    python $utils/cpdir.py $bids/$ds $nnUNet_raw/Dataset100_TotalSegMRI/imagesTr -p "sub-*/anat/sub-*.nii.gz" -f
    python $utils/cpdir.py $bids/$ds/derivatives/labels $nnUNet_raw/Dataset100_TotalSegMRI/labelsTr -p "sub-*/anat/sub-*_totalsegmri.nii.gz" -f
    # For single and multi subject datasets, in which there are missing labels get also PAM50_seg
    if [ "$dsn" = "single" ] || [ "$dsn" = "multi" ]; then
        python $utils/cpdir.py $bids/$ds/derivatives/labels $nnUNet_raw/Dataset100_TotalSegMRI/labelsTr_PAM50 -p "sub-*/anat/sub-*_PAM50_seg.nii.gz" -f
    fi
    echo "Replace 'sub-' with dataset name"
    for f in $nnUNet_raw/Dataset100_TotalSegMRI/*/sub-*.nii.gz; do mv $f ${f/sub-/${dsn}_}; done
done

echo "Remove _totalsegmri and PAM50_seg from files name"
for f in $nnUNet_raw/Dataset100_TotalSegMRI/labelsTr/*_totalsegmri.nii.gz; do mv $f ${f/_totalsegmri/}; done
for f in $nnUNet_raw/Dataset100_TotalSegMRI/labelsTr_PAM50/*_PAM50_seg.nii.gz; do mv $f ${f/_PAM50_seg/}; done

echo "Copy images to imagesTr_PAM50 folder"
for f in $nnUNet_raw/Dataset100_TotalSegMRI/labelsTr_PAM50/*.nii.gz; do cp ${f/labelsTr_PAM50/imagesTr} ${f/labelsTr_PAM50/imagesTr_PAM50}; done

echo "Remove images withot segmentation"
for f in $nnUNet_raw/Dataset100_TotalSegMRI/imagesTr*/*.nii.gz; do if [ ! -f ${f/imagesTr/labelsTr} ]; then rm $f; fi; done

echo "Remove _PAM50_seg images and labels when _totalsegmri exists"
for f in $nnUNet_raw/Dataset100_TotalSegMRI/labelsTr/*.nii.gz; do rm -f ${f/labelsTr/labelsTr_PAM50}; rm -f ${f/labelsTr/imagesTr_PAM50}; done

echo "Append '_0000' to the images names"
for f in $nnUNet_raw/Dataset100_TotalSegMRI/imagesTr*/*.nii.gz; do mv $f ${f/.nii.gz/_0000.nii.gz}; done

echo "Fix csf label to include all non cord spinal canal, this will put the spinal canal label in all the voxels (labeled as a backgroupn) between the spinal canal and the spinal cord."
python $utils/fix_csf_label.py -s $nnUNet_raw/Dataset100_TotalSegMRI/labelsTr_PAM50 -o $nnUNet_raw/Dataset100_TotalSegMRI/labelsTr_PAM50

echo "Crop _PAM50 images and segmentations in the most anteior voxel of the lowest vertebrae in the image."
python $utils/generate_croped_images.py -i $nnUNet_raw/Dataset100_TotalSegMRI/imagesTr_PAM50 -s $nnUNet_raw/Dataset100_TotalSegMRI/labelsTr_PAM50 -o $nnUNet_raw/Dataset100_TotalSegMRI/imagesTr_PAM50 -g $nnUNet_raw/Dataset100_TotalSegMRI/labelsTr_PAM50

echo "Move the cropped images and segmentations to the main folder"
for f in $nnUNet_raw/Dataset100_TotalSegMRI/*_PAM50/*.nii.gz; do mv $f ${f/_PAM50/}; done
rm -r $nnUNet_raw/Dataset100_TotalSegMRI/*_PAM50

echo "Transform labels to images space"
python $utils/transform_labels2images.py -i $nnUNet_raw/Dataset100_TotalSegMRI/imagesTr -s $nnUNet_raw/Dataset100_TotalSegMRI/labelsTr -o $nnUNet_raw/Dataset100_TotalSegMRI/labelsTr

echo "Making test folders and moving 10% of the data to test folders"
mkdir -p $nnUNet_raw/Dataset100_TotalSegMRI/imagesTs
mkdir -p $nnUNet_raw/Dataset100_TotalSegMRI/labelsTs

# Make sure each dataset and contrast has 10% of the data in the test folder
for d in spider single multi whole; do
    for c in T1w T2w T2Sw; do
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
python $utils/cpdir.py $nnUNet_raw/Dataset100_TotalSegMRI $nnUNet_raw/Dataset101_TotalSegMRI_step1
python $utils/cpdir.py $nnUNet_raw/Dataset100_TotalSegMRI $nnUNet_raw/Dataset102_TotalSegMRI_step2
python $utils/cpdir.py $nnUNet_raw/Dataset100_TotalSegMRI $nnUNet_raw/Dataset103_TotalSegMRI_full

python $utils/map_labels.py -m $resources/labels_maps/nnunet_step1.json -s $nnUNet_raw/Dataset101_TotalSegMRI_step1/labelsTr -o $nnUNet_raw/Dataset101_TotalSegMRI_step1/labelsTr
python $utils/map_labels.py -m $resources/labels_maps/nnunet_step1.json -s $nnUNet_raw/Dataset101_TotalSegMRI_step1/labelsTs -o $nnUNet_raw/Dataset101_TotalSegMRI_step1/labelsTs
python $utils/map_labels.py -m $resources/labels_maps/nnunet_step2_input.json -s $nnUNet_raw/Dataset102_TotalSegMRI_step2/labelsTr -o $nnUNet_raw/Dataset102_TotalSegMRI_step2/imagesTr --output-seg-suffix _0001
python $utils/map_labels.py -m $resources/labels_maps/nnunet_step2_input.json -s $nnUNet_raw/Dataset102_TotalSegMRI_step2/labelsTs -o $nnUNet_raw/Dataset102_TotalSegMRI_step2/imagesTs --output-seg-suffix _0001
python $utils/map_labels.py -m $resources/labels_maps/nnunet_step2.json -s $nnUNet_raw/Dataset102_TotalSegMRI_step2/labelsTr -o $nnUNet_raw/Dataset102_TotalSegMRI_step2/labelsTr
python $utils/map_labels.py -m $resources/labels_maps/nnunet_step2.json -s $nnUNet_raw/Dataset102_TotalSegMRI_step2/labelsTs -o $nnUNet_raw/Dataset102_TotalSegMRI_step2/labelsTs
python $utils/map_labels.py -m $resources/labels_maps/nnunet_full.json -s $nnUNet_raw/Dataset103_TotalSegMRI_full/labelsTr -o $nnUNet_raw/Dataset103_TotalSegMRI_full/labelsTr
python $utils/map_labels.py -m $resources/labels_maps/nnunet_full.json -s $nnUNet_raw/Dataset103_TotalSegMRI_full/labelsTs -o $nnUNet_raw/Dataset103_TotalSegMRI_full/labelsTs

echo "Copy dataset files"
cp $resources/datasets/dataset_step1.json $nnUNet_raw/Dataset101_TotalSegMRI_step1/dataset.json
cp $resources/datasets/dataset_step2.json $nnUNet_raw/Dataset102_TotalSegMRI_step2/dataset.json
cp $resources/datasets/dataset_full.json $nnUNet_raw/Dataset103_TotalSegMRI_full/dataset.json
