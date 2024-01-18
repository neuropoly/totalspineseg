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
PATH_NNUNET_DATA="/home/GRAMES.POLYMTL.CA/p118739/data/nnUNet_raw/Dataset100_TotalSegMRI"
NNUNET_IMG_FOLDER="imagesTr"
NNUNET_LABEL_FOLDER="labelsTr"
SACRUM_FOLDER="sacrum-seg"
OUTPUT_FOLDER="final-seg"

PATH_REPO="/home/GRAMES.POLYMTL.CA/p118739/code/totalsegmentator-mri"
PATH_NNUNET_MODEL="/home/GRAMES.POLYMTL.CA/p118739/data/nnUNet_results/Dataset300_SacrumDataset/nnUNetTrainer__nnUNetPlans__3d_fullres/"
PATH_NNUNET_ENV="${HOME}/code/nnUNet/nnUNet_env"
FOLD="0"

SACRUM_VALUE="50"

# Print variables to allow easier debug
echo "See variables:"
echo "PATH_NNUNET_DATA: ${PATH_NNUNET_DATA}"
echo "NNUNET_IMG_FOLDER: ${NNUNET_IMG_FOLDER}"
echo "NNUNET_LABEL_FOLDER: ${NNUNET_LABEL_FOLDER}"
echo "SACRUM_FOLDER: ${SACRUM_FOLDER}"
echo "OUTPUT_FOLDER: ${OUTPUT_FOLDER}"
echo
echo "PATH_REPO: ${PATH_REPO}"
echo "PATH_NNUNET_MODEL: ${PATH_NNUNET_MODEL}"
echo "PATH_NNUNET_ENV: ${PATH_NNUNET_ENV}"
echo "FOLD: ${FOLD}"
echo
echo "SACRUM_VALUE: ${SACRUM_VALUE}"
echo 

# FUNCTIONS
# ======================================================================================================================
# Segment sacrum using our nnUNet model
segment_sacrum_nnUNet(){
  local file_in="$1"
  local file_out="$2"

  echo "Segmenting sacrum using our nnUNet model."
  # Run rootlets segmentation
  # TODO: the hard-coded path to the conda environment is not ideal.
  "${PATH_NNUNET_ENV}"/bin/python3 "${PATH_REPO}"/scripts/run_inference_single_subject.py -i "${file_in}" -o "${file_out}" -path-model "${PATH_NNUNET_MODEL}" -fold "${FOLD}" -use-gpu
}

# Add sacrum mask to the existing segmentation
add_sacrum_to_seg(){
  local file_mask="$1"
  local file_seg="$2"
  local file_out="$3"

  echo "Segmenting sacrum using our nnUNet model."
  # Run rootlets segmentation
  # TODO: the hard-coded path to the conda environment is not ideal.
  "${PATH_NNUNET_ENV}"/bin/python3 "${PATH_REPO}"/scripts/add_mask_to_seg.py -i "${file_mask}" -seg "${file_seg}" -o "${file_out}" -val "${SACRUM_VALUE}"
}

# ======================================================================================================================
# SCRIPT STARTS HERE
# ======================================================================================================================
# Go to folder where data will be copied and processed
cd $PATH_NNUNET_DATA

# Create SACRUM_FOLDER if missing
if [[ ! -d ${SACRUM_FOLDER} ]]; then
    echo "Creating folder ${PATH_NNUNET_DATA}/${SACRUM_FOLDER}"
    mkdir ${SACRUM_FOLDER}
fi

# Create OUTPUT_FOLDER if missing
if [[ ! -d ${OUTPUT_FOLDER} ]]; then
    echo "Creating folder ${PATH_NNUNET_DATA}/${OUTPUT_FOLDER}"
    mkdir ${OUTPUT_FOLDER}
fi

# Compute nnUNet segmentation for each image within the NNUNET_IMG_FOLDER
for img in $(find "$NNUNET_IMG_FOLDER" -name "spider*" -printf '%P\n'  -maxdepth 1 -mindepth 1 -printf '%P\n' | head -n 4); do
    img_in="${PATH_NNUNET_DATA}/${NNUNET_IMG_FOLDER}/${img}"
    img_out="${PATH_NNUNET_DATA}/${SACRUM_FOLDER}/${img/0000/label-sacrum_seg}"
    segment_sacrum_nnUNet "$img_in" "$img_out"
done

# Create a new segmentation by adding the sacrum to the previous segmentation
for lbl in $(find "$SACRUM_FOLDER"  -maxdepth 1 -mindepth 1 -printf '%P\n' | head -n 4); do
    label_sacrum="${PATH_NNUNET_DATA}/${SACRUM_FOLDER}/${lbl}"
    label_seg="${PATH_NNUNET_DATA}/${NNUNET_IMG_FOLDER}/${lbl/_label-sacrum_seg/}"
    label_out="${PATH_NNUNET_DATA}/${OUTPUT_FOLDER}/${lbl/_label-sacrum_seg/}"
    add_sacrum_to_seg "$label_sacrum" "$label_seg" "$label_out"
done

