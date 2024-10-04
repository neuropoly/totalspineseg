#!/bin/bash

# This script calls nnUNetV2's inference to generate sacrum masks using a JSON config file (see totalspineseg/ressources/configs) and saves labels following BIDS' convention.

# The following variables and paths MUST be updated before running the script:
#  - PATH_CONFIG: to the config file `test_sacrum.json`
#  - DERIVATIVE_FOLDER: name of the derivative folder (default=labels)
#  - PATH_REPO: to the repository
#  - PATH_NNUNET_MODEL: to the nnunet model Dataset300_SacrumDataset
#  - AUTHOR: the author

# The totalspineseg environment must be activated before running the script

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
PATH_CONFIG="$TOTALSPINESEG/totalspineseg/resources/configs/test_sacrum.json"

LABEL_SUFFIX="_label-sacrum_seg"
PATH_REPO="$TOTALSPINESEG"
NNUNET_RESULTS="$TOTALSPINESEG_DATA/nnUNet/results/sacrum"
NNUNET_EXPORTS="$TOTALSPINESEG_DATA/nnUNet/exports"
NNUNET_MODEL="Dataset300_SacrumDataset"
PATH_NNUNET_MODEL="$NNUNET_RESULTS/$NNUNET_MODEL/nnUNetTrainer__nnUNetPlans__3d_fullres/"
PROCESS="nnUNet3D"
DERIVATIVE_FOLDER="labels"
FOLD="0"

# Print variables to allow easier debug
echo "See variables:"
echo "PATH_CONFIG: ${PATH_CONFIG}"
echo "DERIVATIVE_FOLDER: ${DERIVATIVE_FOLDER}"
echo "LABEL_SUFFIX: ${LABEL_SUFFIX}"
echo
echo "PATH_REPO: ${PATH_REPO}"
echo "NNUNET_RESULTS: ${NNUNET_RESULTS}"
echo "NNUNET_EXPORTS: ${NNUNET_EXPORTS}"
echo "NNUNET_MODEL: ${NNUNET_MODEL}"
echo "FOLD: ${FOLD}"
echo

# FUNCTIONS
# ======================================================================================================================
# Segment sacrum using our nnUNet model
segment_sacrum_nnUNet(){
  local file_in="$1"
  local file_out="$2"
  local nnunet_model="$3"
  local fold="$4"

  # Call python function
  python3 "${PATH_REPO}"/totalspineseg/utils/run_nnunet_inference_single_subject.py -i "${file_in}" -o "${file_out}" -path-model "${nnunet_model}" -fold "${fold}" -use-gpu -use-best-checkpoint
}

# Generate a json sidecar file
generate_json(){
  local path_json="$1"
  local process="$2"

  # Call python function
  python3 "${PATH_REPO}"/totalspineseg/utils/create_json_sidecar.py -path-json "${path_json}" -process "${process}"
}

# Keep largest component only
keep_largest_component(){
  local seg_in="$1"
  local seg_out="$2"

  # Call python function
  python3 "${PATH_REPO}"/totalspineseg/utils/largest_component_filewise.py --seg-in "${seg_in}" --seg-out "${seg_out}"
}

# Keep largest component only
download_weights(){
  local dataset="$1"
  local download_path="$2"
  local export_path="$2"

  # Call python function
  python3 "${PATH_REPO}"/totalspineseg/utils/download_weights_nnunet.py --nnunet-dataset "${dataset}" --download-folder "${download_path}" --export-folder "${export_path}"
}

# ======================================================================================================================
# SCRIPT STARTS HERE
# ======================================================================================================================
# Fetch datasets path
DATASETS_PATH=$(jq -r '.DATASETS_PATH' "${PATH_CONFIG}")

# Go to folder where data will be copied and processed
cd "$DATASETS_PATH"

# Fetch TESTING files
FILES=$(jq -r '.TESTING[]' "${PATH_CONFIG}")

# Download and install nnUNet weights
download_weights "$NNUNET_MODEL" "$NNUNET_RESULTS" "$NNUNET_EXPORTS"

# Loop across the files
for FILE_PATH in $FILES; do
    BIDS_FOLDER=$(echo "$FILE_PATH" | cut -d / -f 1)
    IN_FILE_NAME=$(echo "$FILE_PATH" | awk -F / '{print $NF}' )
    OUT_FILE_NAME=${IN_FILE_NAME/".nii.gz"/"${LABEL_SUFFIX}.nii.gz"}
    IMG_PATH=${FILE_PATH/"${BIDS_FOLDER}/"/}
    SUB_PATH=${IMG_PATH/"/${IN_FILE_NAME}"/}
    BIDS_DERIVATIVES="${BIDS_FOLDER}/derivatives/${DERIVATIVE_FOLDER}"
    OUT_FOLDER="${BIDS_DERIVATIVES}/${SUB_PATH}"
    OUT_PATH="${OUT_FOLDER}/${OUT_FILE_NAME}"

    # Create DERIVATIVES_FOLDER if missing
    if [[ ! -d ${OUT_FOLDER} ]]; then
        echo "Creating folders $OUT_FOLDER"
        mkdir -p "${OUT_FOLDER}"
    fi
    
    # Generate output segmentation
    echo "Generate segmentation ${FILE_PATH} ${OUT_PATH}"
    segment_sacrum_nnUNet "$FILE_PATH" "$OUT_PATH" "$PATH_NNUNET_MODEL" "$FOLD"
    keep_largest_component "$OUT_PATH" "$OUT_PATH"

    # Generate json sidecar
    JSON_PATH=${OUT_PATH/".nii.gz"/".json"}
    echo "Generate jsonsidecar ${JSON_PATH}"
    generate_json "$JSON_PATH" "$PROCESS"

done

