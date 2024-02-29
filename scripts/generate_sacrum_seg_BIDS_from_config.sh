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
PATH_CONFIG="/home/GRAMES.POLYMTL.CA/p118739/data/config_data/add_sacrum.json"
DERIVATIVE_FOLDER="labels"
LABEL_SUFFIX="_label-sacrum_seg"
AUTHOR="Nathan Molinier"
PROCESS="nnUNet3D"

PATH_REPO="/home/GRAMES.POLYMTL.CA/p118739/code/totalsegmentator-mri"
PATH_NNUNET_MODEL="/home/GRAMES.POLYMTL.CA/p118739/data/nnUNet_results/Dataset300_SacrumDataset/nnUNetTrainer__nnUNetPlans__3d_fullres/"
PATH_NNUNET_ENV="${HOME}/code/nnUNet/nnUNet_env"
FOLD="0"

# Print variables to allow easier debug
echo "See variables:"
echo "PATH_CONFIG: ${PATH_CONFIG}"
echo "DERIVATIVE_FOLDER: ${DERIVATIVE_FOLDER}"
echo "LABEL_SUFFIX: ${LABEL_SUFFIX}"
echo
echo "PATH_REPO: ${PATH_REPO}"
echo "PATH_NNUNET_MODEL: ${PATH_NNUNET_MODEL}"
echo "PATH_NNUNET_ENV: ${PATH_NNUNET_ENV}"
echo "FOLD: ${FOLD}"
echo

# FUNCTIONS
# ======================================================================================================================
# Segment sacrum using our nnUNet model
segment_sacrum_nnUNet(){
  local file_in="$1"
  local file_out="$2"

  # Call python function
  # TODO: the hard-coded path to the conda environment is not ideal.
  "${PATH_NNUNET_ENV}"/bin/python3 "${PATH_REPO}"/scripts/run_inference_single_subject.py -i "${file_in}" -o "${file_out}" -path-model "${PATH_NNUNET_MODEL}" -fold "${FOLD}" -use-gpu
}

# Generate a json sidecar file
generate_json(){
  local path_json="$1"
  local process="$2"
  local author="$3"

  # Call python function
  # TODO: the hard-coded path to the conda environment is not ideal.
  "${PATH_NNUNET_ENV}"/bin/python3 "${PATH_REPO}"/scripts/create_jsonsidecar.py -path-json "${path_json}" -process "${process}" -author "${author}"
}

# Keep largest component only
keep_largest_component(){
  local seg_in="$1"
  local seg_out="$2"

  # Call python function
  # TODO: the hard-coded path to the conda environment is not ideal.
  "${PATH_NNUNET_ENV}"/bin/python3 "${PATH_REPO}"/scripts/keep_largest_component.py --seg-in "${seg_in}" --seg-out "${seg_out}"
}


# ======================================================================================================================
# SCRIPT STARTS HERE
# ======================================================================================================================
# Fetch datasets path
DATASETS_PATH=$(jq -r '.DATASETS_PATH' ${PATH_CONFIG})

# Go to folder where data will be copied and processed
cd "$DATASETS_PATH"

# Fetch TESTING files
FILES=$(jq -r '.TESTING[]' ${PATH_CONFIG})

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
    segment_sacrum_nnUNet "$FILE_PATH" "$OUT_PATH"
    keep_largest_component "$OUT_PATH" "$OUT_PATH"

    # Generate json sidecar
    JSON_PATH=${OUT_PATH/".nii.gz"/".json"}
    echo "Generate jsonsidecar ${JSON_PATH}"
    generate_json "$JSON_PATH" "$PROCESS" "$AUTHOR"

    # Create QC report
    QC_PATH="${DATASETS_PATH}/qc"
    echo "Add ${FILE_PATH} to QC ${QC_PATH}"
    sct_qc -i "$FILE_PATH" -s "$OUT_PATH" -p sct_label_vertebrae -qc "$QC_PATH"

done

