# Uncomment for full verbose
set -x

# Immediately exit if error
set -e -o pipefail

# Exit if user presses CTRL+C (Linux) or CMD+C (OSX)
trap "echo Caught Keyboard Interrupt within script. Exiting now.; exit" INT

# FUNCTIONS
# =========================
# Segment rootlets using our nnUNet model
segment_sacrum_nnUNet(){
  local file="$1"

  echo "Segmenting rootlets using our nnUNet model."
  # Run rootlets segmentation
  # TODO: the hard-coded path to the conda environment is not ideal. But the script also needs to be run inside the
  #  sct_venv environment --> explore if two venvs can be activated at the same time
  ${HOME}/miniconda3/envs/nnunet/bin/python ${PATH_REPO}/packaging/run_inference_single_subject.py -i ${file}.nii.gz -o ${file}_label-rootlet_nnunet.nii.gz -path-model ${PATH_NNUNET_MODEL} -fold ${FOLD}
}

# GET PARAMS
# ======================================================================================================================
# SET DEFAULT VALUES FOR PARAMETERS.
# ----------------------------------------------------------------------------------------------------------------------
PATH_IMG=""
CONFIG_DATA=""
OUTPUT_DIR="results/"
OUTPUT_TXT=""
SUFFIX_SEG="_seg-manual"
VERBOSE=1

# Print variables to allow easier debug
echo "Retrieved variables from from the caller sct_run_batch:"
echo "PATH_DATA: ${PATH_DATA}"
echo "PATH_DATA_PROCESSED: ${PATH_DATA_PROCESSED}"
echo "PATH_RESULTS: ${PATH_RESULTS}"
echo "PATH_LOG: ${PATH_LOG}"
echo "PATH_QC: ${PATH_QC}"
