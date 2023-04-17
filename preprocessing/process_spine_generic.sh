#!/bin/bash
#
# Process spine generic data.
#
# Usage :
#   ./process_spine_generic
#
# Manual segmentations or labels should be located under:
# PATH_DATA/derivatives/labels/SUBJECT/<CONTRAST>/
#
# Authors : Victor Baillet

# The following global variables are retrieved from the caller sct_run_batch
# but could be overwritten by uncommenting the lines below:
PATH_DATA_PROCESSED="~/data_processed"
PATH_RESULTS="~/results"
PATH_LOG="~/log"
PATH_QC="~/qc"

# Uncomment for full verbose
set -x

# Immediately exit if error
# set -e -o pipefail

# Exit if user presses CTRL+C (Linux) or CMD+C (OSX)
trap "echo Caught Keyboard Interrupt within script. Exiting now.; exit" INT

# Retrieve input params
SUBJECT=$1

# get starting time:
start='date +%s'

# FUNCTIONS
# =============================================================================

# Check if manual label already exists. If it does, copy it locally. If it does
# not, perform labeling
label_if_does_not_exist(){
    local file="$1"
    local file_seg="$2"
    # Update global variable with segmentation file name
    FILELABEL="${file}_labels"
    FILELABELMANUAL="${PATH_DATA}/derivatives/labels/${SUBJECT}/anat/${FILELABEL}-manuel.nii.gz"
    echo "Looking for manual label: $FILELABELMANUAL"
    if [[ -e $FILELABELMANUAL ]]; then
        echo "Found! Using manual labels."
        rsync -avzh $FILELABELMANUAL ${FILELABEL}.nii.gz
    else
        echo "Not found. Proceeding with automatic labeling."
        # Generate labeled segmentation
        sct_label_vertebrae -i ${file}.nii.gz -s ${file_seg}.nii.gz -c t1
        # Create labels in the cord at C3 and C5 mid-vertebral levels
        sct_label_utils -i ${file_seg}_labeled.nii.gz -vert-body 3,5 -o ${FILELABEL}.nii.gz
    fi
}

# Check if manual segmentation already exists. If it does, copy it locally. If
# it does not, perform seg.
segment_if_does_not_exist(){
    local file="$1"
    local contrast="$2"
    # Find contrast
    if [[ $contrast == "dwi" ]]; then 
        folder_contrast="dwi"
    else
        folder_contrast="anat"
    fi
    # Update global variable with segmentation file name
    FILESEG="${file}_seg"
    FILESEGMANUAL="${PATH_DATA}/derivatives/labels/${SUBJECT}/${folder_contrast}/${FILESEG}-manuel.nii.gz"
    echo
    echo "Looking for manual segmentation: $FILESEGMANUAL"
    if [[ -e $FILESEGMANUAL ]]; then
        echo "Found! Using manual segmentation."
        rsync -avzh $FILESEGMANUAL ${FILESEG}.nii.gz
        sct_qc -i ${file}.nii.gz -s ${FILESEG}.nii.gz -p sct_deepseg_sc -qc ${PATH_QC} -qc-subject ${SUBJECT}
    else
        echo "Not found. Proceeding with automatic segmentation."
        # Segment spinal cord
        sct_deepseg_sc -i ${file}.nii.gz -c $contrast -qc ${PATH_QC} -qc-subject ${SUBJECT}
    fi

}
cd ${SUBJECT}/anat/

# T1w
# -----------------------------------------------------------------------------
file_t1="${SUBJECT}_T1w"
# Reorient to RPI and resample to 1mm iso (supposed to be the effective resolution)
sct_image -i ${file_t1}.nii.gz -setorient RPI -o ${file_t1}_RPI.nii.gz 
sct_resample -i ${file_t1}_RPI.nii.gz -mm 1x1x1 -o ${file_t1}_RPI_r.nii.gz
file_t1="${file_t1}_RPI_r"
# Segment spinal cord (only if it does not exist)
segment_if_does_not_exist $file_t1 "t1"
#FILESEG="${file_t1}_seg"
file_t1_seg=$FILESEG
# Create mid-vertebral levels in the cord (only if it does not exist)
label_if_does_not_exist ${file_t1} ${file_t1_seg}
#FILELABEL="${file_t1}_labels"
file_label=$FILELABEL
# Register to PAM50
sct_register_to_template -i ${file_t1}.nii.gz -s ${file_t1_seg}.nii.gz -l ${file_label}.nii.gz -c t1 -param step=1,type=seg,algo=centermassrot:step=2,type=seg,algo=syn,slicewise=1,smooth=0,iter=5:step=3,type=im,algo=syn,slicewise=1,smooth=0,iter=3 -qc ${PATH_QC} -qc ${PATH_QC} -qc-subject ${SUBJECT}
# Rename warping fields for clarity
mv warp_template2anat.nii.gz warp_template2T1w.nii.gz
mv warp_anat2template.nii.gz warp_T1w2template.nii.gz
# Warp template without the white matter atlas (we don't need it at this point)
sct_warp_template -d ${file_t1}.nii.gz -w warp_template2T1w.nii.gz -ofolder label_T1w
#Generate QC report to assess vertebral labeling
sct_qc -i ${file_t1}.nii.gz -s label_T1w/template/PAM50_levels.nii.gz -p sct_label_vertebrae -qc ${PATH_QC} -qc-subject ${SUBJECT}
