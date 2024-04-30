#!/bin/bash

# Check for -h option
if [ "$1" = "-h" ]; then

    echo " This script is designed to register spinal cord MRI data to the PAM50 template. "
    echo " The data is expected to be organized in the BIDS format. The main steps include:"
    echo ""
    echo " 1. Create spinal cord segmentation and initial automatic vertebrae label "
    echo "    (use manual segmentation if available)."
    echo " 2. For each subject, use a GUI to manually correct or confirm the vertebrae labels."
    echo " 3. Register the image to the PAM50 template and bring the PAM50 segmentation file into the image space."
    echo ""
    echo " The script takes several command-line arguments for customization:"
    echo "   -d: BIDS data folder (default: ".")."
    echo "   -o: Output folder (default: "output")."
    echo "   -s: PAM50 segmentation file (default: "../PAM50_seg.nii.gz")."
    echo "   -r: Overwrite existing files (0/1). 0: Do not overwrite, 1: Run again even if files exist (default: 0)."
    echo "   -c: Cleanup (0/1). 0: Do not Cleanup, 1: Remove all files except the necessary segmentation and label files. (default: 0)."
    echo "   -l: Log folder (default: "output/logs")."
    echo "   -j: Number of jobs you want to run in parallel. (default: half the number of cores available on the system)."
    echo "   -v: Verbose (0/1). 0: Display only errors/warnings, 1: Errors/warnings + info messages (default: 1)."
    echo ""
    echo " After running the script, a <SUBJECT>_PAM50_seg.nii.gz file will be created in the SUBJECT output folder "
    echo " with the segmentation."
    echo ""
    echo " The script has three main parts:"
    echo ""
    echo " 1. Create spinal cord segmentation and initial automatic vertebrae label:"
    echo "    - Copies the source data to the output folder, preserving the BIDS structure."
    echo "    - If manual labels are available, they are copied to the output folder."
    echo "    - Spinal cord segmentation is created using sct_deepseg_sc from the Spinal Cord Toolbox (SCT)."
    echo "    - Vertebrae labels are created using sct_label_vertebrae."
    echo ""
    echo " 2. For each subject, use a GUI to manually correct or confirm the vertebrae labels:"
    echo "    - The script opens SCT's interactive labeling tool (sct_label_utils) for each subject, "
    echo "      allowing the user to manually correct vertebrae labels."
    echo ""
    echo " 3. Register the image to the PAM50 template:"
    echo "    - Registration is performed using sct_register_to_template."
    echo "    - The PAM50 segmentation file is moved into the image space using sct_apply_transfo."
    echo "    - Unnecessary intermediate files are removed using the find command, preserving only the final "
    echo "      segmentation and label files."
    echo ""
    echo " Note: This script requires the Spinal Cord Toolbox (SCT) to be installed."
    echo ""
    echo ""
    echo " Data organization - follow the BIDS structure:"
    echo " data"
    echo " ├── derivatives"
    echo " │   └── labels"
    echo " │       ├── sub-errsm38"
    echo " │       │   └── anat"
    echo " │       │       ├── sub-errsm38_T1w_labels-disc-manual.json"
    echo " │       │       ├── sub-errsm38_T1w_labels-disc-manual.json"
    echo " │       │       ├── sub-errsm38_T1w_seg.json"
    echo " │       │       └── sub-errsm38_T1w_seg-manual.nii.gz"
    echo " │       └── sub-errsm38"
    echo " │           └── anat"
    echo " │               ├── sub-errsm38_T1w_seg-manual.json"
    echo " │               └── sub-errsm38_T1w_seg-manual.nii.gz"
    echo " ├── sub-errsm37"
    echo " │   └── anat"
    echo " │       ├── sub-errsm37_T2w.json"
    echo " │       ├── sub-errsm37_T2w.nii.gz"
    echo " │       ├── sub-errsm37_T1w.json"
    echo " │       └── sub-errsm37_T1w.nii.gz"
    echo " └── sub-errsm38"
    echo "     └── anat"
    echo "         ├── sub-errsm38_T1w.json"
    echo "         └── sub-errsm38_T1w.nii.gz"
    echo ""
    echo " Dependencies"
    echo " Spinal Cord Toolbox (SCT) - https://spinalcordtoolbox.com/"
    echo ""
    echo " Usage:"
    echo "   ./reg2pam50.sh"
    echo "     [-d <BIDS data folder (default: .).>] \\"
    echo "     [-o <Output folder (default: output).>] \\"
    echo "     [-s <PAM50 segmentation file (default: ../PAM50_seg.nii.gz).>] \\"
    echo "     [-r <Overwrite (0/1). 0: Do not Overwrite existing files, 1: Run again even if files exist (default: 0).>] \\"
    echo "     [-c <Cleanup (0/1). 0: Do not Cleanup, 1: Remove all files except the necessary segmentation and label files. (default: 0).>] \\"
    echo "     [-l <Log folder (default: output/logs).>] \\"
    echo "     [-j <Number of jobs you want to run in parallel. (default: The minium between the number of cores available on the system minus one and the RAM divided by 8GB).>] \\"
    echo "     [-v <Verbose (0/1). 0: Display only errors/warnings, 1: Errors/warnings + info messages (default: 1).>]"

    exit 0
fi

# BASH SETTINGS
# ======================================================================================================================

# Uncomment for full verbose
# set -v

# Immediately exit if error
set -e

# Exit if user presses CTRL+C (Linux) or CMD+C (OSX)
trap "echo Caught Keyboard Interrupt within script. Exiting now.; exit" INT

# GET PARAMS
# ======================================================================================================================

# Set default values for parameters.
# ----------------------------------------------------------------------------------------------------------------------
DATA_DIR="."
OUTPUT_DIR="output"
PAM50_SEG="../PAM50_seg.nii.gz"
OVERWRITE=0
CLEANUP=0
LOG_DIR="output/logs"
# RAM requirement in GB
RAM_REQUIREMENT=8
# Get the number of CPUs, subtract some for system processes
LEAVE_CPUS=1
JOBS_FOR_CPUS=$(( $(($(lscpu -p | egrep -v '^#' | wc -l) - $LEAVE_CPUS < 1 ? 1 : $(lscpu -p | egrep -v '^#' | wc -l) - $LEAVE_CPUS )) ))
# Get the total memory in GB divided by 10, rounded down to nearest integer, and ensure the value is at least 1
JOBS_FOR_RAMGB=$(( $(awk -v ram_req="$RAM_REQUIREMENT" '/MemTotal/ {print int($2/1024/1024/ram_req < 1 ? 1 : $2/1024/1024/ram_req)}' /proc/meminfo) ))
# Get the minimum of JOBS_FOR_CPUS and JOBS_FOR_RAMGB
JOBS=$(( JOBS_FOR_CPUS < JOBS_FOR_RAMGB ? JOBS_FOR_CPUS : JOBS_FOR_RAMGB ))
VERBOSE=1

# Get command-line parameters to override default values.
# ----------------------------------------------------------------------------------------------------------------------
while getopts d:o:s:r:c:l:j:v: flag
do
  case "${flag}" in
    d) DATA_DIR=${OPTARG};;
    o) OUTPUT_DIR=${OPTARG};;
    s) PAM50_SEG=${OPTARG};;
    r) OVERWRITE=${OPTARG};;
    c) CLEANUP=${OPTARG};;
    l) LOG_DIR=${OPTARG};;
    j) JOBS=${OPTARG};;
    v) VERBOSE=${OPTARG};;
  esac
done

# Validate the given parameters.
# ----------------------------------------------------------------------------------------------------------------------

# Ensure the data directory exists.
if [[ ! -d ${DATA_DIR} ]]; then
    echo "Folder not found ${DATA_DIR}"
    exit 1
fi

# Ensure the output directory exists, creating it if necessary.
mkdir -p ${OUTPUT_DIR}
if [[ ! -d ${OUTPUT_DIR} ]]; then
    echo "Folder not found ${OUTPUT_DIR}"
    exit 1
fi

# Ensure the PAM50 segmentation file exists.
if [[ ! -f ${PAM50_SEG} ]]; then
    echo "File not found ${PAM50_SEG}"
    exit 1
fi

# Ensure the OVERWRITE parameter is either 0 or 1.
if [[ ${OVERWRITE} != 0 ]] && [[ ${OVERWRITE} != 1 ]]; then
    echo "Error: -r param must be 0 or 1."
    exit 1
fi

# Ensure the CLEANUP parameter is either 0 or 1.
if [[ ${CLEANUP} != 0 ]] && [[ ${CLEANUP} != 1 ]]; then
    echo "Error: -c param must be 0 or 1."
    exit 1
fi

# Ensure the log directory exists, creating it if necessary.
mkdir -p ${LOG_DIR}
if [[ ! -d ${LOG_DIR} ]]; then
    echo "Folder not found ${LOG_DIR}"
    exit 1
fi

# Ensure the VERBOSE parameter is either 0 or 1.
if [[ ${VERBOSE} != 0 ]] && [[ ${VERBOSE} != 1 ]]; then
    echo "Error: -v param must be 0 or 1."
    exit 1
fi

# Get full path for all parameters.
# ----------------------------------------------------------------------------------------------------------------------

DATA_DIR=$(realpath "${DATA_DIR}")
OUTPUT_DIR=$(realpath "${OUTPUT_DIR}")
PAM50_SEG=$(realpath "${PAM50_SEG}")
LOG_DIR=$(realpath "${LOG_DIR}")

# Export all parameters.
# ----------------------------------------------------------------------------------------------------------------------

export DATA_DIR OUTPUT_DIR PAM50_SEG OVERWRITE CLEANUP LOG_DIR JOBS VERBOSE

# Print the parameters if VERBOSE is enabled.
# ----------------------------------------------------------------------------------------------------------------------

if [[ ${VERBOSE} == 1 ]]; then
    echo ""
    echo "Running with the following parameters:"
    echo "DATA_DIR=${DATA_DIR}"
    echo "OUTPUT_DIR=${OUTPUT_DIR}"
    echo "PAM50_SEG=${PAM50_SEG}"
    echo "OVERWRITE=${OVERWRITE}"
    echo "CLEANUP=${CLEANUP}"
    echo "LOG_DIR=${LOG_DIR}"
    echo "JOBS=${JOBS}"
    echo "VERBOSE=${VERBOSE}"
    echo ""
fi

# SCRIPT STARTS HERE
# ======================================================================================================================

# Define the contrasts to be processed.
export CONTRASTS="t1 t2"

# Get running start date and  time for log
export start_date=$(date +%Y%m%d%H%M%S)

# Step 1 - Create spinal cord segmentation and initial automatic vertebrae label.
# ----------------------------------------------------------------------------------------------------------------------

# Change to the data directory.
cd ${DATA_DIR}

seg_sc_and_auto_label_vertebrae () {

    SUBJECT=$1

    # Copy the source data to the output data folder.
    cp -r -u ${DATA_DIR}/${SUBJECT} ${OUTPUT_DIR}
    # If there is no anat folder in the subject folder, skip to the next subject.
    if [[ ! -d ${OUTPUT_DIR}/${SUBJECT}/anat ]]; then
        continue
    fi
    # Copy manual labels to the output folder, if available.
    if [[ -d ${DATA_DIR}/derivatives/labels/${SUBJECT} ]]; then
        cp -r -u ${DATA_DIR}/derivatives/labels/${SUBJECT} ${OUTPUT_DIR}
    fi
    # Change the working directory to the output subject folder.
    cd ${OUTPUT_DIR}/${SUBJECT}/anat
    # If a manual spinal cord segmentation file exists, use it instead of the automatic one.
    for f in sub-*_*w_seg-manual.nii.gz; do
        if [[ -e ${f} ]]; then
            # Remove '-manual' from segmentation file
            mv ${f} "${f%-manual.nii.gz}.nii.gz" || :
        fi
    done
    # Process each contrast.
    for c in ${CONTRASTS[@]}; do
        if [[ -f ${SUBJECT}_${c^^}w.nii.gz ]]; then
            # Get disks label of T1 from T2 and vice versa
            cn=$( [ "$c" == "t1" ] && echo "t2" || echo "t1" )
            if [[ ! -f ${SUBJECT}_${c^^}w_labels-disc-manual.nii.gz && -f ${SUBJECT}_${c^^}w_seg.nii.gz && -f ${SUBJECT}_${cn^^}w_labels-disc-manual.nii.gz && -f ${SUBJECT}_${cn^^}w_seg.nii.gz && -f ${SUBJECT}_${cn^^}w.nii.gz ]] ; then
                sct_register_multimodal -i ${SUBJECT}_${cn^^}w.nii.gz -d ${SUBJECT}_${c^^}w.nii.gz -iseg ${SUBJECT}_${cn^^}w_seg.nii.gz -dseg ${SUBJECT}_${c^^}w_seg.nii.gz  -param step=1,type=seg,algo=slicereg,smooth=1 -v ${VERBOSE}
                sct_apply_transfo -i ${SUBJECT}_${cn^^}w_labels-disc-manual.nii.gz -d ${SUBJECT}_${c^^}w.nii.gz -w warp_${SUBJECT}_${cn^^}w2${SUBJECT}_${c^^}w.nii.gz -o ${SUBJECT}_${c^^}w_labels-disc-manual.nii.gz -x label -v ${VERBOSE}
                sct_label_vertebrae -i ${SUBJECT}_${c^^}w.nii.gz -s ${SUBJECT}_${c^^}w_seg.nii.gz -discfile ${SUBJECT}_${c^^}w_labels-disc-manual.nii.gz -c ${c} -v ${VERBOSE} -qc ${OUTPUT_DIR}/qc
            fi
            # Create spinal cord segmentation file if it does not exist or if the OVERWRITE option is enabled.
            if [[ ! -f ${SUBJECT}_${c^^}w_seg.nii.gz ]] || [[ ${OVERWRITE} == 1 ]]; then
                sct_deepseg_sc -i ${SUBJECT}_${c^^}w.nii.gz -c ${c} -v ${VERBOSE} -qc ${OUTPUT_DIR}/qc
            fi
            # Label each vertebra if a labeled disc file does not exist and if the manual labels file does not exist, or if the OVERWRITE option is enabled.
            if [[ ! -f ${SUBJECT}_${c^^}w_seg_labeled_discs.nii.gz && ! -f ${SUBJECT}_${c^^}w_labels-disc-manual.nii.gz ]] || [[ ${OVERWRITE} == 1 ]]; then
                sct_label_vertebrae -i ${SUBJECT}_${c^^}w.nii.gz -s ${SUBJECT}_${c^^}w_seg.nii.gz -c ${c} -v ${VERBOSE} || :
            fi
        fi
    done

}

export -f  seg_sc_and_auto_label_vertebrae

# Process all subjects in the data directory in parallel.
ls -d sub-* | parallel -j ${JOBS} "seg_sc_and_auto_label_vertebrae {} >> ${LOG_DIR}/{}_$start_date.log"

# Step 2 - For each subject, use the GUI to manually correct or confirm the vertebrae labels.
# ----------------------------------------------------------------------------------------------------------------------

# Change to the data directory.
cd ${DATA_DIR}

# Iterate over all subjects in the data directory.
for SUBJECT in sub-*; do
    # If there is no anat folder in the subject folder, skip to the next subject.
    if [[ ! -d ${OUTPUT_DIR}/${SUBJECT}/anat ]]; then
        continue
    fi
    # Change the working directory to the output subject folder.
    cd ${OUTPUT_DIR}/${SUBJECT}/anat
    # Process each contrast.
    for c in ${CONTRASTS[@]}; do
        if [[ -f ${SUBJECT}_${c^^}w.nii.gz ]]; then
            # Perform manual correction of vertebrae labels if the manual labels file does not exist or if the OVERWRITE option is enabled.
            if [[ ! -f ${SUBJECT}_${c^^}w_labels-disc-manual.nii.gz ]] || [[ ${OVERWRITE} == 1 ]]; then
                if [[ -f ${SUBJECT}_${c^^}w_seg_labeled_discs.nii.gz ]]; then
                    # If the labeled discs file exists, use it as input for the label utils command.
                    sct_label_utils -i ${SUBJECT}_${c^^}w.nii.gz -create-viewer 1:21 -ilabel ${SUBJECT}_${c^^}w_seg_labeled_discs.nii.gz -v ${VERBOSE} -o ${SUBJECT}_${c^^}w_labels-disc-manual.nii.gz -msg "Place labels at the posterior tip of each inter-vertebral disc. E.g. Label 3: C2/C3, Label 4: C3/C4, etc." || :
                else
                    # If the labeled discs file does not exist, run the label utils command without the input label file.
                    sct_label_utils -i ${SUBJECT}_${c^^}w.nii.gz -create-viewer 1:21 -v ${VERBOSE} -o ${SUBJECT}_${c^^}w_labels-disc-manual.nii.gz -msg "Place labels at the posterior tip of each inter-vertebral disc. E.g. Label 3: C2/C3, Label 4: C3/C4, etc." || :
                fi
            fi
        fi
    done
done

# Step 3 - Register the image to the PAM50 template.
# ----------------------------------------------------------------------------------------------------------------------

# Change to the data directory.
cd ${DATA_DIR}

register_to_pam50 () {

    SUBJECT=$1

    # If there is no anat folder in the subject folder, skip to the next subject.
    if [[ ! -d ${OUTPUT_DIR}/${SUBJECT}/anat ]]; then
        continue
    fi
    # Change the working directory to the output subject folder.
    cd ${OUTPUT_DIR}/${SUBJECT}/anat
    # Process each contrast.
    for c in ${CONTRASTS[@]}; do
        if [[ -f ${SUBJECT}_${c^^}w.nii.gz ]]; then
            # Continue if no manual disk labels file is found.
            if [[ ! -f ${SUBJECT}_${c^^}w_labels-disc-manual.nii.gz ]]; then
                echo "File not found ${SUBJECT}_${c^^}w_labels-disc-manual.nii.gz"
            # Register to PAM50 template and create PAM50 segmentation file if it does not exist or if the OVERWRITE option is enabled.
            elif [[ ! -f ${SUBJECT}_${c^^}w_PAM50_seg.nii.gz ]] || [[ ${OVERWRITE} == 1 ]]; then
                # Register to PAM50 template.
                if [[ ! -f ${SUBJECT}_${c^^}w/warp_template2anat.nii.gz ]] || [[ ${OVERWRITE} == 1 ]]; then
                    # Ensure segmentation in the same space as the image
                    sct_register_multimodal -i ${SUBJECT}_${c^^}w_seg.nii.gz -d ${SUBJECT}_${c^^}w.nii.gz -identity 1 -x nn
                    sct_register_to_template -i ${SUBJECT}_${c^^}w.nii.gz -s ${SUBJECT}_${c^^}w_seg_reg.nii.gz -ldisc ${SUBJECT}_${c^^}w_labels-disc-manual.nii.gz -ofolder ${SUBJECT}_${c^^}w -c ${c} -qc ${OUTPUT_DIR}/qc
                fi
                # Move PAM50 segmentation file into the image space.
                sct_apply_transfo -i ${PAM50_SEG} -d ${SUBJECT}_${c^^}w.nii.gz -w ${SUBJECT}_${c^^}w/warp_template2anat.nii.gz -x nn -o ${SUBJECT}_${c^^}w_PAM50_seg.nii.gz
            fi
        fi
    done
    # Cleanup - Remove all files except the necessary segmentation and label files.
    if [[ ${CLEANUP} == 1 ]]; then
        find . -type f -not -regex "\./${SUBJECT}_[^_]+w_\(seg\|labels-disc-manual\|PAM50_seg\)\.nii\.gz" -exec rm -f {} \;
        find . -mindepth 1 -maxdepth 1 -type d -exec rm -rf {} \;
    fi

}

export -f  register_to_pam50

# Process all subjects in the data directory in parallel.
ls -d sub-* | parallel -j ${JOBS} "register_to_pam50 {} >> ${LOG_DIR}/{}_$start_date.log 2>&1; if [ ! -s ${LOG_DIR}/{}_$start_date.log ]; then rm ${LOG_DIR}/{}_$start_date.log; fi"
