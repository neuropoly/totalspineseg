#!/bin/bash

# This script inference tha trained TotalSpineSeg nnUNet model.
# this script get the following parameters in the terminal:
#   1'st param: The input folder containing the .nii.gz images to run the model on.
#   2'nd param: The output folder where the models outputs will be stored.

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

INPUT_FOLDER="$1"
OUTPUT_FOLDER="$2"
if [[ $3 == -step1 ]]; then STEP1=1; else STEP1=0; fi
if [[ $3 == -soft || $4 == -soft ]]; then SOFT=1; else SOFT=0; fi

# set TOTALSPINESEG and TOTALSPINESEG_DATA if not set
TOTALSPINESEG="$(realpath "${TOTALSPINESEG:-totalspineseg}")"
TOTALSPINESEG_DATA="$(realpath "${TOTALSPINESEG_DATA:-data}")"

# ensure the custom nnUNetTrainer is defined in the nnUNet library and add it if it is not
source "$TOTALSPINESEG"/scripts/add_nnunet_trainer.sh

# Set the path to the resources folder
resources="$TOTALSPINESEG"/totalspineseg/resources

# RAM requirement in GB
RAM_REQUIREMENT=8
# Get the number of CPUs, subtract some for system processes
LEAVE_CPUS=0
# set CPU_COUNT to be min of $SLURM_JOB_CPUS_PER_NODE if defined and $(lscpu -p | egrep -v '^#' | wc -l)
CPU_COUNT=${SLURM_JOB_CPUS_PER_NODE:-$(lscpu -p | egrep -v '^#' | wc -l)}
JOBS_FOR_CPUS=$(($CPU_COUNT - $LEAVE_CPUS < 1 ? 1 : $CPU_COUNT - $LEAVE_CPUS ))
# Get the total memory in GB divided by RAM_REQUIREMENT, rounded down to nearest integer, and ensure the value is at least 1
JOBS_FOR_RAMGB=$(( $(awk -v ram_req="$RAM_REQUIREMENT" '/MemTotal/ {print int($2/1024/1024/ram_req < 1 ? 1 : $2/1024/1024/ram_req)}' /proc/meminfo) ))
# Get the minimum of JOBS_FOR_CPUS and JOBS_FOR_RAMGB
JOBS=$(( JOBS_FOR_CPUS < JOBS_FOR_RAMGB ? JOBS_FOR_CPUS : JOBS_FOR_RAMGB ))
# Set the device to cpu if cuda is not available
DEVICE=$(python3 -c "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')")

export nnUNet_def_n_proc=$JOBS
export nnUNet_n_proc_DA=$JOBS

# Set nnunet params
export nnUNet_raw="$TOTALSPINESEG_DATA"/nnUNet/raw
export nnUNet_preprocessed="$TOTALSPINESEG_DATA"/nnUNet/preprocessed
export nnUNet_results="$TOTALSPINESEG_DATA"/nnUNet/results
export nnUNet_exports="$TOTALSPINESEG_DATA"/nnUNet/exports

nnUNetTrainer=nnUNetTrainer_16000epochs
nnUNetPlans=nnUNetPlans
configuration=3d_fullres

echo ""
echo "Running with the following parameters:"
echo "INPUT_FOLDER=${INPUT_FOLDER}"
echo "OUTPUT_FOLDER="${OUTPUT_FOLDER}""
echo "nnUNet_raw=${nnUNet_raw}"
echo "nnUNet_preprocessed=${nnUNet_preprocessed}"
echo "nnUNet_results=${nnUNet_results}"
echo "nnUNet_exports=${nnUNet_exports}"
echo "nnUNetTrainer=${nnUNetTrainer}"
echo "nnUNetPlans=${nnUNetPlans}"
echo "configuration=${configuration}"
echo "JOBS=${JOBS}"
echo "DEVICE=${DEVICE}"
echo ""

FOLD=0
step1_dataset=101
step2_dataset=102

if [ ! -d "$nnUNet_results"/Dataset${step1_dataset}_* ]; then
    nnUNetv2_install_pretrained_model_from_zip "$nnUNet_exports"/Dataset${step1_dataset}_*.zip
fi

if [ ! -d "$nnUNet_results"/Dataset${step2_dataset}_* ]; then
    nnUNetv2_install_pretrained_model_from_zip "$nnUNet_exports"/Dataset${step2_dataset}_*.zip
fi

# Make output dir with copy of the input images convert 4D images to 3D
totalspineseg_average4d -i "${INPUT_FOLDER}" -o "${OUTPUT_FOLDER}"/input --image-suffix "" --output-image-suffix "" -r

# Add _0000 to inputs if not exists to run nnunet
for f in "${OUTPUT_FOLDER}"/input/*.nii.gz; do if [[ "$f" != *_0000.nii.gz ]]; then mv "$f" "${f/.nii.gz/_0000.nii.gz}"; fi; done

#Transform images to canonical space
totalspineseg_reorient_canonical -i "${OUTPUT_FOLDER}"/input -o "${OUTPUT_FOLDER}"/input -r

# resampled to 1x1x1mm
totalspineseg_resample -i "${OUTPUT_FOLDER}"/input -o "${OUTPUT_FOLDER}"/input --image-suffix "" --output-image-suffix "" -r

# Generate preview images
totalspineseg_preview_jpg -i "${OUTPUT_FOLDER}"/input -o "${OUTPUT_FOLDER}"/preview --output-suffix _input -r

# Run step 1 model
# Check if the final checkpoint exists, if not use the latest checkpoint
if [ -f "$nnUNet_results"/Dataset${step1_dataset}_*/${nnUNetTrainer}__${nnUNetPlans}__${configuration}/fold_${FOLD}/checkpoint_final.pth ]; then CHECKPOINT=checkpoint_final.pth; else CHECKPOINT=checkpoint_latest.pth; fi
# Check if the probabilities should be saved
if [ $SOFT -eq 1 ]; then SAVE_PROBABILITIES=--save_probabilities; else SAVE_PROBABILITIES=""; fi
nnUNetv2_predict -d $step1_dataset -i "${OUTPUT_FOLDER}"/input -o "${OUTPUT_FOLDER}"/step1_raw -f $FOLD -c $configuration -p $nnUNetPlans -tr $nnUNetTrainer -npp $JOBS -nps $JOBS -chk $CHECKPOINT -device $DEVICE $SAVE_PROBABILITIES

# Generate preview images for step 1
totalspineseg_preview_jpg -i "${OUTPUT_FOLDER}"/input -s "${OUTPUT_FOLDER}"/step1_raw -o "${OUTPUT_FOLDER}"/preview --output-suffix _step1_raw -r

# Extract the largest connected component of step 1 segmentation after binarization and dilation into the input folder of step 2 model
totalspineseg_largest_component -s "${OUTPUT_FOLDER}"/step1_raw -o "${OUTPUT_FOLDER}"/step1_output --binarize --dilate 5 -r

# Use an iterative algorithm to label IVDs with the definite labels, based on the C2-C3, C7-T1 and L5-S1 IVD labels output by step 1 model.
totalspineseg_iterative_label -s "${OUTPUT_FOLDER}"/step1_output -o "${OUTPUT_FOLDER}"/step1_output --disc-labels 1 2 3 4 5 --init-disc 2:224 5:202 3:219 4:207 --output-disc-step -1 --map-input 6:92 7:201 8:201 9:200 -r

# Fill spinal cancal label to include all non cord spinal canal, this will put the spinal canal label in all the voxels (labeled as a backgroupn) between the spinal canal and the spinal cord.
totalspineseg_fill_canal -s "${OUTPUT_FOLDER}"/step1_output -o "${OUTPUT_FOLDER}"/step1_output --canal-label 201 --cord-label 200 --largest-canal --largest-cord -r

# Transform labels to input images space
totalspineseg_transform_seg2image -i "${OUTPUT_FOLDER}"/input -s "${OUTPUT_FOLDER}"/step1_output -o "${OUTPUT_FOLDER}"/step1_output -r

# Generate preview images for the step 1 labeled
totalspineseg_preview_jpg -i "${OUTPUT_FOLDER}"/input -s "${OUTPUT_FOLDER}"/step1_output -o "${OUTPUT_FOLDER}"/preview --output-suffix _step1_output -r

if [ $SOFT -eq 1 ]; then
    # Extract the spinal cord and spinal canal soft segmentation from the step 1 model output
    totalspineseg_extract_soft -n "${OUTPUT_FOLDER}"/step1_raw -s "${OUTPUT_FOLDER}"/step1_output -o "${OUTPUT_FOLDER}"/spinal_cord --label 8 --seg-labels 200 --dilate 1 -r
    totalspineseg_extract_soft -n "${OUTPUT_FOLDER}"/step1_raw -s "${OUTPUT_FOLDER}"/step1_output -o "${OUTPUT_FOLDER}"/spinal_canal --label 6 --seg-labels 200 201 --dilate 1 -r
    rm "${OUTPUT_FOLDER}"/step1_raw/*.{npz,pkl}
fi

if [ $STEP1 -eq 0 ]; then

    # Copy the original image into the input folder of step 2 model to use as the 1'st channel
    totalspineseg_cpdir "${OUTPUT_FOLDER}"/input "${OUTPUT_FOLDER}"/step2_input -p "*_0000.nii.gz" -r

    # Crop the images to the bounding box of the non-zero part of the step 1 segmentation (with 10 voxels margin), this will also delete images without segmentation
    totalspineseg_crop_image2seg -i "${OUTPUT_FOLDER}"/step2_input -s "${OUTPUT_FOLDER}"/step1_output -o "${OUTPUT_FOLDER}"/step2_input -m 10 -r

    # Transform step 1 segmentation to the cropped images space
    totalspineseg_transform_seg2image -i "${OUTPUT_FOLDER}"/step2_input -s "${OUTPUT_FOLDER}"/step1_output -o "${OUTPUT_FOLDER}"/step2_input --output-seg-suffix _0001 -r

    # Now, we map the IVDs labels from the step1 model output to the odd IVDs to use as the 2'nd channel, this will also delete labels without odd IVDs
    totalspineseg_map_labels -s "${OUTPUT_FOLDER}"/step2_input -o "${OUTPUT_FOLDER}"/step2_input -m "$resources"/labels_maps/nnunet_step2_input.json --seg-suffix _0001 --output-seg-suffix _0001 -r

    # Generate preview images for step 2 input
    totalspineseg_preview_jpg -i "${OUTPUT_FOLDER}"/input -s "${OUTPUT_FOLDER}"/step2_input -o "${OUTPUT_FOLDER}"/preview --seg-suffix _0001 --output-suffix _step2_input -r

    # Remove images without the 2'nd channel
    for f in "${OUTPUT_FOLDER}"/step2_input/*_0000.nii.gz; do if [ ! -f "${f/_0000.nii.gz/_0001.nii.gz}" ]; then rm $f; fi; done

    # Run step 2 model
    # Check if the final checkpoint exists, if not use the latest checkpoint
    if [ -f "$nnUNet_results"/Dataset${step1_dataset}_*/${nnUNetTrainer}__${nnUNetPlans}__${configuration}/fold_${FOLD}/checkpoint_final.pth ]; then CHECKPOINT=checkpoint_final.pth; else CHECKPOINT=checkpoint_latest.pth; fi
    nnUNetv2_predict -d $step2_dataset -i "${OUTPUT_FOLDER}"/step2_input -o "${OUTPUT_FOLDER}"/step2_raw -f $FOLD -c $configuration -p $nnUNetPlans -tr $nnUNetTrainer -npp $JOBS -nps $JOBS -chk $CHECKPOINT -device $DEVICE

    # Generate preview images for step 2
    totalspineseg_preview_jpg -i "${OUTPUT_FOLDER}"/input -s "${OUTPUT_FOLDER}"/step2_raw -o "${OUTPUT_FOLDER}"/preview --output-suffix _step2_raw -r

    # Extract the largest connected component of step 2 segmentation after binarization and dilation into the output folder
    totalspineseg_largest_component -s "${OUTPUT_FOLDER}"/step2_raw -o "${OUTPUT_FOLDER}"/step2_output --binarize --dilate 5 -r

    # Use an iterative algorithm to to assign an individual label value to each vertebrae and IVD in the final segmentation mask.
    totalspineseg_iterative_label -s "${OUTPUT_FOLDER}"/step2_output -o "${OUTPUT_FOLDER}"/step2_output --disc-labels 1 2 3 4 5 6 7 --vertebrea-labels 9 10 11 12 13 14 --vertebrea-extra-labels 8 --init-disc 4:224 7:202 5:219 6:207 --init-vertebrae 11:40 14:17 12:34 13:23 --step-diff-label --step-diff-disc --output-disc-step -1 --output-vertebrea-step -1 --map-output 17:92 --map-input 14:92 15:201 16:201 17:200 -r

    # Fill spinal cancal label to include all non cord spinal canal, this will put the spinal canal label in all the voxels (labeled as a backgroupn) between the spinal canal and the spinal cord.
    totalspineseg_fill_canal -s "${OUTPUT_FOLDER}"/step2_output -o "${OUTPUT_FOLDER}"/step2_output --canal-label 201 --cord-label 200 --largest-canal --largest-cord -r

    # Transform labels to input images space
    totalspineseg_transform_seg2image -i "${OUTPUT_FOLDER}"/input -s "${OUTPUT_FOLDER}"/step2_output -o "${OUTPUT_FOLDER}"/step2_output -r

    # Generate preview images for the final output
    totalspineseg_preview_jpg -i "${OUTPUT_FOLDER}"/input -s "${OUTPUT_FOLDER}"/step2_output -o "${OUTPUT_FOLDER}"/preview --output-suffix _step2_output -r

fi
