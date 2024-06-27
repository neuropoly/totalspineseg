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

# Set the path to the resources folder
resources=totalspineseg/totalspineseg/resources

INPUT_FOLDER=$1
OUTPUT_FOLDER=$2

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

export nnUNet_def_n_proc=$JOBS
export nnUNet_n_proc_DA=$JOBS

# Set nnunet params
export nnUNet_raw=data/nnUNet/raw
export nnUNet_preprocessed=data/nnUNet/preprocessed
export nnUNet_results=data/nnUNet/results
export nnUNet_exports=data/nnUNet/exports

nnUNetTrainer=nnUNetTrainer_16000epochs
nnUNetPlans=nnUNetPlans
configuration=3d_fullres

echo ""
echo "Running with the following parameters:"
echo "INPUT_FOLDER=${INPUT_FOLDER}"
echo "OUTPUT_FOLDER=${OUTPUT_FOLDER}"
echo "nnUNet_raw=${nnUNet_raw}"
echo "nnUNet_preprocessed=${nnUNet_preprocessed}"
echo "nnUNet_results=${nnUNet_results}"
echo "nnUNet_exports=${nnUNet_exports}"
echo "nnUNetTrainer=${nnUNetTrainer}"
echo "nnUNetPlans=${nnUNetPlans}"
echo "configuration=${configuration}"
echo "JOBS=${JOBS}"
echo ""

# ensure the custom nnUNetTrainer is defined in the nnUNet library and add it if it is not
source "$(dirname "$(realpath "${BASH_SOURCE[0]}")")/add_nnunet_trainer.sh"

FOLD=0
step1_dataset=101
step2_dataset=102

if [ ! -d $nnUNet_results/Dataset${step1_dataset}_* ]; then
    nnUNetv2_install_pretrained_model_from_zip $nnUNet_exports/Dataset${step1_dataset}_*_fold_$FOLD.zip
fi

if [ ! -d $nnUNet_results/Dataset${step2_dataset}_* ]; then
    nnUNetv2_install_pretrained_model_from_zip $nnUNet_exports/Dataset${step2_dataset}_*_fold_$FOLD.zip
fi

# Make output dir with copy of the input images resampled to 1x1x1mm
totalspineseg_generate_resampled_images -i ${INPUT_FOLDER} -o ${OUTPUT_FOLDER}/input --image-suffix "" --output-image-suffix ""

# Add _0000 to inputs if not exists to run nnunet
for f in ${OUTPUT_FOLDER}/input/*.nii.gz; do if [[ $f != *_0000.nii.gz ]]; then mv $f ${f/.nii.gz/_0000.nii.gz}; fi; done

# Run step 1 model
nnUNetv2_predict -d $step1_dataset -i ${OUTPUT_FOLDER}/input -o ${OUTPUT_FOLDER}/step1 -f $FOLD -c $configuration -p $nnUNetPlans -tr $nnUNetTrainer -npp $JOBS -nps $JOBS

# Transform labels to images space
totalspineseg_transform_labels2images -i ${OUTPUT_FOLDER}/input -s ${OUTPUT_FOLDER}/step1 -o ${OUTPUT_FOLDER}/step1

# Distinguished odd and even IVDs based on the C2-C3, C7-T1 and L5-S1 IVD labels output by the first model:
# First we will use an iterative algorithm to label IVDs with the definite labels
totalspineseg_generate_labels_sequential -s ${OUTPUT_FOLDER}/step1 -o ${OUTPUT_FOLDER}/step2_input --output-seg-suffix _0001 --disc-labels 1 2 3 4 5 --init-disc 2:224 5:202 3:219 4:207 --combine-before-label
# Then, we map the IVDs labels to the odd and even IVDs to use as the 2'nd channel of step 2 model.
totalspineseg_map_labels -s ${OUTPUT_FOLDER}/step2_input -o ${OUTPUT_FOLDER}/step2_input -m $resources/labels_maps/nnunet_step2_input.json --seg-suffix _0001 --output-seg-suffix _0001

# For each of the created odd and even IVDs segmentation, copy the original image to use as the 1'st channel in step 2 model input folder
for i in ${OUTPUT_FOLDER}/step2_input/*; do
    cp ${OUTPUT_FOLDER}/input/$(basename ${i//0001.nii.gz/0000.nii.gz}) ${i//0001.nii.gz/0000.nii.gz}
done

# Run step 2 model with postprocessing
nnUNetv2_predict -d $step2_dataset -i ${OUTPUT_FOLDER}/step2_input -o ${OUTPUT_FOLDER}/step2 -f $FOLD -c $configuration -p $nnUNetPlans -tr $nnUNetTrainer -npp $JOBS -nps $JOBS

# Use an iterative algorithm to to assign an individual label value to each vertebrae and IVD in the final segmentation mask.
totalspineseg_generate_labels_sequential -s ${OUTPUT_FOLDER}/step2 -o ${OUTPUT_FOLDER}/output --sacrum-labels 14 --csf-labels 16 --sc-labels 17 --disc-labels 2 3 4 5 6 7 --vertebrea-labels 9 10 11 12 13 14 --init-disc 4:224 7:202 5:219 6:207 --init-vertebrae 11:40 14:17 12:34 13:23 --step-diff-label --step-diff-disc

# Fix csf label to include all non cord spinal canal, this will put the spinal canal label in all the voxels (labeled as a backgroupn) between the spinal canal and the spinal cord.
totalspineseg_fix_csf_label -s ${OUTPUT_FOLDER}/output -o ${OUTPUT_FOLDER}/output --largest-cord --largest-canal

# Generate preview images
totalspineseg_generate_seg_jpg -i ${OUTPUT_FOLDER}/input -o ${OUTPUT_FOLDER}/preview --output-suffix _input
totalspineseg_generate_seg_jpg -i ${OUTPUT_FOLDER}/input -s ${OUTPUT_FOLDER}/step1 -o ${OUTPUT_FOLDER}/preview --output-suffix _step1
totalspineseg_generate_seg_jpg -i ${OUTPUT_FOLDER}/input -s ${OUTPUT_FOLDER}/step2 -o ${OUTPUT_FOLDER}/preview --output-suffix _step2
totalspineseg_generate_seg_jpg -i ${OUTPUT_FOLDER}/input -s ${OUTPUT_FOLDER}/output -o ${OUTPUT_FOLDER}/preview --output-suffix _output
