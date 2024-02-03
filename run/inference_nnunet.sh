#!/bin/bash

# This script inference tha trained TotalSegMRI nnUNet model.
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

INPUT_FOLDER=$1
OUTPUT_FOLDER=$2

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

echo ""
echo "Running with the following parameters:"
echo "INPUT_FOLDER=${INPUT_FOLDER}"
echo "OUTPUT_FOLDER=${OUTPUT_FOLDER}"
echo "JOBS=${JOBS}"
echo ""

# Make output dir with copy of the input
mkdir -p ${OUTPUT_FOLDER}/input
cp ${INPUT_FOLDER}/*.nii.gz ${OUTPUT_FOLDER}/input

# Add _0000 to inputs if not exists to run nnunet
for f in ${OUTPUT_FOLDER}/input/*.nii.gz; do if [[ $f != *_0000.nii.gz ]]; then mv $f ${f/.nii.gz/_0000.nii.gz}; fi; done

# Run the first model with postprocessing
nnUNetv2_predict -d 206 -i ${OUTPUT_FOLDER}/input -o ${OUTPUT_FOLDER}/206 -f  0 1 2 3 4 -c 3d_fullres -npp $JOBS -nps $JOBS
nnUNetv2_apply_postprocessing -i ${OUTPUT_FOLDER}/206 -o ${OUTPUT_FOLDER}/206_pp -pp_pkl_file $nnUNet_results/Dataset206_TotalSegMRI/nnUNetTrainer__nnUNetPlans__3d_fullres/crossval_results_folds_0_1_2_3_4/postprocessing.pkl -np $JOBS -plans_json $nnUNet_results/Dataset206_TotalSegMRI/nnUNetTrainer__nnUNetPlans__3d_fullres/plans.json

# Distinguished odd and even IVDs based on the C2-C3, C7-T1 and L5-S1 IVD labels output by the first model:
# First we will use an iterative algorithm to label IVDs with the definite labels
python totalsegmentator-mri/src/totalsegmri/utils/generate_labels_sequential.py -i ${OUTPUT_FOLDER}/206_pp -o ${OUTPUT_FOLDER}/210_input --output-seg-suffix _0001 --disc-labels 1 2 3 4 --init-disc 2,224 3,219 4,202 --combine-before-label
# Then, we map the IVDs labels to the odd and even IVDs to use as the 2'nd channel of the second model.
python totalsegmentator-mri/src/totalsegmri/utils/map_labels.py -i ${OUTPUT_FOLDER}/210_input -o ${OUTPUT_FOLDER}/210_input -m totalsegmentator-mri/src/totalsegmri/resources/labels_maps/nnunet_210_0001.json --seg-suffix _0001 --output-seg-suffix _0001

# For each of the created odd and even IVDs segmentation, copy the original image to use as the 1'st channel intp the second model input folder
for i in ${OUTPUT_FOLDER}/210_input/*; do
    cp ${OUTPUT_FOLDER}/input/$(basename ${i//0001.nii.gz/0000.nii.gz}) ${i//0001.nii.gz/0000.nii.gz}
done

# Run the second model with postprocessing
nnUNetv2_predict -d 210 -i ${OUTPUT_FOLDER}/210_input -o ${OUTPUT_FOLDER}/210 -f  0 1 2 3 4 -c 3d_fullres -npp $JOBS -nps $JOBS
nnUNetv2_apply_postprocessing -i ${OUTPUT_FOLDER}/210 -o ${OUTPUT_FOLDER}/210_pp -pp_pkl_file $nnUNet_results/Dataset210_TotalSegMRI/nnUNetTrainer__nnUNetPlans__3d_fullres/crossval_results_folds_0_1_2_3_4/postprocessing.pkl -np $JOBS -plans_json $nnUNet_results/Dataset210_TotalSegMRI/nnUNetTrainer__nnUNetPlans__3d_fullres/plans.json

# Use an iterative algorithm to to assign an individual label value to each vertebrae and IVD in the final segmentation mask.
python totalsegmentator-mri/src/totalsegmri/utils/generate_labels_sequential.py -i ${OUTPUT_FOLDER}/210_pp -o ${OUTPUT_FOLDER}/output  --disc-labels 2 3 4 5 6 --vertebrea-labels 8 9 10 11 12 --init-disc 4,224 5,219 6,202 --init-vertebrae 10,41 11,34 12,18

# Use the spinal cord and canal form the first model output.
python totalsegmentator-mri/src/totalsegmri/utils/map_labels.py -i ${OUTPUT_FOLDER}/206_pp -o ${OUTPUT_FOLDER}/output -m totalsegmentator-mri/src/totalsegmri/resources/labels_maps/nnunet_206_canal.json --add-output

# Fix csf label to include all non cord spinal canal, this will put the spinal canal label in all the voxels (labeled as a backgroupn) between the spinal canal and the spinal cord.
python totalsegmentator-mri/src/totalsegmri/utils/fix_csf_label.py -i ${OUTPUT_FOLDER}/output -o ${OUTPUT_FOLDER}/output
