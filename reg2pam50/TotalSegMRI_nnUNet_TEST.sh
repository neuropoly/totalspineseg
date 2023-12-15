#!/bin/bash
#SBATCH --account=def-jcohen
#SBATCH --time=00-00:30:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gpus-per-node=a100:1
#SBATCH --mail-user=Yehuda.Warszawer@sheba.health.gov.il
#SBATCH --mail-type=ALL

# Get resources
# salloc --account=def-jcohen --time=0:10:00 --cpus-per-task=16 --mem=128G --gpus-per-node=a100:1

# Get params
########################################
DATASET_ID=$1
# DATASET_ID=100
FOLD=$2
# FOLDS="all 0 1 2 3 4"
#######################################

# Set variables
PROJ_PATH=~/TotalSegMRI
JOB_PATH=~/scratch/TotalSegMRI
ENVDIR=${JOB_PATH}/venv

# Make dirs
mkdir -p $PROJ_PATH
mkdir -p $JOB_PATH

# Set nnunet params
export nnUNet_raw=${JOB_PATH}/nnUNet_raw
export nnUNet_preprocessed=${JOB_PATH}/nnUNet_preprocessed
export nnUNet_results=${JOB_PATH}/nnUNet_results

export nnUNet_def_n_proc=16
export nnUNet_n_proc_DA=$nnUNet_def_n_proc

CONFIGURATION="3d_fullres"

DATASET=Dataset${DATASET_ID}_TotalSegMRI

echo ""
echo "Running with the following parameters:"
echo "DATASET_ID=${DATASET_ID}"
echo "FOLD=${FOLD}"
echo "CONFIGURATION=${CONFIGURATION}"
echo ""

# Create and activate virtual env
if [ ! -d $ENVDIR ]; then
    module load python/3.10
    virtualenv --no-download $ENVDIR
    source $ENVDIR/bin/activate
    pip install --no-index torch torchvision torchaudio nnunetv2 nilearn
else
    source $ENVDIR/bin/activate
fi

if [ -f $nnUNet_results/${DATASET}/nnUNetTrainer__nnUNetPlans__${CONFIGURATION}/fold_${FOLD}/checkpoint_best.pth ]; then
    rm -rf ${PROJ_PATH}/test/${DATASET_ID}/labels
    nnUNetv2_predict -i ${PROJ_PATH}/test/${DATASET_ID}/images -o ${PROJ_PATH}/test/${DATASET_ID}/labels -chk checkpoint_best.pth -d ${DATASET_ID} -c ${CONFIGURATION} -f ${FOLD} -npp $nnUNet_def_n_proc -nps $nnUNet_def_n_proc
fi

# nnUNetv2_predict -d Dataset206_TotalSegMRI -i INPUT_FOLDER -o OUTPUT_FOLDER -f  0 1 2 3 4 -tr nnUNetTrainer -c 3d_fullres -p nnUNetPlans
# nnUNetv2_apply_postprocessing -i OUTPUT_FOLDER -o OUTPUT_FOLDER_PP -pp_pkl_file /home/yehudaw/scratch/TotalSegMRI/nnUNet_results/Dataset206_TotalSegMRI/nnUNetTrainer__nnUNetPlans__3d_fullres/crossval_results_folds_0_1_2_3_4/postprocessing.pkl -np 32 -plans_json /home/yehudaw/scratch/TotalSegMRI/nnUNet_results/Dataset206_TotalSegMRI/nnUNetTrainer__nnUNetPlans

rm -rf $PROJ_PATH/test/${DATASET_ID}/jpgs/*
python ~/generate_seg_jpg.py -i $PROJ_PATH/test/${DATASET_ID}/images -s $PROJ_PATH/test/${DATASET_ID}/labels -o $PROJ_PATH/test/${DATASET_ID}/jpgs -r 1
