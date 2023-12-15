#!/bin/bash
#SBATCH --account=def-jcohen
#SBATCH --time=05-00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --gpus-per-node=a100:1
#SBATCH --mail-user=Yehuda.Warszawer@sheba.health.gov.il
#SBATCH --mail-type=ALL

# Get resources
# salloc --account=def-jcohen --time=0:20:00 --cpus-per-task=32 --mem=128G --gpus-per-node=a100:1

# Get params
########################################
DATASET_ID=$1
# DATASET_ID=100
FOLDS=$2
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

export nnUNet_def_n_proc=32
export nnUNet_n_proc_DA=$nnUNet_def_n_proc

CONFIGURATIONS="3d_fullres"
# CONFIGURATIONS="2d 3d_lowres 3d_fullres 3d_cascade_fullres"
DATASET=Dataset${DATASET_ID}_TotalSegMRI
mkdir -p ${PROJ_PATH}/nnUNet_results/${DATASET}

echo ""
echo "Running with the following parameters:"
echo "DATASET_ID=${DATASET_ID}"
echo "FOLDS=${FOLDS}"
echo "CONFIGURATIONS=${CONFIGURATIONS}"
echo ""

exclude="*/fold_*/validation"

# Create and activate virtual env
if [ ! -d $ENVDIR ]; then
    module load python/3.10
    virtualenv --no-download $ENVDIR
    source $ENVDIR/bin/activate
    pip install --no-index torch torchvision torchaudio nnunetv2 nilearn
else
    source $ENVDIR/bin/activate
fi

# Get data
if [ ! -d ${nnUNet_raw}/${DATASET} ]; then unzip ${DATASET}.zip -d ${nnUNet_raw}; fi

# Preprocessing
if [ ! -d ${nnUNet_preprocessed}/${DATASET} ]; then nnUNetv2_plan_and_preprocess -d ${DATASET_ID} -c ${CONFIGURATIONS} -npfp $nnUNet_def_n_proc -np $nnUNet_def_n_proc --verify_dataset_integrity; fi

# Training the model
for c in $CONFIGURATIONS; do
    for f in $FOLDS; do
        if [ -f ${nnUNet_results}/${DATASET}/*__$c/fold_$f/checkpoint_final.pth ]; then
            echo "Already trained configuration:$c, fold:$f"
            continue
        else
            PARAMS=""
            if [ -f ${nnUNet_results}/${DATASET}/*__$c/fold_$f/checkpoint_best.pth ]; then
                PARAMS="$PARAMS --c"
            fi
            if [ -f ${nnUNet_raw}/${DATASET}/checkpoint_init.pth ]; then
                PARAMS="$PARAMS -pretrained_weights ${nnUNet_raw}/${DATASET}/checkpoint_init.pth"
            fi
            echo "Training configuration:$c, fold:$f"
            nnUNetv2_train ${DATASET_ID} $c $f --npz $PARAMS
        fi
    done
    # Copy results to PROJ folder
    rsync -rtua --info=progress2 --exclude="${exclude}" ${nnUNet_results}/${DATASET}/ ${PROJ_PATH}/nnUNet_results/${DATASET}/
done

# Automatically determine the best configuration
if [[ ! -z "${f// /}" ]]; then
  nnUNetv2_find_best_configuration ${DATASET_ID} -np $nnUNet_def_n_proc -c ${CONFIGURATIONS} -f $FOLDS
fi

deactivate
