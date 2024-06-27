#!/bin/bash

# This script train the TotalSpineSeg nnUNet models.
# It get also optional parameters DATASET and FOLD.
# By default, it trains the models for datasets 101, 102, and 103 with fold 0.

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
export nnUNet_raw="$TOTALSPINESEG_DATA"/nnUNet/raw
export nnUNet_preprocessed="$TOTALSPINESEG_DATA"/nnUNet/preprocessed
export nnUNet_results="$TOTALSPINESEG_DATA"/nnUNet/results
export nnUNet_exports="$TOTALSPINESEG_DATA"/nnUNet/exports

nnUNetTrainer=nnUNetTrainer_16000epochs
nnUNetPlanner=ExperimentPlanner
nnUNetPlans=nnUNetPlans
configuration=3d_fullres

echo ""
echo "Running with the following parameters:"
echo "nnUNet_raw=${nnUNet_raw}"
echo "nnUNet_preprocessed=${nnUNet_preprocessed}"
echo "nnUNet_results=${nnUNet_results}"
echo "nnUNet_exports=${nnUNet_exports}"
echo "nnUNetTrainer=${nnUNetTrainer}"
echo "nnUNetPlanner=${nnUNetPlanner}"
echo "nnUNetPlans=${nnUNetPlans}"
echo "configuration=${configuration}"
echo "JOBS=${JOBS}"
echo ""

# ensure the custom nnUNetTrainer is defined in the nnUNet library and add it if it is not
source "$TOTALSPINESEG"/scripts/add_nnunet_trainer.sh

# Set the datasets to work with - default is 101 102 103
DATASETS=${1:-101 102 103}

# Set the fold to work with - default is 0
FOLD=${2:-0}

echo "Working with datasets: ${DATASETS[@]}, fold: $FOLD"

for d in ${DATASETS[@]}; do

    # Get the dataset name
    d_name=$(basename $(ls -d "$nnUNet_raw"/Dataset${d}_*))

    if [ ! -f "$nnUNet_preprocessed"/$d_name/${nnUNetPlans}.json ]; then
        echo "Preprocess dataset $d_name"
        nnUNetv2_plan_and_preprocess -d $d -pl $nnUNetPlanner -c $configuration -npfp $JOBS -np $JOBS --verify_dataset_integrity
        jq '.configurations["3d_fullres"].patch_size = [224, 160, 80]' "$nnUNet_preprocessed"/$d_name/${nnUNetPlans}.json > "$nnUNet_preprocessed"/$d_name/${nnUNetPlans}_new.json
        mv "$nnUNet_preprocessed"/$d_name/${nnUNetPlans}_new.json "$nnUNet_preprocessed"/$d_name/${nnUNetPlans}.json
    fi

    echo "Training nnUNet model for dataset $d_name"
    nnUNetv2_train $d $configuration $FOLD -tr $nnUNetTrainer -p $nnUNetPlans --c

    echo "Export the model for dataset $d_name in "$nnUNet_exports""
    mkdir -p "$nnUNet_exports"
    mkdir -p "$nnUNet_results"/$d_name/ensembles
    nnUNetv2_export_model_to_zip -d $d -o "$nnUNet_exports"/${d_name}_fold_$FOLD.zip -c $configuration -f $FOLD -tr $nnUNetTrainer -p $nnUNetPlans

    echo "Testing nnUNet model for dataset $d_name"
    mkdir -p "$nnUNet_results"/$d_name/${nnUNetTrainer}__${nnUNetPlans}__${configuration}/fold_${FOLD}/test
    nnUNetv2_predict -d $d -i "$nnUNet_raw"/$d_name/imagesTs -o "$nnUNet_results"/$d_name/${nnUNetTrainer}__${nnUNetPlans}__${configuration}/fold_${FOLD}/test -f $FOLD -c $configuration -tr $nnUNetTrainer -p $nnUNetPlans -npp $JOBS -nps $JOBS
    nnUNetv2_evaluate_folder "$nnUNet_raw"/$d_name/labelsTs "$nnUNet_results"/$d_name/${nnUNetTrainer}__${nnUNetPlans}__${configuration}/fold_${FOLD}/test -djfile "$nnUNet_results"/$d_name/${nnUNetTrainer}__${nnUNetPlans}__${configuration}/dataset.json -pfile "$nnUNet_results"/$d_name/${nnUNetTrainer}__${nnUNetPlans}__${configuration}/plans.json -np $JOBS

    p=$(realpath .)
    cd "$nnUNet_results"
    zip $p/"$nnUNet_exports"/${d_name}_fold_$FOLD.zip $d_name/${nnUNetTrainer}__${nnUNetPlans}__${configuration}/fold_${FOLD}/test/summary.json
    cd $p

    echo "Export nnUNet dataset list for dataset $d_name"
    cd "$nnUNet_raw"/$d_name
    ls */ > $p/"$nnUNet_results"/$d_name/${nnUNetTrainer}__${nnUNetPlans}__${configuration}/fold_${FOLD}/dataset.txt
    cd $p/"$nnUNet_results"
    zip $p/"$nnUNet_exports"/${d_name}_fold_$FOLD.zip $d_name/${nnUNetTrainer}__${nnUNetPlans}__${configuration}/fold_${FOLD}/dataset.txt
    cd $p

done
