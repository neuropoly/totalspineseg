#!/bin/bash

# This script train the TotalSpineSeg nnUNet models.
# It get also optional parameters DATASET and FOLD.
# By default, it trains the models for datasets 101 and 102 with fold 0.

# The script excpects the following environment variables to be set:
#   TOTALSPINESEG: The path to the TotalSpineSeg repository.
#   TOTALSPINESEG_DATA: The path to the TotalSpineSeg data folder.
#   TOTALSPINESEG_JOBS: The number of CPU cores to use. Default is the number of CPU cores available.
#   TOTALSPINESEG_DEVICE: The device to use. Default is "cuda" if available, otherwise "cpu".
#   TOTALSPINESEG_JOBSNN: The number of jobs to use for the nnUNet. Default is the number of CPU cores available or the available memory in GB divided by 8, whichever is smaller.

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

# Set the datasets to work with - default is 101 102
DATASETS=${1:-101 102}
if [ "$DATASETS" == all ]; then DATASETS=(101 102 103); fi

# Set the fold to work with - default is 0
FOLD=${2:-0}

# set TOTALSPINESEG and TOTALSPINESEG_DATA if not set
TOTALSPINESEG="$(realpath "${TOTALSPINESEG:-totalspineseg}")"
TOTALSPINESEG_DATA="$(realpath "${TOTALSPINESEG_DATA:-data}")"

# ensure the custom nnUNetTrainer is defined in the nnUNet library and add it if it is not
source "$TOTALSPINESEG"/scripts/add_nnunet_trainer.sh

# Get the number of CPUs
CORES=${SLURM_JOB_CPUS_PER_NODE:-$(lscpu -p | egrep -v '^#' | wc -l)}

# Get memory in GB
RAMGB=$(awk '/MemTotal/ {print int($2/1024/1024)}' /proc/meminfo)

# Set the device to cpu if cuda is not available
DEVICE=${TOTALSPINESEG_DEVICE:-$(python3 -c "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')")}

# Set the number of jobs
JOBS=${TOTALSPINESEG_JOBS:-$CORES}

# Set the number of jobs for the nnUNet
JOBSNN=$(( JOBS < $((RAMGB / 8)) ? JOBS : $((RAMGB / 8)) ))
JOBSNN=$(( JOBSNN < 1 ? 1 : JOBSNN ))
JOBSNN=${TOTALSPINESEG_JOBSNN:-$JOBSNN}

export nnUNet_def_n_proc=$JOBSNN
export nnUNet_n_proc_DA=$JOBSNN

# Set nnunet params
export nnUNet_raw="$TOTALSPINESEG_DATA"/nnUNet/raw
export nnUNet_preprocessed="$TOTALSPINESEG_DATA"/nnUNet/preprocessed
export nnUNet_results="$TOTALSPINESEG_DATA"/nnUNet/results
export nnUNet_exports="$TOTALSPINESEG_DATA"/nnUNet/exports

nnUNetTrainer=nnUNetTrainer_16000epochs
nnUNetPlanner=ExperimentPlanner
nnUNetPlans=nnUNetPlans
configuration=3d_fullres
data_identifier=nnUNetPlans_3d_fullres

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
echo "data_identifier=${data_identifier}"
echo "JOBSNN=${JOBSNN}"
echo "DEVICE=${DEVICE}"
echo ""

echo "Working with datasets: ${DATASETS[@]}, fold: $FOLD"

for d in ${DATASETS[@]}; do

    # Get the dataset name
    d_name=$(basename "$(ls -d "$nnUNet_raw"/Dataset${d}_*)")

    if [ ! -f "$nnUNet_preprocessed"/$d_name/${nnUNetPlans}.json ]; then
        echo "Preprocess dataset $d_name"
        nnUNetv2_plan_and_preprocess -d $d -pl $nnUNetPlanner -c $configuration -npfp $JOBSNN -np $JOBSNN --verify_dataset_integrity
        # Change the patch size to [224, 160, 80] for dataset 103 and [128, 96, 96] for 101 and 102
        if [ $d -eq 103 ]; then
            jq ".configurations[\"${configuration}\"].patch_size = [224, 160, 80]" "$nnUNet_preprocessed"/$d_name/${nnUNetPlans}.json > "$nnUNet_preprocessed"/$d_name/${nnUNetPlans}_new.json
        else
            jq ".configurations[\"${configuration}\"].patch_size = [128, 96, 96]" "$nnUNet_preprocessed"/$d_name/${nnUNetPlans}.json > "$nnUNet_preprocessed"/$d_name/${nnUNetPlans}_new.json
        fi
        mv "$nnUNet_preprocessed"/$d_name/${nnUNetPlans}_new.json "$nnUNet_preprocessed"/$d_name/${nnUNetPlans}.json
    fi

    echo "Training nnUNet model for dataset $d_name"
    # if already decompressed do not decompress again
    if [ $(find "$nnUNet_preprocessed"/$d_name/$data_identifier -name "*.npy" | wc -l) -eq $(( 2 * $(find "$nnUNet_preprocessed"/$d_name/$data_identifier -name "*.npz" | wc -l))) ]; then DECOMPRESSED="--use_compressed"; else DECOMPRESSED=""; fi
    nnUNetv2_train $d $configuration $FOLD -tr $nnUNetTrainer -p $nnUNetPlans --c -device $DEVICE $DECOMPRESSED

    echo "Export the model for dataset $d_name in "$nnUNet_exports""
    mkdir -p "$nnUNet_exports"
    mkdir -p "$nnUNet_results"/$d_name/ensembles
    nnUNetv2_export_model_to_zip -d $d -o "$nnUNet_exports"/${d_name}__${nnUNetTrainer}__${nnUNetPlans}__${configuration}__fold_$FOLD.zip -c $configuration -f $FOLD -tr $nnUNetTrainer -p $nnUNetPlans

    echo "Testing nnUNet model for dataset $d_name"
    mkdir -p "$nnUNet_results"/$d_name/${nnUNetTrainer}__${nnUNetPlans}__${configuration}/fold_${FOLD}/test
    nnUNetv2_predict -d $d -i "$nnUNet_raw"/$d_name/imagesTs -o "$nnUNet_results"/$d_name/${nnUNetTrainer}__${nnUNetPlans}__${configuration}/fold_${FOLD}/test -f $FOLD -c $configuration -tr $nnUNetTrainer -p $nnUNetPlans -npp $JOBSNN -nps $JOBSNN
    nnUNetv2_evaluate_folder "$nnUNet_raw"/$d_name/labelsTs "$nnUNet_results"/$d_name/${nnUNetTrainer}__${nnUNetPlans}__${configuration}/fold_${FOLD}/test -djfile "$nnUNet_results"/$d_name/${nnUNetTrainer}__${nnUNetPlans}__${configuration}/dataset.json -pfile "$nnUNet_results"/$d_name/${nnUNetTrainer}__${nnUNetPlans}__${configuration}/plans.json -np $JOBSNN

    p="$(realpath .)"
    cd "$nnUNet_results"
    zip "$nnUNet_exports"/${d_name}__${nnUNetTrainer}__${nnUNetPlans}__${configuration}__fold_$FOLD.zip $d_name/${nnUNetTrainer}__${nnUNetPlans}__${configuration}/fold_${FOLD}/test/summary.json
    cd "$p"

    echo "Export nnUNet dataset list for dataset $d_name"
    cd "$nnUNet_raw"/$d_name
    ls */ > "$nnUNet_results"/$d_name/dataset.txt
    cd "$nnUNet_results"
    zip "$nnUNet_exports"/${d_name}__${nnUNetTrainer}__${nnUNetPlans}__${configuration}__fold_$FOLD.zip $d_name/dataset.txt
    cd "$nnUNet_preprocessed"
    zip "$nnUNet_exports"/${d_name}__${nnUNetTrainer}__${nnUNetPlans}__${configuration}__fold_$FOLD.zip $d_name/splits_final.json
    cd "$p"

done
