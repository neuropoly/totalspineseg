#!/bin/bash

# This script get the datasets require to train the model from:
#   git@data.neuro.polymtl.ca:datasets/whole-spine.git
#   git@data.neuro.polymtl.ca:datasets/spider-challenge-2023.git
#   git@github.com:spine-generic/data-multi-subject.git

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

# Set TOTALSPINESEG and TOTALSPINESEG_DATA if not set
TOTALSPINESEG="$(realpath "${TOTALSPINESEG:-totalspineseg}")"
TOTALSPINESEG_DATA="$(realpath "${TOTALSPINESEG_DATA:-data}")"

# Fetch path to data list
data_json="$TOTALSPINESEG/totalspineseg/resources/data/retrain_tss.json"

# Set the paths to the BIDS data folders
bids="$TOTALSPINESEG_DATA"/bids

# Make sure $TOTALSPINESEG_DATA/bids exists and enter it
mkdir -p "$bids"
CURR_DIR="$(realpath .)"
cd "$bids"

datasets=(
    git@data.neuro.polymtl.ca:p118739/whole-spine-mirror.git # TODO: Update link to whole spine
    git@data.neuro.polymtl.ca:datasets/spider-challenge-2023.git
    git@github.com:spine-generic/data-multi-subject.git
)

branches=(
    nm/add_spine_and_canal
    nm/update_some_labels
    nm/add_spine_and_canal
)

# Clone datasets and checkout on the right branch
for i in "${!datasets[@]}"; do
    ds=${datasets[i]}
    branch=${branches[i]}
    dsn=$(basename $ds .git)

    # Clone the dataset from the specified repository
    git clone "$ds"

    # Enter the dataset directory
    cd "$dsn"

    # Checkout on the branch
    git checkout "$branch"

    # Move back to the parent directory to process the next dataset
    cd ..
done

# Rename whole-spine
# TODO: remove this step in the future
mv whole-spine-mirror whole-spine

keys=(
    IMAGE
    LABEL_SPINE
    LABEL_CORD
    LABEL_CANAL
)

# Download necessary data from git annex
for key in "${keys[@]}"; do
    for path in $(jq -r ".TRAINING | .[].$key" "$data_json"); do
        IFS='/' read -r rep_path rel_path <<< "$path"
        echo git -C "$rep_path" annex get "$rel_path"
    done
done

# Return to the original working directory
cd "$CURR_DIR"
