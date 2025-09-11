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
data_json="$TOTALSPINESEG/totalspineseg/resources/data/training_data.json"

# Set the paths to the BIDS data folders
bids="$TOTALSPINESEG_DATA"/bids

# Make sure $TOTALSPINESEG_DATA/bids exists and enter it
mkdir -p "$bids"
CURR_DIR="$(realpath .)"
cd "$bids"

datasets=(
    git@data.neuro.polymtl.ca:datasets/whole-spine.git
    git@data.neuro.polymtl.ca:datasets/spider-challenge-2023.git
    git@github.com:spine-generic/data-multi-subject.git
)

commits=(
    45bc28f3cf522feec5160e2f145acd03759ede39
    d9be04cfb27da100fe03d968e220cccebbbc9a3f
    646e4bf21c4182a2f8e9f5ee4892d4cd7cbe0dc3
)

# Clone datasets and checkout on the right branch
for i in "${!datasets[@]}"; do
    ds=${datasets[i]}
    commit=${commits[i]}
    dsn=$(basename $ds .git)

    # Clone the dataset from the specified repository
    git clone "$ds"

    # Enter the dataset directory
    cd "$dsn"

    # Checkout on the commit
    git checkout "$commit"

    # Move back to the parent directory to process the next dataset
    cd ..
done

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
        git -C "$rep_path" annex get "$rel_path"
    done
done

for key in "${keys[@]}"; do
    for path in $(jq -r ".TESTING | .[].$key" "$data_json"); do
        IFS='/' read -r rep_path rel_path <<< "$path"
        git -C "$rep_path" annex get "$rel_path"
    done
done

# Return to the original working directory
cd "$CURR_DIR"
