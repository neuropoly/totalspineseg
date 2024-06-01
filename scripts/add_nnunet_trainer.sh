#!/bin/bash

# Locate the path to the nnunetv2 library using Python
NNUNETV2_PATH=$(python -c "import nnunetv2; import os; print(os.path.dirname(nnunetv2.__file__))")

# Define the target file
TARGET_FILE="$NNUNETV2_PATH/training/nnUNetTrainer/variants/training_length/nnUNetTrainer_Xepochs.py"

# Check if the nnUNetTrainer_16000epochs class is present in the target file
if ! grep -q "class nnUNetTrainer_16000epochs" "$TARGET_FILE"; then
  # Add the class definition to the target file
  cat <<EOL >> "$TARGET_FILE"

class nnUNetTrainer_16000epochs(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 16000

EOL
fi
