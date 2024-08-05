#!/bin/bash

# Locate the path to the nnunetv2 library using Python
NNUNETV2_PATH="$(python3 -c "import nnunetv2; print(nnunetv2.__path__[0])")"

# Define the target file
TARGET_FILE="$NNUNETV2_PATH/training/nnUNetTrainer/nnUNetTrainer_16000epochs.py"

# Ensure the target file exists and create it if it is not
if [ ! -f "$TARGET_FILE" ]; then
  # Add the class definition to the target file
  cat <<EOL > "$TARGET_FILE"
import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
class nnUNetTrainer_16000epochs(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 16000
EOL
fi
