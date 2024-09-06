# nn-UNet Datasets

This folder contains JSON files for nn-UNet datasets used in training the model.

## Dataset Files

1. `dataset_step1.json`: Configuration for Dataset101 (First step of segmentation)
2. `dataset_step2.json`: Configuration for Dataset102 (Second step of segmentation)
3. `dataset_full.json`: Configuration for Dataset103 (Single-step full segmentation)

These files are copied to their respective folders in `$TOTALSPINESEG_DATA/nnUnet/raw/DatasetXXX/dataset.json`.

## References

- Main project [README](../../../README.md) - For information about the datasets.

- Datasets preparation script: [`prepare_datasets.sh`](../../../scripts/prepare_datasets.sh) - For information how these files are copied into the destination dataset folder.

- [nnU-Net GitHub repository](https://github.com/MIC-DKFZ/nnUNet) - For information about dataset.json structure and usage.