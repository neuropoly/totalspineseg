# nn-UNet Datasets

This folder contains JSON files for nn-UNet datasets used in training the model.

## Dataset Files

1. `dataset_step1.json`: Configuration for Dataset101_TotalSegMRI_step1 (First step of segmentation)
2. `dataset_step2.json`: Configuration for Dataset102_TotalSegMRI_step2 (Second step of segmentation)
3. `dataset_full.json`: Configuration for Dataset103_TotalSegMRI_full (Single-step full segmentation)

These files are copied to their respective folders in `data/nnUnet/nnUNet_raw/DatasetXXX/dataset.json`.

## References

- Main project [README](../../../../README.md) - For information about the datasets.

- Datasets preparation script: [`prepare_nnunet_datasets.sh`](../../../../scripts/prepare_nnunet_datasets.sh) - For information how these files are copied into the destination dataset folder.

- [nnU-Net GitHub repository](https://github.com/MIC-DKFZ/nnUNet) - For information about dataset.json structure and usage.