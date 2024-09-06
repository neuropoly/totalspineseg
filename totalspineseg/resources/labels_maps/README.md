# Label Map Files

This folder contains JSON map files used for preparing datasets for specific model training. These files are crucial in the process of mapping labels from the original dataset to the labels used in each specific training model dataset.

## Purpose

The JSON map files are used by the [`utils/map_labels.py`](../../utils/map_labels.py) script to transform the original dataset labels into the specific label structure required for each training model dataset (Dataset101, Dataset102, Dataset103).

## Usage

The mapping process is part of the overall dataset preparation workflow. For detailed information on how these map files are used in the context of dataset preparation, please refer to the [`scripts/prepare_datasets.sh`](../../../scripts/prepare_datasets.sh) script.

## Additional Information

- For comprehensive information about the different datasets (Dataset101, Dataset102, Dataset103) and their specific label structures, please consult the main project [README](../../../README.md).

- To understand the exact mapping process and how these JSON files are utilized, review the [`utils/map_labels.py`](../../utils/map_labels.py) script.

- The [`scripts/prepare_datasets.sh`](../../../scripts/prepare_datasets.sh) script provides the overall context of how these map files fit into the dataset preparation workflow.