# Use an active learning strategy to improve the ground truths

This README explains the steps that were performed to improve the quality of the ground truths segmentations used to train TotalSpineSeg. The used scripts were designed to work on [BIDS datasets](https://bids-specification.readthedocs.io/en/stable/).

## Original ground truths limitation
> For more information about the training, see https://www.researchgate.net/publication/389881289_TotalSpineSeg_Robust_Spine_Segmentation_with_Landmark-Based_Labeling_in_MRI

Given the time consuming task of manual segmentation of the different classes segmented by TotalSpineSeg: spinal cord, spinal canal, vertebrae and intervertebral discs, the [PAM50 atlas](https://pubmed.ncbi.nlm.nih.gov/29061527/) was registered using the intervertebral discs in the native space of all the scans showing the cervical and/or the thoracic region and was cosidered as the ground truth segmentation. But due to varying anatomies this atlas was not perfectly aligned with every subject space. The problem is that these slight anatomical inconsistencies do appear on the final trained model predictions and are often associated with over/under segmentations of the anatomical structures. This is the case for models trained prior to `r20250224` (included). 

Due to the time-consuming nature of manually segmenting the anatomical structures identified by TotalSpineSeg namely the spinal cord, spinal canal, vertebrae, and intervertebral discs, the PAM50 atlas was registered to the native space of each scan that included the cervical and/or thoracic regions to serve as ground truth segmentation. However, because of inter-subject anatomical variability, the atlas did not perfectly align with every subject anatomy. These subtle misalignments introduced inconsistencies that propagated into the model’s predictions, often resulting in over- or under-segmentation of certain structures. This issue affects models trained up to and including release r20250224.

## Strategy

### Automatic refinement

To improve the ground truths segmentations rapidly, [nnInteractive](https://github.com/MIC-DKFZ/nnInteractive) will be used. Datasets will be assumed to follow BIDS convention.

1. Install TotalSpineSeg in an environment
> See main README.md

2. Install [nnInteractive](https://github.com/MIC-DKFZ/nnInteractive) in another environment

3. Update the script `refine_bids.sh`:
    - update custom variables (`BIDS_FOLDER` and `TOTALSPINESEG` path)
    - update environment activation and deactivation with newly setup environment

4. Run `refine_bids.sh`

###  Manual segmentation

Manual segmentation may be needed afterwards.







