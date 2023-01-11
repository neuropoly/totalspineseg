# totalsegmentator-mri
Code for the TotalSegmentator MRI project.

## Data organization

As a starting point, a few MPRAGE data are under our private [google folder](https://drive.google.com/drive/folders/1CAkz4ZuxQjWza7GAXhXxTkKcyB9p3yME).

We will follow the BIDS structure:
```
├── derivatives
│   └── manual_masks
│       └── sub-errsm37
│           └── anat
└── sub-errsm37
    └── anat
        ├── sub-errsm37_T1w.json
        └── sub-errsm37_T1w.nii.gz
```
