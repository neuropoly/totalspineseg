"""
Based on https://github.com/neuropoly/totalsegmentator-mri 
"""

import os, argparse, textwrap
from scipy.ndimage import label
from pathlib import Path
import numpy as np
import nibabel as nib


def get_parser():
    # parse command line arguments
    parser = argparse.ArgumentParser(
        description=textwrap.dedent(f'''
            This script processes a NIfTI segmentation file, leaving the largest component for each label.
        '''),
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--seg-in', type=str, required=True,
        help='Input segmentation path'
    )
    parser.add_argument(
        '--seg-out', type=str, required=True, 
        help='Output segmentation path'
    )
    parser.add_argument(
        '--verbose', '-v', type=int, default=1, choices=[0, 1],
        help='Verbosity level. 0: Errors/warnings only, 1: Errors/warnings + info (default: 1)'
    )
    return parser


def main():
    """
    """
    parser = get_parser()
    args = parser.parse_args()

    # Get arguments
    seg_in = args.seg_in
    seg_out = args.seg_out
    verbose = args.verbose

    if verbose:
        print(textwrap.dedent(f'''
            Running {Path(__file__).stem} with the following params:
            seg_in = "{seg_in}"
            seg_out = "{seg_out}"
            verbose = {verbose}
        '''))
    
    # Keep largest component
    keep_largest_component(seg_in, seg_out)
    

def keep_largest_component(
            seg_in,
            seg_out
        ):
    
    # Load segmentation
    seg = nib.load(seg_in)
    seg_data = seg.get_fdata()

    # Convert data to uint8 to avoid issues with segmentation IDs
    seg_data_src = seg_data.astype(np.uint8)

    seg_data = np.zeros_like(seg_data_src)

    for l in [_ for _ in np.unique(seg_data_src) if _ != 0]:
        mask = seg_data_src == l
        mask_labeled, num_labels = label(mask, np.ones((3, 3, 3)))
        # Find the label of the largest component
        label_sizes = np.bincount(mask_labeled.ravel())[1:]  # Skip 0 label size
        largest_label = label_sizes.argmax() + 1  # +1 because bincount labels start at 0
        seg_data[mask_labeled == largest_label] = l

    # Create result segmentation
    seg = nib.Nifti1Image(seg_data, seg.affine, seg.header)
    seg.set_data_dtype(np.uint8)

    # Make sure output directory exists
    if not os.path.exists(os.path.dirname(seg_out)):
        os.makedirs(os.path.dirname(seg_out))

    # Save mapped segmentation
    nib.save(seg, seg_out)


if __name__ == '__main__':
    main()