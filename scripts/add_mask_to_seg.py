import os
import argparse
import numpy as np
from image import Image


def get_parser():
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Add a binary masked region to an existing segmentation file. The pixel values are set using the flag "-val".')
    parser.add_argument('-i', type=str, help='Input binary mask. Example: sub-001_T2w_label-sacrum_seg.nii.gz', required=True)
    parser.add_argument('-seg', type=str, help='Existing segmentation where the mask will be added. Example: sub-001_T2w_label-totalseg_dseg.nii.gz', required=True)
    parser.add_argument('-o', type=str, help='Output path where the final segmentation will be stored. Example: sub-001_T2w_label-totalseg_dseg.nii.gz', required=True)
    parser.add_argument('-val', type=int, help='Pixel values (integer) which will be used for the masked region.', required=True)
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    fname_mask = args.i
    fname_seg = args.seg
    fname_file_out = args.o
    val = args.val

    # Check if output directory exists
    if not os.path.exists(os.path.dirname(fname_file_out)):
        os.makedirs(os.path.dirname(fname_file_out))

    # Check if paths exist
    if not os.path.exists(fname_mask):
        raise ValueError(f'Missing input mask {fname_mask}')
    
    if not os.path.exists(fname_seg):
        raise ValueError(f'Missing output segmentation {fname_seg}')
    
    # Open images
    mask = Image(fname_mask)
    seg = Image(fname_seg)

    # Check image size
    if mask.data.shape != seg.data.shape:
        raise ValueError(f'Input mask and output segmentation should have the same shape !')

    # Check orientation
    if mask.orientation != seg.orientation:
        print(f'Changing mask orientation to {seg.orientation}')
        mask.change_orientation(seg.orientation)
    
    # Check if mask is binary
    if not np.sort(np.unique(mask.data)).tolist() == [0,1]:
        raise ValueError('Mask should be binary')
    
    # Check if value not already in seg
    if val in np.unique(seg.data).astype(int):
        raise ValueError(f'The value {val} is already present in the segmentation')
    
    # Add mask with value to segmentation
    zeros_seg = (seg.data == 0).astype(int) # Only zeros will be replaced, not already existing values
    added_mask = zeros_seg*mask.data
    seg.data[added_mask.astype(bool)] = val # Add new value

    # Save new seg
    seg.save(fname_file_out)


if __name__ == '__main__':
    main()