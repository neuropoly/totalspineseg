
import os, sys, argparse, textwrap
from pathlib import Path
import numpy as np
from SynthSeg.estimate_priors import build_intensity_stats


def main():

    # Description and arguments
    parser = argparse.ArgumentParser(
        description=textwrap.dedent(f'''
        This script uses SynthSeg.estimate_priors.build_intensity_stats to calculate signal statistics (mean + std) for each mask, for all subjects, and saves the rsults to `prior_means.npy` and `prior_stds.npy` files. The data is assumed to follow the BIDS structure (or similar):

        data
        ├── sub-errsm37
        │   └── anat
        │       ├── sub-errsm37_T1w.json
        │       └── sub-errsm37_T1w.nii.gz
        └── sub-errsm38
            └── anat
                ├── sub-errsm38_T1w.json
                └── sub-errsm38_T1w.nii.gz
        
        segmentations
        ├── sub-errsm37
        │   └── anat
        │       └── sub-errsm37_T1w_seg.nii.gz
        └── sub-errsm38
            └── anat
                └── sub-errsm38_T1w_seg.nii.gz

        '''),
        epilog=textwrap.dedent('''
        Example:
        build_intensity_stats -d data -s segmentations -o priors
        '''),
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        '--imgs-dir', '-d', type= DirPath(), required=True,
        help='The folder where input images are located for each subject.'
    )
    parser.add_argument(
        '--segs-dir', '-s', type= DirPath(), required=True,
        help='The folder where segmentations are located for each subject.'
    )
    parser.add_argument(
        '--output-dir', '-o', type= DirPath(True), required=True,
        help='The folder where output prior_means.npy and prior_stds.npy files will be saved to'
    )
    parser.add_argument(
        '--subject-prefix', type= str, default='sub-',
        help='Subject prefix, defaults to "sub-" which is the prefix used for BIDS directories.'
    )
    parser.add_argument(
        '--subject-subdir', type= str, default='anat',
        help='The subfolder inside subject folder contaning the masks, defaults to "anat" which is used for BIDS directories.'
    )
    parser.add_argument(
        '--img-suffix', type= str, default='_T1w',
        help='The suffix for the input images, defaults to "_T1w_seg".'
    )
    parser.add_argument(
        '--seg-suffix', type= str, default='_T1w_seg',
        help='The suffix for the input segmentation, defaults to "_T1w_seg".'
    )
    parser.add_argument(
        '--verbose', '-v', type= int, default=1, choices=[0, 1],
        help='verbose. 0: Display only errors/warnings, 1: Errors/warnings + info messages (default: 1)'
    )

    try:
        args = parser.parse_args()
    except BaseException as e:
        sys.exit()

    # Get args
    imgs_path = args.imgs_dir
    segs_path = args.segs_dir
    output_path = args.output_dir
    subject_prefix = args.subject_prefix
    subject_subdir = args.subject_subdir
    img_suffix = args.img_suffix
    seg_suffix = args.seg_suffix
    verbose = args.verbose

    if verbose:
        print(textwrap.dedent(f''' 
            Running with the following arguments:
            imgs_dir = "{imgs_path}"
            segs_dir = "{segs_path}"
            output_dir = "{output_path}"
            subject_prefix = "{subject_prefix}"
            subject_subdir = "{subject_subdir}"
            seg_suffix = "{seg_suffix}"
            img_suffix = "{img_suffix}"
            verbose = {verbose}
        '''))

    # Init lists for images and segmantations directories.
    segs = []
    imgs = []

    # Loop over all subjects and look for images and sgmentations files
    for sub_path in segs_path.glob(f'{subject_prefix}*'):

        # Get subject name
        subject_name = sub_path.name
        if verbose: print(f'Working on {subject_name}')

        # Look for segmentation file and ensure its existance
        seg_path = segs_path / subject_name / subject_subdir / f'{subject_name}{seg_suffix}.nii.gz'
        if not seg_path.exists():
            print(f'No segmentation found for {subject_name}')
            continue
        
        # Look for image file and ensure its existance
        img_path = imgs_path / subject_name / subject_subdir / f'{subject_name}{img_suffix}.nii.gz'
        if not img_path.exists():
            print(f'No image found for {subject_name}')
            continue
        
        # Add image and segentation to the list
        segs.append(f'{seg_path}')
        imgs.append(f"{img_path}")

    # calculate signal statistics
    build_intensity_stats(
        list_image_dir=imgs,
        list_labels_dir=segs,
        estimation_labels=np.array(list(range(1, 105))),
        result_dir=f'{output_path}',
        rescale=True
    )


class DirPath(object):
    """
    Get path parameter from argparse and return it as pathlib Path object.

    Args:
    create (bool): Indicate if the directorie should be created. Default: False.
    """

    def __init__(self, create:bool=False):
        self.create = create

    def __call__(self, dir):

        path = Path(dir)

        # Create dir if create was specify
        if self.create and not path.exists():
            try:
                path.mkdir(parents=True, exist_ok=True)
            except: pass

        # Test if path exists
        if path.is_dir():
            return path
        else:
            raise argparse.ArgumentTypeError(f'readable_dir:{path} is not a valid path')


if __name__ == '__main__':
    main()
