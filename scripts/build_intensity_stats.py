
import os, sys, argparse, textwrap, json, re
from pathlib import Path
import numpy as np
from SynthSeg.estimate_priors import build_intensity_stats


def main():

    # Description and arguments
    parser = argparse.ArgumentParser(
        description=textwrap.dedent(f'''
        This script uses SynthSeg.estimate_priors.build_intensity_stats to calculate signal statistics (mean + std) '''
        '''for each mask, for all subjects, and saves the rsults to `prior_means.npy` and `prior_stds.npy` files. '''
        '''This script will also generate `labels.npy` - list of unique labels from which signal statistics was '''
        '''calculated. If masks-class-ids was provided `classes.npy` will be generated too, with the class coresponding '''
        '''to the labels in `labels.npy`. The output signal statistics will be calculated for each mask id (from masks-ids '''
        '''argument), with duplicated ids dropped. The output signal statistics will be order by the mask id.
        The data is assumed to follow the BIDS structure (or similar):

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
        help='The folder where output files will be saved to'
    )
    parser.add_argument(
        '--masks-ids', '-m', type=argparse.FileType('r',encoding='utf-8'), required=True,
        help='json file contaning mapping of each mask to a unique number. for example: '
            '{"vertebrae_L3": 2, "organ_spleen": 15}'
    )
    parser.add_argument(
        '--masks-class-ids', '-c', type=argparse.FileType('r',encoding='utf-8'), default=None,
        help='json file contaning mapping of each mask to a class. enables to regroup structures into '
            'classes of similar intensity statistics. for example: {"vertebrae_L3": 2, "vertebrae_L4": 2}'
    )
    parser.add_argument(
        '--default-stats', type=int, default=(0, 0, 0, 0), metavar=('mean mean', 'mean std', 'std mean', 'std std'), nargs=4,
        help='Default mean (mean+std) and std (mean+std) to use if the calculated value was 0 '
        '(the label is empty). default values is 0.'
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
    masks_ids_file = args.masks_ids
    masks_class_ids_file = args.masks_class_ids
    default_stats = args.default_stats
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
            masks_ids = "{masks_ids_file.name if masks_ids_file else ''}"
            masks_class_ids = "{masks_class_ids_file.name if masks_class_ids_file else ''}"
            default_stats = "{default_stats}"
            subject_prefix = "{subject_prefix}"
            subject_subdir = "{subject_subdir}"
            seg_suffix = "{seg_suffix}"
            img_suffix = "{img_suffix}"
            verbose = {verbose}
        '''))

    # Number of non sided labels
    n_neutral_labels = None
    # Get labels from masks-ids json file
    estimation_labels =  None
    if masks_ids_file:
        
        masks_ids = json.load(masks_ids_file)
        masks_ids_file.close()
        
        # Make estimation_labels unique and sorted as follow: background, non-sided, left, right
        background_labels = [masks_ids[k] for k in ['background']]
        left_labels = [masks_ids[k] for k in masks_ids if re.match(r'^.*_left(_\d+)?$', k)]
        right_labels = [masks_ids[k] for k in masks_ids if re.match(r'^.*_right(_\d+)?$', k)]
        non_sided_labels = [v for v in masks_ids.values() if v not in background_labels + left_labels + right_labels]
        # Make estimation_labels
        estimation_labels = np.array(
            background_labels + non_sided_labels + left_labels + right_labels
        )
        # Ensure estimation_labels not containing duplicates
        estimation_labels = estimation_labels[np.sort(np.unique(estimation_labels, return_index=True)[1])]
        # Get number of non sided labels
        n_neutral_labels = len(background_labels + non_sided_labels)

    # Get class from masks-class-ids json file
    estimation_classes = None
    if masks_class_ids_file:

        masks_class_ids = json.load(masks_class_ids_file)
        masks_class_ids_file.close()
        
        # Ensure all masks are in masks_class_ids
        if masks_ids is None or estimation_labels is None or not all(m in masks_class_ids for m in masks_ids.keys()):
            print(f'Not all masks from {masks_ids_file.name} are in {masks_class_ids.name}')
            sys.exit()
        
        # Create mapping from mask id to mask class id
        masks_ids_class_ids_map = {mask_id: masks_class_ids[mask] for mask, mask_id in masks_ids.items()}

        # Create list of masks class ids matching to estimation_labels
        estimation_labels_classes = [masks_ids_class_ids_map[mask_id] for mask_id in estimation_labels.tolist()]

        # Generate a map to a new unique and sequential id for each class
        seq_estimation_labels_classes_map = {c: i for i, c in enumerate(set(estimation_labels_classes))}

        # Get list of unique and sequential class ids for each label in estimation_labels
        estimation_classes = np.array([seq_estimation_labels_classes_map[c] for c in estimation_labels_classes])

    # Init lists for images and segmantations directories.
    segs = []
    imgs = []

    # Loop over all subjects and look for images and sgmentations files
    for sub_path in imgs_path.glob(f'{subject_prefix}*'):

        # Get subject name
        subject_name = sub_path.name
        if verbose: print(f'Working on {subject_name}')

        # Look for segmentation file and ensure its existance
        seg_path = segs_path / subject_name / subject_subdir / f'{subject_name}{seg_suffix}.nii.gz'

        # If segmentation not fount try to find it directly in segs_path (not in subdir)
        if not seg_path.exists():
            seg_path = segs_path / f'{subject_name}{seg_suffix}.nii.gz'

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
        imgs.append(f'{img_path}')

    # calculate signal statistics
    build_intensity_stats(
        list_image_dir=imgs,
        list_labels_dir=segs,
        estimation_labels=estimation_labels,
        estimation_classes=estimation_classes,
        result_dir=f'{output_path}',
        rescale=True
    )

    # Update default stats from input argument
    if default_stats != (0, 0, 0, 0):
        for name, vals in zip(('mean', 'std'), np.split(np.array(default_stats), 2)):
            fname = f"{output_path / f'prior_{name}s.npy'}"
            # Load output file (means or stds)
            arr = np.load(fname)
            # Update from input argument where the value is 0
            for i, v in enumerate(vals):
                arr[i][arr[i]==0] = v
            # Save the file
            np.save(fname, arr)
    
    # Save labels and class to file
    if estimation_labels is not None:
        np.save(output_path / 'labels.npy', estimation_labels)
    if estimation_classes is not None:
        np.save(output_path / 'classes.npy', estimation_classes)
    if n_neutral_labels:
        np.save(output_path / 'n_neutral_labels.npy', n_neutral_labels)


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
