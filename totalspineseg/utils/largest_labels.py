import sys, argparse, textwrap
from scipy.ndimage import label, binary_dilation, generate_binary_structure, iterate_structure
from pathlib import Path
import numpy as np
import nibabel as nib
import multiprocessing as mp
from functools import partial
from tqdm.contrib.concurrent import process_map
import warnings

warnings.filterwarnings("ignore")

def main():
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description=textwrap.dedent(f'''
            This script processes NIfTI segmentation files, leaving the largest component for each label.
        '''),
        epilog=textwrap.dedent('''
            Examples:
            largest_labels -s labels -o labels_largest
            For BIDS:
            largest_labels -s derivatives/labels -o derivatives/labels --seg-suffix "_seg" --output-seg-suffix "_seg_largest" -d "sub-" -u "anat"
        '''),
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        '--segs-dir', '-s', type=Path, required=True,
        help='Folder containing input segmentations.'
    )
    parser.add_argument(
        '--output-dir', '-o', type=Path, required=True, 
        help='Folder to save output segmentations.'
    )
    parser.add_argument(
        '--subject-dir', '-d', type=str, default=None, nargs='?', const='',
        help=textwrap.dedent('''
            Is every subject has its oen direcrory.
            If this argument will be provided without value it will look for any directory in the segmentation directory.
            If value also provided it will be used as a prefix to subject directory (for example "sub-"), defaults to False (no subjet directory).
        '''),
    )
    parser.add_argument(
        '--subject-subdir', '-u', type=str, default='', 
        help='Subfolder inside subject folder containing masks (for example "anat"), defaults to no subfolder.'
    )
    parser.add_argument(
        '--prefix', '-p', type=str, default='', 
        help='File prefix to work on.'
    )
    parser.add_argument(
        '--seg-suffix', type=str, default='',
        help='Segmentation suffix, defaults to "".'
    )
    parser.add_argument(
        '--output-seg-suffix', type=str, default='',
        help='Suffix for output segmentation, defaults to "".'
    )
    parser.add_argument(
        '--binarize', action="store_true", default=False,
        help='If provided, binarize the segmentation to non-zero values before taking the largest component.'
    )
    parser.add_argument(
        '--dilate', type=int, default=0,
        help='Number of voxels to dilate the segmentation before taking the largest component, defaults to 0 (no dilation).'
    )
    parser.add_argument(
        '--override', '-r', action="store_true", default=False,
        help='Override existing output files, defaults to false (Do not override).'
    )
    parser.add_argument(
        '--max-workers', '-w', type=int, default=mp.cpu_count(),
        help='Max worker to run in parallel proccess, defaults to multiprocessing.cpu_count().'
    )
    parser.add_argument(
        '--verbose', '-v', type=int, default=1, choices=[0, 1],
        help='Verbosity level. 0: Errors/warnings only, 1: Errors/warnings + info (default: 1)'
    )

    try:
        args = parser.parse_args()
    except BaseException as e:
        sys.exit()

    # Get arguments
    segs_path = args.segs_dir
    output_path = args.output_dir
    subject_dir = args.subject_dir
    subject_subdir = args.subject_subdir
    prefix = args.prefix
    seg_suffix = args.seg_suffix
    output_seg_suffix = args.output_seg_suffix
    binarize = args.binarize
    dilate = args.dilate
    override = args.override
    max_workers = args.max_workers
    verbose = args.verbose

    if verbose:
        print(textwrap.dedent(f'''
            Running {Path(__file__).stem} with the following params:
            segs_dir = "{segs_path}"
            output_dir = "{output_path}"
            subject_dir = "{subject_dir}"
            subject_subdir = "{subject_subdir}"
            prefix = "{prefix}"
            seg_suffix = "{seg_suffix}"
            output_seg_suffix = "{output_seg_suffix}"
            binarize = {binarize}
            dilate = {dilate}
            override = {override}
            max_workers = {max_workers}
            verbose = {verbose}
        '''))
    
    glob_pattern = ""
    if subject_dir is not None:
        glob_pattern += f"{subject_dir}*/"
    if len(subject_subdir) > 0:
        glob_pattern += f"{subject_subdir}/"
    glob_pattern += f'{prefix}*{seg_suffix}.nii.gz'

    # Process the NIfTI image and segmentation files
    segs_path_list = list(segs_path.glob(glob_pattern))

    # Create a partially-applied function with the extra arguments
    partial_largest_labels = partial(
        largest_labels,
        segs_path=segs_path,
        output_path=output_path,
        seg_suffix=seg_suffix,
        output_seg_suffix=output_seg_suffix,
        binarize=binarize,
        dilate=dilate,
        override=override,
   )

    with mp.Pool() as pool:
        process_map(partial_largest_labels, segs_path_list, max_workers=max_workers)
    

def largest_labels(
        seg_path,
        segs_path,
        output_path,
        seg_suffix,
        output_seg_suffix,
        binarize,
        dilate,
        override,
    ):
    
    output_seg_path = output_path / seg_path.relative_to(segs_path).parent / seg_path.name.replace(f'{seg_suffix}.nii.gz', f'{output_seg_suffix}.nii.gz')

    # If the output image already exists and we are not overriding it, return
    if not override and output_seg_path.exists():
        return

    # Load segmentation
    seg = nib.load(seg_path)
    seg_data = np.asanyarray(seg.dataobj).round().astype(np.uint8)

    if binarize:
        seg_data_src = seg_data.copy()
        seg_data = (seg_data != 0).astype(np.uint8)

    binary_dilation_structure = iterate_structure(generate_binary_structure(3, 1), dilate)
    output_seg_data = np.zeros_like(seg_data)

    for l in [_ for _ in np.unique(seg_data) if _ != 0]:
        mask = seg_data == l
        if dilate > 0:
            # Dilate
            mask_labeled, num_labels = label(binary_dilation(mask, binary_dilation_structure), np.ones((3, 3, 3)))
            # Undo dilate
            mask_labeled *= mask
        else:
            mask_labeled, num_labels = label(mask, np.ones((3, 3, 3)))
        # Find the label of the largest component
        label_sizes = np.bincount(mask_labeled.ravel())[1:]  # Skip 0 label size
        largest_label = label_sizes.argmax() + 1  # +1 because bincount labels start at 0
        output_seg_data[mask_labeled == largest_label] = l

    if binarize:
        output_seg_data = output_seg_data * seg_data_src

    # Create result segmentation
    output_seg = nib.Nifti1Image(output_seg_data, seg.affine, seg.header)
    output_seg.set_qform(seg.affine)
    output_seg.set_sform(seg.affine)
    output_seg.set_data_dtype(np.uint8)
    # Make sure output directory exists
    output_seg_path.parent.mkdir(parents=True, exist_ok=True)
    # Save mapped segmentation
    nib.save(output_seg, output_seg_path)

if __name__ == '__main__':
    main()