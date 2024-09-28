import sys, argparse, textwrap
from pathlib import Path
import numpy as np
import nibabel as nib
import multiprocessing as mp
from functools import partial
from tqdm.contrib.concurrent import process_map
import scipy.ndimage as ndi
import warnings

warnings.filterwarnings("ignore")

def main():

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description=' '.join(f'''
            This script processes NIfTI segmentation files, leaving the largest component for each label.
        '''.split()),
        epilog=textwrap.dedent('''
            Examples:
            largest_component -s labels -o labels_largest
            For BIDS:
            largest_component -s derivatives/labels -o derivatives/labels --seg-suffix "_seg" --output-seg-suffix "_seg_largest" -p "sub-*/anat/"
        '''),
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        '--segs-dir', '-s', type=Path, required=True,
        help='Folder containing input segmentations.'
    )
    parser.add_argument(
        '--output-segs-dir', '-o', type=Path, required=True,
        help='Folder to save output segmentations.'
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
        '--overwrite', '-r', action="store_true", default=False,
        help='Overwrite existing output files, defaults to false (Do not overwrite).'
    )
    parser.add_argument(
        '--max-workers', '-w', type=int, default=mp.cpu_count(),
        help='Max worker to run in parallel proccess, defaults to multiprocessing.cpu_count().'
    )
    parser.add_argument(
        '--quiet', '-q', action="store_true", default=False,
        help='Do not display inputs and progress bar, defaults to false (display).'
    )

    # Parse the command-line arguments
    args = parser.parse_args()

    # Get arguments
    segs_path = args.segs_dir
    output_segs_path = args.output_segs_dir
    prefix = args.prefix
    seg_suffix = args.seg_suffix
    output_seg_suffix = args.output_seg_suffix
    binarize = args.binarize
    dilate = args.dilate
    overwrite = args.overwrite
    max_workers = args.max_workers
    quiet = args.quiet

    # Print the argument values if not quiet
    if not quiet:
        print(textwrap.dedent(f'''
            Running {Path(__file__).stem} with the following params:
            segs_dir = "{segs_path}"
            output_segs_dir = "{output_segs_path}"
            prefix = "{prefix}"
            seg_suffix = "{seg_suffix}"
            output_seg_suffix = "{output_seg_suffix}"
            binarize = {binarize}
            dilate = {dilate}
            overwrite = {overwrite}
            max_workers = {max_workers}
            quiet = {quiet}
        '''))

    largest_component_mp(
        segs_path=segs_path,
        output_segs_path=output_segs_path,
        prefix=prefix,
        seg_suffix=seg_suffix,
        output_seg_suffix=output_seg_suffix,
        binarize=binarize,
        dilate=dilate,
        overwrite=overwrite,
        max_workers=max_workers,
        quiet=quiet,
    )

def largest_component_mp(
        segs_path,
        output_segs_path,
        prefix='',
        seg_suffix='',
        output_seg_suffix='',
        binarize=False,
        dilate=0,
        overwrite=False,
        max_workers=mp.cpu_count(),
        quiet=False,
    ):
    '''
    Wrapper function to handle multiprocessing.
    '''
    segs_path = Path(segs_path)
    output_segs_path = Path(output_segs_path)

    glob_pattern = f'{prefix}*{seg_suffix}.nii.gz'

    # Process the NIfTI image and segmentation files
    seg_path_list = list(segs_path.glob(glob_pattern))
    output_seg_path_list = [output_segs_path / _.relative_to(segs_path).parent / _.name.replace(f'{seg_suffix}.nii.gz', f'{output_seg_suffix}.nii.gz') for _ in seg_path_list]

    process_map(
        partial(
            _largest_component,
            binarize=binarize,
            dilate=dilate,
            overwrite=overwrite,
        ),
        seg_path_list,
        output_seg_path_list,
        max_workers=max_workers,
        chunksize=1,
        disable=quiet,
    )

def _largest_component(
        seg_path,
        output_seg_path,
        binarize=False,
        dilate=0,
        overwrite=False,
    ):
    '''
    Wrapper function to handle IO.
    '''
    seg_path = Path(seg_path)
    output_seg_path = Path(output_seg_path)

    # If the output image already exists and we are not overriding it, return
    if not overwrite and output_seg_path.exists():
        return

    # Load segmentation
    seg = nib.load(seg_path)

    output_seg = largest_component(
        seg,
        binarize=binarize,
        dilate=dilate,
    )

    # Ensure correct segmentation dtype, affine and header
    output_seg = nib.Nifti1Image(
        np.asanyarray(output_seg.dataobj).round().astype(np.uint8),
        output_seg.affine, output_seg.header
    )
    output_seg.set_data_dtype(np.uint8)
    output_seg.set_qform(output_seg.affine)
    output_seg.set_sform(output_seg.affine)

    # Make sure output directory exists and save the segmentation
    output_seg_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(output_seg, output_seg_path)

def largest_component(
        seg,
        binarize = False,
        dilate = 0,
    ):
    '''
    Leave the largest component for each label in the segmentation.

    Parameters
    ----------
    seg : nibabel.Nifti1Image
        Segmentation image.
    binarize : bool, optional
        If provided, binarize the segmentation to non-zero values before taking the largest component, by default False.
    dilate : int, optional
        Number of voxels to dilate the segmentation before taking the largest component, by default 0.

    Returns
    -------
    nibabel.Nifti1Image
        Output segmentation image.
    '''
    seg_data = np.asanyarray(seg.dataobj).round().astype(np.uint8)

    if binarize:
        seg_data_src = seg_data.copy()
        seg_data = (seg_data != 0).astype(np.uint8)

    binary_dilation_structure = ndi.iterate_structure(ndi.generate_binary_structure(3, 1), dilate)
    output_seg_data = np.zeros_like(seg_data)

    for l in [_ for _ in np.unique(seg_data) if _ != 0]:
        mask = seg_data == l
        if dilate > 0:
            # Dilate
            mask_labeled, num_labels = ndi.label(ndi.binary_dilation(mask, binary_dilation_structure), np.ones((3, 3, 3)))
            # Undo dilate
            mask_labeled *= mask
        else:
            mask_labeled, num_labels = ndi.label(mask, np.ones((3, 3, 3)))
        # Find the label of the largest component
        label_sizes = np.bincount(mask_labeled.ravel())[1:]  # Skip 0 label size
        largest_label = label_sizes.argmax() + 1  # +1 because bincount labels start at 0
        output_seg_data[mask_labeled == largest_label] = l

    if binarize:
        output_seg_data = output_seg_data * seg_data_src

    output_seg = nib.Nifti1Image(output_seg_data, seg.affine, seg.header)

    return output_seg

if __name__ == '__main__':
    main()