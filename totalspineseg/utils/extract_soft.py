import sys, argparse, textwrap
import multiprocessing as mp
from functools import partial
from tqdm.contrib.concurrent import process_map
import scipy.ndimage as ndi
from pathlib import Path
import nibabel as nib
import numpy as np
import torchio as tio
import warnings

warnings.filterwarnings("ignore")


def main():

    # Description and arguments
    parser = argparse.ArgumentParser(
        description=' '.join(f'''
            This script processes npz and NIfTI (Neuroimaging Informatics Technology Initiative) segmentation files.
            It extracts the soft segmentation from the npz file.
        '''.split()),
        epilog=textwrap.dedent('''
            Examples:
            extract_soft -n npzs -s labels -o canal_soft
            For BIDS:
            extract_soft -n derivatives/labels -s derivatives/labels -o derivatives/labels --npz-suffix "" --seg-suffix "_seg" --output-seg-suffix "_cord" -p "sub-*/anat/"
        '''),
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--npzs-dir', '-n', type=Path, required=True,
        help='The folder where input npz files are located (required).'
    )
    parser.add_argument(
        '--segs-dir', '-s', type=Path, required=True,
        help='The folder where input NIfTI segmentation files are located (required).'
    )
    parser.add_argument(
        '--output-segs-dir', '-o', type=Path, required=True,
        help='The folder where output soft segmentations will be saved (required).'
    )
    parser.add_argument(
        '--prefix', '-p', type=str, default='',
        help='File prefix to work on.'
    )
    parser.add_argument(
        '--npz-suffix', type=str, default='',
        help='Image suffix, defaults to "".'
    )
    parser.add_argument(
        '--seg-suffix', type=str, default='',
        help='Segmentation suffix, defaults to "".'
    )
    parser.add_argument(
        '--output-seg-suffix', type=str, default='',
        help='Image suffix for output, defaults to "".'
    )
    parser.add_argument(
        '--label', type=int, default=1,
        help='Label to extract, defaults to 1.'
    )
    parser.add_argument(
        '--seg-labels', type=int, nargs='+', default=[1],
        help='The labels to extract in the segmentation, defaults to 1.'
    )
    parser.add_argument(
        '--dilate', type=int, default=0,
        help='Number of voxels to dilate the segmentation before masking the soft segmentation, defaults to 0 (no dilation and no masking).'
    )
    parser.add_argument(
        '--largest', action="store_true", default=False,
        help='Take the largest component when using dilate.'
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

    # Get the command-line argument values
    npzs_path = args.npzs_dir
    segs_path = args.segs_dir
    output_segs_path = args.output_segs_dir
    prefix = args.prefix
    npz_suffix = args.npz_suffix
    seg_suffix = args.seg_suffix
    output_seg_suffix = args.output_seg_suffix
    label = args.label
    seg_labels = args.seg_labels
    dilate = args.dilate
    largest = args.largest
    overwrite = args.overwrite
    max_workers = args.max_workers
    quiet = args.quiet

    # Print the argument values if not quiet
    if not quiet:
        print(textwrap.dedent(f'''
            Running {Path(__file__).stem} with the following params:
            npzs_path = "{npzs_path}"
            segs_path = "{segs_path}"
            output_segs_path = "{output_segs_path}"
            prefix = "{prefix}"
            npz_suffix = "{npz_suffix}"
            seg_suffix = "{seg_suffix}"
            output_seg_suffix = "{output_seg_suffix}"
            label = {label}
            seg_labels = {seg_labels}
            dilate = {dilate}
            largest = {largest}
            overwrite = {overwrite}
            max_workers = {max_workers}
            quiet = {quiet}
        '''))

    extract_soft_mp(
        npzs_path=npzs_path,
        segs_path=segs_path,
        output_segs_path=output_segs_path,
        prefix=prefix,
        npz_suffix=npz_suffix,
        seg_suffix=seg_suffix,
        output_seg_suffix=output_seg_suffix,
        label=label,
        seg_labels=seg_labels,
        dilate=dilate,
        largest=largest,
        overwrite=overwrite,
        max_workers=max_workers,
        quiet=quiet,
    )

def extract_soft_mp(
        npzs_path,
        segs_path,
        output_segs_path,
        prefix='',
        npz_suffix='',
        seg_suffix='',
        output_seg_suffix='',
        label=1,
        seg_labels=[1],
        dilate=0,
        largest=False,
        overwrite=False,
        max_workers=mp.cpu_count(),
        quiet=False,
    ):
    '''
    Wrapper function to handle multiprocessing.
    '''
    npzs_path = Path(npzs_path)
    segs_path = Path(segs_path)
    output_segs_path = Path(output_segs_path)

    glob_pattern = f'{prefix}*{npz_suffix}.npz'

    # Process the NIfTI npz and segmentation files
    npz_path_list = list(npzs_path.glob(glob_pattern))
    seg_path_list = [segs_path / _.relative_to(npzs_path).parent / _.name.replace(f'{npz_suffix}.npz', f'{seg_suffix}.nii.gz') for _ in npz_path_list]
    output_segs_path_list = [output_segs_path / _.relative_to(npzs_path).parent / _.name.replace(f'{npz_suffix}.npz', f'{output_seg_suffix}.nii.gz') for _ in npz_path_list]

    process_map(
        partial(
            _extract_soft,
            label=label,
            seg_labels=seg_labels,
            dilate=dilate,
            largest=largest,
            overwrite=overwrite,
        ),
        npz_path_list,
        seg_path_list,
        output_segs_path_list,
        max_workers=max_workers,
        chunksize=1,
        disable=quiet,
    )

def _extract_soft(
        npz_path,
        seg_path,
        output_seg_path,
        label=1,
        seg_labels=[1],
        dilate=0,
        largest=False,
        overwrite=False,
    ):
    '''
    Wrapper function to handle IO.
    '''
    npz_path = Path(npz_path)
    seg_path = Path(seg_path)
    output_seg_path = Path(output_seg_path)

    # If the output seg already exists and we are not overriding it, return
    if not overwrite and output_seg_path.exists():
        return

    # Check if the segmentation file exists
    if not seg_path.is_file():
        output_seg_path.is_file() and output_seg_path.unlink()
        print(f'Error: {seg_path}, Segmentation file not found')
        return

    npz_data = np.load(npz_path)
    seg = nib.load(seg_path)

    output_seg = extract_soft(
        npz_data,
        seg,
        label=label,
        seg_labels=seg_labels,
        dilate=dilate,
        largest=largest,
    )

    # Ensure correct segmentation dtype, affine and header
    output_seg = nib.Nifti1Image(
        np.asanyarray(output_seg.dataobj).astype(np.float32),
        output_seg.affine, output_seg.header
    )
    output_seg.set_data_dtype(np.float32)
    output_seg.set_qform(output_seg.affine)
    output_seg.set_sform(output_seg.affine)

    # Make sure output directory exists and save the segmentation
    output_seg_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(output_seg, output_seg_path)

def extract_soft(
        npz_data,
        seg,
        label = 1,
        seg_labels = [1],
        dilate = 0,
        largest=False,
    ):
    '''
    Extract the soft segmentation from the npz data for the given label.

    Parameters
    ----------
    npz_data : Any
        The npz data containing the soft segmentations in 'probabilities' key.
    seg : nibabel.Nifti1Image
        The segmentation image to mask the soft segmentation.
    label : int, optional
        The label to extract, defaults to 1.
    seg_labels : list, optional
        The labels to extract in the segmentation, defaults to [1].
    dilate : int, optional
        Number of voxels to dilate the segmentation before masking the soft segmentation, defaults to 0 - no dilation and no masking.
    largest : bool, optional
        Take the largest component when using dilate, defaults to False.

    Returns
    -------
    nibabel.Nifti1Image
        The soft segmentation image.
    '''
    # Extract the soft segmentation
    output_seg_data = np.transpose(npz_data['probabilities'][label - 1], axes=(2, 1, 0))

    # Dilate the segmentation and mask the soft segmentation
    if dilate > 0:
        # Extract the segmentation mask
        seg_data = np.asanyarray(seg.dataobj).round().astype(np.uint8)
        mask = np.isin(seg_data, seg_labels)

        # Dilate the mask
        mask = ndi.binary_dilation(mask, ndi.iterate_structure(ndi.generate_binary_structure(3, 1), dilate))

        # Get the largest component
        if largest:
            mask = largest_component(mask)

        # Mask the soft segmentation
        output_seg_data[~mask] = 0

    output_seg = nib.Nifti1Image(output_seg_data, seg.affine, seg.header)

    return output_seg

def largest_component(mask):
    if mask.sum() == 0:
        return mask
    mask_labeled, num_labels = ndi.label(mask, np.ones((3, 3, 3)))
    # Find the label of the largest component
    label_sizes = np.bincount(mask_labeled.ravel())[1:]  # Skip 0 label size
    largest_label = label_sizes.argmax() + 1  # +1 because bincount labels start at 0
    return mask_labeled == largest_label

if __name__ == '__main__':
    main()