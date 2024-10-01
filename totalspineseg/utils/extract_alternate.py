import sys, argparse, textwrap
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
        description=' '.join(f'''
            Extract alternate labels from the segmentation.
            This script extracts binary masks that include every other intervertebral discs (IVD).
            It loops through the segmentation labels from superior to inferior, selecting alternating discs.
            To choose the first IVD to include, it uses the first disc in the image that matches the labels provided in the --prioratize-labels argument, if supplied.
            If --prioratize-labels is not provided, it starts from the first disc in the image.
            For inference purposes, this prioritization is not needed, as the goal is simply to include every other disc in the mask, without concern for which disc is selected first.
        '''.split()),
        epilog=textwrap.dedent('''
            Examples:
            extract_alternate -s labels -o levels --labels 60-100 -r
            For BIDS:
            extract_alternate -s derivatives/labels -o derivatives/labels --seg-suffix "_seg" --output-seg-suffix "_levels" -p "sub-*/anat/" --labels 60-100 -r
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
        '--labels', type=lambda x:list(range(int(x.split('-')[0]), int(x.split('-')[-1]) + 1)), nargs='+', required=True,
        help='The labels to extract alternate elements from.'
    )
    parser.add_argument(
        '--prioratize-labels', type=lambda x:list(range(int(x.split('-')[0]), int(x.split('-')[-1]) + 1)), nargs='+', default=[],
        help='Specify labels that will be prioratized in the output, the first label in the list will be included in the output, defaults to [] (The first label in the list that is in the segmentation).'
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
    labels = [l for raw in args.labels for l in (raw if isinstance(raw, list) else [raw])]
    prioratize_labels = [l for raw in args.prioratize_labels for l in (raw if isinstance(raw, list) else [raw])]
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
            labels = {labels}
            prioratize_labels = {prioratize_labels}
            overwrite = {overwrite}
            max_workers = {max_workers}
            quiet = {quiet}
        '''))

    extract_alternate_mp(
        segs_path=segs_path,
        output_segs_path=output_segs_path,
        prefix=prefix,
        seg_suffix=seg_suffix,
        output_seg_suffix=output_seg_suffix,
        labels=labels,
        prioratize_labels=prioratize_labels,
        overwrite=overwrite,
        max_workers=max_workers,
        quiet=quiet,
    )

def extract_alternate_mp(
        segs_path,
        output_segs_path,
        prefix='',
        seg_suffix='',
        output_seg_suffix='',
        labels=[],
        prioratize_labels=[],
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
            _extract_alternate,
            labels=labels,
            prioratize_labels=prioratize_labels,
            overwrite=overwrite,
        ),
        seg_path_list,
        output_seg_path_list,
        max_workers=max_workers,
        chunksize=1,
        disable=quiet,
    )

def _extract_alternate(
        seg_path,
        output_seg_path,
        labels=[],
        prioratize_labels=[],
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

    try:
        output_seg = extract_alternate(
            seg,
            labels=labels,
            prioratize_labels=prioratize_labels,
        )
    except ValueError as e:
        output_seg_path.is_file() and output_seg_path.unlink()
        print(f'Error: {seg_path}, {e}')
        return

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

def extract_alternate(
        seg,
        labels=[],
        prioratize_labels=[],
    ):
    '''
    This function extracts binary masks that include every other intervertebral discs (IVD).
    It loops through the segmentation labels from superior to inferior, selecting alternating discs.
    To choose the first IVD to include, it uses the first disc in the image that matches the labels provided in the prioratize_labels argument, if supplied.
    If prioratize_labels is not provided, it starts from the first disc in the image.
    For inference purposes, this prioritization is not needed, as the goal is simply to include every other disc in the mask, without concern for which disc is selected first.

    Parameters
    ----------
    seg : nibabel.Nifti1Image
        The input segmentation.
    labels : list of int
        The labels to extract alternate elements from.
    prioratize_labels : list of int
        Specify labels that will be prioratized in the output, the first label in the list will be included in the output.

    Returns
    -------
    nibabel.Nifti1Image
        The output segmentation with the vertebrae levels.
    '''
    seg_data = np.asanyarray(seg.dataobj).round().astype(np.uint8)

    output_seg_data = np.zeros_like(seg_data)

    # Get the labels in the segmentation
    labels = np.array(labels)[np.isin(labels, seg_data)]

    # Get the labels to prioratize in the output that are in the segmentation and in the labels
    prioratize_labels = np.array(prioratize_labels)[np.isin(prioratize_labels, labels)]

    selected_labels = labels[::2]

    if len(prioratize_labels) > 0 and prioratize_labels[0] not in selected_labels:
        selected_labels = labels[1::2]

    output_seg_data[np.isin(seg_data, selected_labels)] = 1

    output_seg = nib.Nifti1Image(output_seg_data, seg.affine, seg.header)

    return output_seg

if __name__ == '__main__':
    main()