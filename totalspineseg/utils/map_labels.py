import sys, argparse, textwrap, json
import numpy as np
import nibabel as nib
import multiprocessing as mp
from functools import partial
from pathlib import Path
from tqdm.contrib.concurrent import process_map
import warnings

warnings.filterwarnings("ignore")

def main():

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description=textwrap.dedent(f'''
            Map segmentation labels to other, new labels using json or a dict mapping.
        '''),
        epilog=textwrap.dedent('''
            Examples:
            map_labels -s labels -o labels_mapped -m map.json
            map_labels -s labels -o labels_mapped -m 92:1 93:2
            map_labels -s labels -o labels_mapped  -m map.json -d sub- -s anat
            For BIDS:
            map_labels -s derivatives/labels -o derivatives/labels -m map.json --seg-suffix "_seg" --output-seg-suffix "_seg_mapped" -d "sub-" -u "anat"
        '''),
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        '--segs-dir', '-s', type=Path, required=True,
        help='Folder containing input segmentations for each subject.'
    )
    parser.add_argument(
        '--output-segs-dir', '-o', type=Path, required=True,
        help='Folder to save output segmentations for each subject.'
    )
    parser.add_argument(
        '--update-segs-dir', type=Path, default=None,
        help='Folder containing segmentations to be update. We update this segmentations with all non-zero values of the mapped input labels.'
    )
    parser.add_argument(
        '--update-from-segs-dir', type=Path, default=None,
        help='Folder containing segmentations to be update from. We update the mapped input labels with all non-zero values of this segmentations.'
    )
    parser.add_argument(
        '--map', '-m', type=str, nargs='+', default=[],
        help=textwrap.dedent('''
            JSON file or dict mapping each input_label to an output_label.
            The format should be input_label:output_label without any spaces.
            For example, you can use a JSON file like map.json containing {"1": 2, "2": 15},
            or provide a dict directly like 1:2 2:15
        '''),
    )
    parser.add_argument(
        '--subject-dir', '-d', type=str, default=None, nargs='?', const='',
        help=textwrap.dedent('''
            Is every subject has its oen direcrory.
            If this argument will be provided without value it will look for any directory in the segmentation directory.
            If value also provided it will be used as a prefix to subject directory, defaults to False (no subjet directory).
        '''),
    )
    parser.add_argument(
        '--subject-subdir', '-u', type=str, default='',
        help='Subfolder inside subject folder containing masks, defaults to no subfolder.'
    )
    parser.add_argument(
        '--prefix', '-p', type=str, default='',
        help='File prefix to work on.'
    )
    parser.add_argument(
        '--seg-suffix', type=str, default='',
        help='Suffix for input segmentation, defaults to "".'
    )
    parser.add_argument(
        '--output-seg-suffix', type=str, default='',
        help='Suffix for output segmentation, defaults to "".'
    )
    parser.add_argument(
        '--update-seg-suffix', type=str, default='',
        help='Suffix for --update-segs-dir segmentation, defaults to "".'
    )
    parser.add_argument(
        '--update-from-seg-suffix', type=str, default='',
        help='Suffix for --update-from-segs-dir segmentation, defaults to "".'
    )
    parser.add_argument(
        '--keep-unmapped', action="store_true", default=False,
        help='Keep unmapped labels as they are in the output segmentation, defaults to false (discard unmapped labels).'
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
    output_segs_path = args.output_segs_dir
    update_segs_path = args.update_segs_dir
    update_from_segs_path = args.update_from_segs_dir
    map_list = args.map
    subject_dir = args.subject_dir
    subject_subdir = args.subject_subdir
    prefix = args.prefix
    seg_suffix = args.seg_suffix
    output_seg_suffix = args.output_seg_suffix
    update_seg_suffix = args.update_seg_suffix
    update_from_seg_suffix = args.update_from_seg_suffix
    keep_unmapped = args.keep_unmapped
    override = args.override
    max_workers = args.max_workers
    verbose = args.verbose

    if verbose:
        print(textwrap.dedent(f'''
            Running {Path(__file__).stem} with the following params:
            segs_dir = "{segs_path}"
            output_segs_dir = "{output_segs_path}"
            update_segs_dir = "{update_segs_path}"
            update_from_segs_dir = "{update_from_segs_path}"
            map = {map_list}
            subject_dir = "{subject_dir}"
            subject_subdir = "{subject_subdir}"
            prefix = "{prefix}"
            seg_suffix = "{seg_suffix}"
            output_seg_suffix = "{output_seg_suffix}"
            update_seg_suffix = "{update_seg_suffix}"
            update_from_seg_suffix = "{update_from_seg_suffix}"
            keep_unmapped = {keep_unmapped}
            override = {override}
            max_workers = {max_workers}
            verbose = {verbose}
        '''))

    # Load map into a dict
    if len(map_list) == 1 and map_list[0][-5:] == '.json':
        # Load label mappings from JSON file
        with open(map_list[0], 'r', encoding='utf-8') as map_file:
            map_dict = json.load(map_file)
    else:
        try:
            map_dict = {int(l_in): int(l_out) for l_in, l_out in map(lambda x:x.split(':'), map_list)}
        except:
            raise ValueError("Input param map is not in the right structure. Make sure it is in the right format, e.g., 1:2 3:5")

    map_labels_mp(
        segs_path=segs_path,
        output_segs_path=output_segs_path,
        update_segs_path=update_segs_path,
        update_from_segs_path=update_from_segs_path,
        map_dict=map_dict,
        subject_dir=subject_dir,
        subject_subdir=subject_subdir,
        prefix=prefix,
        seg_suffix=seg_suffix,
        output_seg_suffix=output_seg_suffix,
        update_seg_suffix=update_seg_suffix,
        update_from_seg_suffix=update_from_seg_suffix,
        keep_unmapped=keep_unmapped,
        override=override,
        max_workers=max_workers,
    )

def map_labels_mp(
        segs_path,
        output_segs_path,
        update_segs_path=None,
        update_from_segs_path=None,
        map_dict={},
        subject_dir=None,
        subject_subdir='',
        prefix='',
        seg_suffix='',
        output_seg_suffix='',
        update_seg_suffix='',
        update_from_seg_suffix='',
        keep_unmapped=False,
        override=False,
        max_workers=mp.cpu_count()
    ):
    segs_path = Path(segs_path)
    output_segs_path = Path(output_segs_path)
    update_segs_path = update_segs_path and Path(update_segs_path)
    update_from_segs_path = update_from_segs_path and Path(update_from_segs_path)

    glob_pattern = ""
    if subject_dir is not None:
        glob_pattern += f"{subject_dir}*/"
    if len(subject_subdir) > 0:
        glob_pattern += f"{subject_subdir}/"
    glob_pattern += f'{prefix}*{seg_suffix}.nii.gz'

    # Process the NIfTI image and segmentation files
    seg_path_list = list(segs_path.glob(glob_pattern))
    output_seg_path_list = [output_segs_path / _.relative_to(segs_path).parent / _.name.replace(f'{seg_suffix}.nii.gz', f'{output_seg_suffix}.nii.gz') for _ in seg_path_list]
    update_seg_path_list = [update_segs_path and update_segs_path / _.relative_to(segs_path).parent / _.name.replace(f'{seg_suffix}.nii.gz', f'{update_seg_suffix}.nii.gz') for _ in seg_path_list]
    update_from_seg_path_list = [update_from_segs_path and update_from_segs_path / _.relative_to(segs_path).parent / _.name.replace(f'{seg_suffix}.nii.gz', f'{update_from_seg_suffix}.nii.gz') for _ in seg_path_list]

    process_map(
        partial(
            _map_labels,
            map_dict=map_dict,
            keep_unmapped=keep_unmapped,
            override=override,
        ),
        seg_path_list,
        output_seg_path_list,
        update_seg_path_list,
        update_from_seg_path_list,
        max_workers=max_workers,
    )

def _map_labels(
        seg_path,
        output_seg_path,
        update_seg_path=None,
        update_from_seg_path=None,
        map_dict={},
        keep_unmapped=False,
        override=False,
    ):
    seg_path = Path(seg_path)
    output_seg_path = Path(output_seg_path)
    update_seg_path = update_seg_path and Path(update_seg_path)
    update_from_seg_path = update_from_seg_path and Path(update_from_seg_path)

    # If the output image already exists and we are not overriding it, return
    if not override and output_seg_path.exists():
        return

    # Load segmentation
    seg = nib.load(seg_path)
    update_seg = update_seg_path and nib.load(update_seg_path)
    update_from_seg = update_from_seg_path and nib.load(update_from_seg_path)

    output_seg = map_labels(
        seg=seg,
        update_seg=update_seg,
        update_from_seg=update_from_seg,
        map_dict=map_dict,
        keep_unmapped=keep_unmapped,
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

def map_labels(
        seg,
        update_seg=None,
        update_from_seg=None,
        map_dict={},
        keep_unmapped=False,
    ):
    seg_data = np.asanyarray(seg.dataobj).round().astype(np.uint8)
    update_seg_data = update_seg and np.asanyarray(update_seg.dataobj).round().astype(np.uint8)
    update_from_seg_data = update_from_seg and np.asanyarray(update_from_seg.dataobj).round().astype(np.uint8)

    # Apply label mapping
    output_seg_data = np.copy(seg_data) if keep_unmapped else np.zeros_like(seg_data)

    # Apply label mapping for all labels that are not mapped to 0
    for orig, new in map_dict.items():
        output_seg_data[seg_data==int(orig)] = int(new)

    # Update the output segmentation with the update_seg_data where the output_seg_data is 0
    if update_seg_data:
        output_seg_data[output_seg_data == 0] = update_seg_data[output_seg_data == 0]

    # Update the output segmentation with the update_from_seg_data where it is not 0
    if update_from_seg_data:
        output_seg_data[update_from_seg_data != 0] = update_from_seg_data[update_from_seg_data != 0]

    output_seg = nib.Nifti1Image(output_seg_data, seg.affine, seg.header)

    return output_seg

if __name__ == '__main__':
    main()