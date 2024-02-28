import sys, argparse, textwrap, json
import numpy as np
import nibabel as nib
import multiprocessing as mp
from functools import partial
from pathlib import Path
from tqdm.contrib.concurrent import process_map

def main():
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description=textwrap.dedent(f'''
            Map segmentation labels to other labels using json mapping file.
        '''),
        epilog=textwrap.dedent('''
            Examples:
            map_labels -s labels -o labels_maped -m map.json
            map_labels -s labels -o labels_maped  -m map.json -d sub- -s anat -w 32
        '''),
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        '--segs-dir', '-s', type=Path, required=True,
        help='Folder containing input segmentations for each subject.'
    )
    parser.add_argument(
        '--output-dir', '-o', type=Path, required=True, 
        help='Folder to save output segmentations for each subject.'
    )
    parser.add_argument(
        '--map', '-m', type=argparse.FileType('r', encoding='utf-8'), required=True,
        help='JSON file mapping each mask to a unique number, e.g. {"1": 2, "2": 15}'
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
        help='Suffix for output segmentation, defaults to "".'
    )
    parser.add_argument(
        '--output-seg-suffix', type=str, default='',
        help='Suffix for output segmentation, defaults to "".'
    )
    parser.add_argument(
        '--default-input', action="store_true", default=False,
        help='Init output from input, defaults to false (init all to 0).'
    )
    parser.add_argument(
        '--add-output', action="store_true", default=False,
        help='Add new mapped labels to output if the output file exist, defaults to false (Overrite the output file).'
    )
    parser.add_argument(
        '--max-workers', '-w', type=int, default=min(32, mp.cpu_count() + 4),
        help='Max worker to run in parallel proccess, defaults to min(32, mp.cpu_count() + 4).'
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
    map_file = args.map
    subject_dir = args.subject_dir
    subject_subdir = args.subject_subdir
    prefix = args.prefix
    seg_suffix = args.seg_suffix
    output_seg_suffix = args.output_seg_suffix
    default_input = args.default_input
    add_output = args.add_output
    max_workers = args.max_workers
    verbose = args.verbose

    if verbose:
        print(textwrap.dedent(f'''
            Running {Path(__file__).stem} with the following params:
            segs_dir = "{segs_path}"
            output_dir = "{output_path}"
            map = "{map_file.name}"
            subject_dir = "{subject_dir}"
            subject_subdir = "{subject_subdir}"
            prefix = "{prefix}"
            seg_suffix = "{seg_suffix}"
            output_seg_suffix = "{output_seg_suffix}"
            default_input = "{default_input}"
            add_output = "{add_output}"
            max_workers = "{max_workers}"
            verbose = {verbose}
        '''))

    # Load label mappings from JSON file
    map_dict = json.load(map_file)
    map_file.close()

    glob_pattern = ""
    if subject_dir is not None:
        glob_pattern += f"{subject_dir}*/"
    if len(subject_subdir) > 0:
        glob_pattern += f"{subject_subdir}/"
    glob_pattern += f'{prefix}*{seg_suffix}.nii.gz'

    # Process the NIfTI image and segmentation files
    segs_path_list = list(segs_path.glob(glob_pattern))

    # Create a partially-applied function with the extra arguments
    partial_map_seg = partial(map_seg, map_dict=map_dict, output_path=output_path, seg_suffix=seg_suffix, output_seg_suffix=output_seg_suffix, add_output=add_output, default_input=default_input)

    with mp.Pool() as pool:
        process_map(partial_map_seg, segs_path_list, max_workers=max_workers)
    

def map_seg(seg_path, map_dict, output_path, seg_suffix, output_seg_suffix, add_output, default_input):
    
    output_seg_path = output_path / seg_path.name.replace(f'{seg_suffix}.nii.gz', f'{output_seg_suffix}.nii.gz')

    # Load segmentation
    seg = nib.load(seg_path)
    seg_data = seg.get_fdata()

    # Convert data to uint8 to avoid issues with segmentation IDs
    seg_data = seg_data.astype(np.uint8)

    # Apply label mapping
    mapped_seg_data = np.copy(seg_data) if default_input else np.zeros_like(seg_data)
    
    # Apply label mapping for all labels that are not mapped to 0
    for orig, new in map_dict.items():
        mapped_seg_data[seg_data==int(orig)] = int(new)

    # Add new mapped labels to output
    if add_output and output_seg_path.is_file():
        # Load segmentation
        output_seg = nib.load(output_seg_path)
        output_seg_data = output_seg.get_fdata()

        # Convert data to uint8 to avoid issues with segmentation IDs
        output_seg_data = output_seg_data.astype(np.uint8)

        # Update output from existing output file
        mapped_seg_data[output_seg_data != 0] = output_seg_data[output_seg_data != 0]

    # Create result segmentation 
    mapped_seg = nib.Nifti1Image(mapped_seg_data, seg.affine, seg.header)
    mapped_seg.set_data_dtype(np.uint8)

    # Make sure output directory exists
    output_seg_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save mapped segmentation
    nib.save(mapped_seg, output_seg_path)

if __name__ == '__main__':
    main()