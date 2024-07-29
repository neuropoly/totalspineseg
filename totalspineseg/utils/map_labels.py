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
        '--output-dir', '-o', type=Path, required=True, 
        help='Folder to save output segmentations for each subject.'
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
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '--add-output', action="store_true", default=False,
        help='If the output file already exists, add all the labels from the existing output to the mapped input before saving (the labels from the output are prioritized), defaults to false (Overwrite the output file).'
    )
    group.add_argument(
        '--add-input', action="store_true", default=False,
        help='If the output file already exists, add the mapped input labels to the existing output labels before saving (the labels from the input are prioritized). Defaults to false (overwrite the output file).'
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
    map_list = args.map
    subject_dir = args.subject_dir
    subject_subdir = args.subject_subdir
    prefix = args.prefix
    seg_suffix = args.seg_suffix
    output_seg_suffix = args.output_seg_suffix
    default_input = args.default_input
    add_output = args.add_output
    add_input = args.add_input
    override = args.override
    max_workers = args.max_workers
    verbose = args.verbose

    if verbose:
        print(textwrap.dedent(f'''
            Running {Path(__file__).stem} with the following params:
            segs_dir = "{segs_path}"
            output_dir = "{output_path}"
            map = {map_list}
            subject_dir = "{subject_dir}"
            subject_subdir = "{subject_subdir}"
            prefix = "{prefix}"
            seg_suffix = "{seg_suffix}"
            output_seg_suffix = "{output_seg_suffix}"
            default_input = {default_input}
            add_output = {add_output}
            add_input = {add_input}
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

    glob_pattern = ""
    if subject_dir is not None:
        glob_pattern += f"{subject_dir}*/"
    if len(subject_subdir) > 0:
        glob_pattern += f"{subject_subdir}/"
    glob_pattern += f'{prefix}*{seg_suffix}.nii.gz'

    # Process the NIfTI image and segmentation files
    segs_path_list = list(segs_path.glob(glob_pattern))

    # Create a partially-applied function with the extra arguments
    partial_map_labels = partial(
        map_labels,
        segs_path=segs_path,
        map_dict=map_dict,
        output_path=output_path,
        seg_suffix=seg_suffix,
        output_seg_suffix=output_seg_suffix,
        add_output=add_output,
        add_input=add_input,
        default_input=default_input,
        override=override,
    )

    with mp.Pool() as pool:
        process_map(partial_map_labels, segs_path_list, max_workers=max_workers)
    

def map_labels(
        seg_path,
        segs_path,
        map_dict,
        output_path,
        seg_suffix,
        output_seg_suffix,
        add_output,
        add_input,
        default_input,
        override,
    ):
    
    output_seg_path = output_path / seg_path.relative_to(segs_path).parent / seg_path.name.replace(f'{seg_suffix}.nii.gz', f'{output_seg_suffix}.nii.gz')

    # If the output image already exists and we are not overriding it, return
    if not override and output_seg_path.exists():
        return

    # Load segmentation
    seg = nib.load(seg_path)
    seg_data = np.asanyarray(seg.dataobj).round().astype(np.uint8)

    # Apply label mapping
    mapped_seg_data = np.copy(seg_data) if default_input else np.zeros_like(seg_data)
    
    # Apply label mapping for all labels that are not mapped to 0
    for orig, new in map_dict.items():
        mapped_seg_data[seg_data==int(orig)] = int(new)

    # Add new mapped labels to output
    if (add_output or add_input) and output_seg_path.is_file():
        # Load segmentation
        output_seg = nib.load(output_seg_path)
        output_seg_data = np.asanyarray(output_seg.dataobj).round().astype(np.uint8)

        if add_output:
            # Update output from existing output file
            mapped_seg_data[output_seg_data != 0] = output_seg_data[output_seg_data != 0]
        elif add_input:
            # Update output from existing output file
            output_seg_data[mapped_seg_data != 0] = mapped_seg_data[mapped_seg_data != 0]
            mapped_seg_data = output_seg_data

    # Create result segmentation 
    mapped_seg = nib.Nifti1Image(mapped_seg_data, seg.affine, seg.header)
    mapped_seg.set_qform(seg.affine)
    mapped_seg.set_sform(seg.affine)
    mapped_seg.set_data_dtype(np.uint8)

    # Make sure output directory exists
    output_seg_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save mapped segmentation
    nib.save(mapped_seg, output_seg_path)

if __name__ == '__main__':
    main()