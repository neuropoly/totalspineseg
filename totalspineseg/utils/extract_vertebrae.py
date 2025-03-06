import sys, argparse, textwrap
from pathlib import Path
import numpy as np
import nibabel as nib
import multiprocessing as mp
from functools import partial
from tqdm.contrib.concurrent import process_map
import warnings
import json

warnings.filterwarnings("ignore")

def main():

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Extract vertebrae labels from discs levels.',
    )

    parser.add_argument(
        '--levels-dir', '-s', type=Path, required=True,
        help='Folder containing input levels.'
    )
    parser.add_argument(
        '--output-dir', '-o', type=Path, required=True,
        help='Folder to save output jsons.'
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
        '--disc-dict', type=dict, required=True,
        help='The disc dict used for the levels.'
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
    levels_dir = args.levels_dir
    output_dir = args.output_dir
    prefix = args.prefix
    seg_suffix = args.seg_suffix
    output_seg_suffix = args.output_seg_suffix
    disc_dict = args.disc_dict
    overwrite = args.overwrite
    max_workers = args.max_workers
    quiet = args.quiet

    # Print the argument values if not quiet
    if not quiet:
        print(textwrap.dedent(f'''
            Running {Path(__file__).stem} with the following params:
            levels_dir = "{levels_dir}"
            output_dir = "{output_dir}"
            prefix = "{prefix}"
            seg_suffix = "{seg_suffix}"
            output_seg_suffix = "{output_seg_suffix}"
            disc_dict = {disc_dict}
            overwrite = {overwrite}
            max_workers = {max_workers}
            quiet = {quiet}
        '''))

    extract_levels_mp(
        levels_dir=levels_dir,
        output_dir=output_dir,
        disc_dict=disc_dict,
        prefix=prefix,
        seg_suffix=seg_suffix,
        output_seg_suffix=output_seg_suffix,
        overwrite=overwrite,
        max_workers=max_workers,
        quiet=quiet,
    )

def extract_vertebrae_mp(
        levels_dir,
        output_dir,
        disc_dict,
        prefix='',
        seg_suffix='',
        output_seg_suffix='',
        overwrite=False,
        max_workers=mp.cpu_count(),
        quiet=False,
    ):
    '''
    Wrapper function to handle multiprocessing.
    '''
    levels_path = Path(levels_dir)
    output_path = Path(output_dir)

    glob_pattern = f'{prefix}*{seg_suffix}.nii.gz'

    # Process the NIfTI image and segmentation files
    levels_path_list = list(levels_path.glob(glob_pattern))
    output_levels_path_list = [output_path / _.relative_to(levels_path).parent / _.name.replace(f'{seg_suffix}.nii.gz', f'{output_seg_suffix}.json') for _ in levels_path_list]

    process_map(
        partial(
            extract_vertebrae,
            disc_dict=disc_dict,
            overwrite=overwrite,
        ),
        levels_path_list,
        output_levels_path_list,
        max_workers=max_workers,
        chunksize=1,
        disable=quiet,
    )

def extract_vertebrae(
        level_path,
        output_path,
        disc_dict,
        overwrite=False,
    ):
    '''
    Extract vertebrae from discs levels
    '''
    level_path = Path(level_path)
    output_path = Path(output_path)

    # If the output image already exists and we are not overriding it, return
    if not overwrite and output_path.exists():
        return

    # Make sure output directory exists and save the segmentation
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load segmentation
    seg = nib.load(level_path)
    seg_data = np.asanyarray(seg.dataobj).round().astype(np.uint8)

    # Extract discs labels
    uniq_labels = [disc_dict[v] for v in np.unique(seg_data) if v not in [0, 61]]

    # Extract vertebrae labels
    vert_list = []
    for disc in uniq_labels:
        # Split disc label into the two adjacent vertebrae
        i = 1
        char = disc[i]
        while not char.isalpha():
            i += 1
            char = disc[i]
        vert_list.append(disc[:i])
        vert_list.append(disc[i:])
    
    vert_dict = {'vertebrae':np.unique(vert_list).tolist()}

    # Save list in JSON
    with open(output_path, 'w') as f:
        json.dump(vert_dict, f, indent=4)


if __name__ == '__main__':
    main()
