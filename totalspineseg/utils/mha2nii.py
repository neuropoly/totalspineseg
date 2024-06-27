import sys, argparse, textwrap
import multiprocessing as mp
from pathlib import Path
from tqdm.contrib.concurrent import process_map
from functools import partial
import SimpleITK as sitk
import warnings

warnings.filterwarnings("ignore")


def main():

    # Description and arguments
    parser = argparse.ArgumentParser(
        description=textwrap.dedent(f'''
        This script processes .mha images and convert them into .nii.gz images.'''
        ),
        epilog=textwrap.dedent('''
            Examples:
            mha2nii -i images_mha -o images
            For BIDS:
            mha2nii -i . -o . --image-suffix "" --output-image-suffix "" -d "sub-" -u "anat"
        '''),
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--images-dir', '-i', type=Path, required=True,
        help='The folder where input .mha images files are located (required).'
    )
    parser.add_argument(
        '--output-dir', '-o', type=Path, required=True,
        help='The folder where output .nii.gz images will be saved (required).'
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
        '--image-suffix', type=str, default='',
        help='Image suffix, defaults to "".'
    )
    parser.add_argument(
        '--max-workers', '-w', type=int, default=mp.cpu_count(),
        help='Max worker to run in parallel proccess, defaults to multiprocessing.cpu_count().'
    )
    parser.add_argument(
        '--verbose', '-v', type=int, default=1, choices=[0, 1],
        help='verbose. 0: Display only errors/warnings, 1: Errors/warnings + info messages. Default is 1.'
    )

    # Parse the command-line arguments
    try:
        args = parser.parse_args()
    except BaseException as e:
        sys.exit()

    # Get the command-line argument values
    images_path = args.images_dir
    output_path = args.output_dir
    subject_dir = args.subject_dir
    subject_subdir = args.subject_subdir
    prefix = args.prefix
    max_workers = args.max_workers
    verbose = args.verbose
    
    # Print the argument values if verbose is enabled
    if verbose:
        print(textwrap.dedent(f''' 
            Running {Path(__file__).stem} with the following params:
            images_path = "{images_path}"
            output_dir = "{output_path}"
            subject_dir = "{subject_dir}"
            subject_subdir = "{subject_subdir}"
            prefix = "{prefix}"
            max_workers = {max_workers}
            verbose = {verbose}
        '''))

    glob_pattern = ""
    if subject_dir is not None:
        glob_pattern += f"{subject_dir}*/"
    if len(subject_subdir) > 0:
        glob_pattern += f"{subject_subdir}/"
    glob_pattern += f'{prefix}*.mha'

    # Process the NIfTI image and segmentation files
    images_path_list = list(images_path.glob(glob_pattern))

    # Create a partially-applied function with the extra arguments
    partial_mha2nii = partial(
        mha2nii,
        images_path=images_path,
        output_path=output_path,
    )

    with mp.Pool() as pool:
        process_map(partial_mha2nii, images_path_list, max_workers=max_workers)
    

def mha2nii(image_path, images_path, output_path):
    
    output_nii_path = output_path / image_path.relative_to(images_path).parent / image_path.name.replace(f'.mha', f'.nii.gz')
    sitk.WriteImage(sitk.ReadImage(image_path), output_nii_path)

if __name__ == '__main__':
    main()
