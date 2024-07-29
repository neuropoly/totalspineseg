import sys, argparse, textwrap
import multiprocessing as mp
from functools import partial
from tqdm.contrib.concurrent import process_map
from scipy.ndimage import laplace
from pathlib import Path
import nibabel as nib
import numpy as np
import warnings

warnings.filterwarnings("ignore")


def main():

    # Description and arguments
    parser = argparse.ArgumentParser(
        description=textwrap.dedent(f'''
        This script processes NIfTI (Neuroimaging Informatics Technology Initiative) image and generate laplace trasformed images.'''
        ),
        epilog=textwrap.dedent('''
            Examples:
            laplace -i images -o images
            For BIDS:
            laplace -i . -o . --image-suffix "" --output-image-suffix "_laplace" -d "sub-" -u "anat"
        '''),
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--images-dir', '-i', type=Path, required=True,
        help='The folder where input NIfTI images files are located (required).'
    )
    parser.add_argument(
        '--output-images-dir', '-o', type=Path, required=True,
        help='The folder where output augmented images will be saved with _a1, _a2 etc. suffixes (required).'
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
        '--image-suffix', type=str, default='_0000',
        help='Image suffix, defaults to "_0000".'
    )
    parser.add_argument(
        '--output-image-suffix', type=str, default='_0000',
        help='Image suffix for output, defaults to "_0000".'
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
        help='verbose. 0: Display only errors/warnings, 1: Errors/warnings + info messages. Default is 1.'
    )

    # Parse the command-line arguments
    try:
        args = parser.parse_args()
    except BaseException as e:
        sys.exit()

    # Get the command-line argument values
    images_path = args.images_dir
    output_images_path = args.output_images_dir
    subject_dir = args.subject_dir
    subject_subdir = args.subject_subdir
    prefix = args.prefix
    image_suffix = args.image_suffix
    output_image_suffix = args.output_image_suffix
    override = args.override
    max_workers = args.max_workers
    verbose = args.verbose
    
    # Print the argument values if verbose is enabled
    if verbose:
        print(textwrap.dedent(f''' 
            Running {Path(__file__).stem} with the following params:
            images_path = "{images_path}"
            output_images_path = "{output_images_path}"
            subject_dir = "{subject_dir}"
            subject_subdir = "{subject_subdir}"
            prefix = "{prefix}"
            image_suffix = "{image_suffix}"
            output_image_suffix = "{output_image_suffix}"
            override = {override}
            max_workers = {max_workers}
            verbose = {verbose}
        '''))

    glob_pattern = ""
    if subject_dir is not None:
        glob_pattern += f"{subject_dir}*/"
    if len(subject_subdir) > 0:
        glob_pattern += f"{subject_subdir}/"
    glob_pattern += f'{prefix}*{image_suffix}.nii.gz'

    # Process the NIfTI image and segmentation files
    images_path_list = list(images_path.glob(glob_pattern))

    # Create a partially-applied function with the extra arguments
    partial_laplace = partial(
        laplace,
        images_path=images_path,
        output_images_path=output_images_path,
        image_suffix=image_suffix,
        output_image_suffix=output_image_suffix,
        override=override,
    )

    with mp.Pool() as pool:
        process_map(partial_laplace, images_path_list, max_workers=max_workers)


def laplace(
        image_path,
        images_path,
        output_images_path,
        image_suffix,
        output_image_suffix,
        override,
    ):

    output_image_path = output_images_path / image_path.relative_to(images_path).parent / image_path.name.replace(f'{image_suffix}.nii.gz', f'{output_image_suffix}.nii.gz')

    # If the output image already exists and we are not overriding it, return
    if not override and output_image_path.exists():
        return

    image = nib.load(image_path)

    # Get the data type of the image
    image_data = np.asanyarray(image.dataobj)
    image_data_dtype = getattr(np, image_data.dtype.name)
    image_data = image_data.astype(np.float64)

    output_image_data = np.abs(laplace(image_data))

    # Rescale the image to the output data type if necessary
    # code from https://github.com/spinalcordtoolbox/spinalcordtoolbox/blob/6.3/spinalcordtoolbox/image.py#L1217
    if "int" in np.dtype(image_data_dtype).name:
        # get min/max from output type
        min_out = np.iinfo(image_data_dtype).min
        max_out = np.iinfo(image_data_dtype).max
        min_in = output_image_data.min()
        max_in = output_image_data.max()
        if (min_in < min_out) or (max_in > max_out):
            data_rescaled = output_image_data * (max_out - min_out) / (max_in - min_in)
            output_image_data = data_rescaled - (data_rescaled.min() - min_out)

    # Make sure output directory exists and save with original header image dtype
    output_image_path.parent.mkdir(parents=True, exist_ok=True)
    output_image = nib.Nifti1Image(output_image_data.astype(image_data_dtype), image.affine, image.header)
    output_image.set_qform(image.affine)
    output_image.set_sform(image.affine)
    output_image.set_data_dtype(image_data_dtype)
    nib.save(output_image, output_image_path)

if __name__ == '__main__':
    main()