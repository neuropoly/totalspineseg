import sys, argparse, textwrap
import multiprocessing as mp
from functools import partial
from tqdm.contrib.concurrent import process_map
from pathlib import Path
import nibabel as nib
import numpy as np
import warnings

warnings.filterwarnings("ignore")


def main():

    # Description and arguments
    parser = argparse.ArgumentParser(
        description=' '.join(f'''
            This script processes NIfTI (Neuroimaging Informatics Technology Initiative) images and average 4D images last dimension to 3D image.
        '''.split()),
        epilog=textwrap.dedent('''
            Examples:
            average4d -i images -o images
            For BIDS:
            average4d -i . -o . --image-suffix "" --output-image-suffix "" -p "sub-*/anat/"
        '''),
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--images-dir', '-i', type=Path, required=True,
        help='The folder where input NIfTI images files are located (required).'
    )
    parser.add_argument(
        '--output-images-dir', '-o', type=Path, required=True,
        help='The folder where output images will be saved (required).'
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
    images_path = args.images_dir
    output_images_path = args.output_images_dir
    prefix = args.prefix
    image_suffix = args.image_suffix
    output_image_suffix = args.output_image_suffix
    overwrite = args.overwrite
    max_workers = args.max_workers
    quiet = args.quiet

    # Print the argument values if not quiet
    if not quiet:
        print(textwrap.dedent(f'''
            Running {Path(__file__).stem} with the following params:
            images_path = "{images_path}"
            output_images_path = "{output_images_path}"
            prefix = "{prefix}"
            image_suffix = "{image_suffix}"
            output_image_suffix = "{output_image_suffix}"
            overwrite = {overwrite}
            max_workers = {max_workers}
            quiet = {quiet}
        '''))

    average4d_mp(
        images_path=images_path,
        output_images_path=output_images_path,
        prefix=prefix,
        image_suffix=image_suffix,
        output_image_suffix=output_image_suffix,
        overwrite=overwrite,
        max_workers=max_workers,
        quiet=quiet,
    )

def average4d_mp(
        images_path,
        output_images_path,
        prefix='',
        image_suffix='_0000',
        output_image_suffix='_0000',
        overwrite=False,
        max_workers=mp.cpu_count(),
        quiet=False,
    ):
    '''
    Wrapper function to handle multiprocessing.
    '''
    images_path = Path(images_path)
    output_images_path = Path(output_images_path)

    glob_pattern = f'{prefix}*{image_suffix}.nii.gz'

    # Process the NIfTI image and segmentation files
    image_path_list = list(images_path.glob(glob_pattern))
    output_image_path_list = [output_images_path / _.relative_to(images_path).parent / _.name.replace(f'{image_suffix}.nii.gz', f'{output_image_suffix}.nii.gz') for _ in image_path_list]

    process_map(
        partial(
            _average4d,
            overwrite=overwrite,
        ),
        image_path_list,
        output_image_path_list,
        max_workers=max_workers,
        chunksize=1,
        disable=quiet,
    )

def _average4d(
        image_path,
        output_image_path,
        overwrite=False,
    ):
    '''
    Wrapper function to handle IO.
    '''
    image_path = Path(image_path)
    output_image_path = Path(output_image_path)

    # If the output image already exists and we are not overriding it, return
    if not overwrite and output_image_path.exists():
        return

    image = nib.load(image_path)

    # Get image dtype from the image data (preferred over header dtype to avoid data loss)
    image_data_dtype = getattr(np, np.asanyarray(image.dataobj).dtype.name)

    # Rescale the image to the output dtype range if necessary
    # Modified from https://github.com/spinalcordtoolbox/spinalcordtoolbox/blob/6.3/spinalcordtoolbox/image.py#L1217
    if "int" in np.dtype(image_data_dtype).name:
        image_data = np.asanyarray(image.dataobj).astype(np.float64)
        image_min, image_max = image_data.min(), image_data.max()
        dtype_min, dtype_max = np.iinfo(image_data_dtype).min, np.iinfo(image_data_dtype).max
        if (image_min < dtype_min) or (dtype_max < image_max):
            data_rescaled = image_data * (dtype_max - dtype_min) / (image_max - image_min)
            image_data = data_rescaled - (data_rescaled.min() - dtype_min)
            image = nib.Nifti1Image(image_data.astype(image_data_dtype), image.affine, image.header)

    output_image = average4d(image)

    # Ensure correct image dtype, affine and header
    output_image = nib.Nifti1Image(
        np.asanyarray(output_image.dataobj).astype(image_data_dtype),
        output_image.affine, output_image.header
    )
    output_image.set_data_dtype(image_data_dtype)
    output_image.set_qform(output_image.affine)
    output_image.set_sform(output_image.affine)

    # Make sure output directory exists and save the image
    output_image_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(output_image, output_image_path)

def average4d(
        image,
    ):
    '''
    Average the last dimension of a 4D image to get a 3D image.

    Parameters
    ----------
    image : nibabel.Nifti1Image
        Input 4D image.

    Returns
    -------
    nibabel.Nifti1Image
        Output 3D image.
    '''
    output_image_data = np.asanyarray(image.dataobj).astype(np.float64)

    # Average the last dimension
    if len(output_image_data.shape) == 4:
        output_image_data = np.mean(output_image_data, axis=-1)

    output_image = nib.Nifti1Image(output_image_data, image.affine, image.header)

    return output_image

if __name__ == '__main__':
    main()