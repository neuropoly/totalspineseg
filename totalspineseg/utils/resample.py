import sys, argparse, textwrap
import multiprocessing as mp
from functools import partial
from tqdm.contrib.concurrent import process_map
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
            This script processes NIfTI (Neuroimaging Informatics Technology Initiative) image and segmentation files.
            It transform image into mXmXm mm resolution.
        '''.split()),
        epilog=textwrap.dedent('''
            Examples:
            resample -i images -o images
            For BIDS:
            resample -i . -o . --image-suffix "" --output-image-suffix "" -p "sub-*/anat/" -m 1
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
        '--mm', '-m', type=float, nargs='+', default=[1.0],
        help='The target voxel size in mm. Can accept 1 or 3 parameters for x, y, z. Default is 1mm.'
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
    mm = tuple(args.mm if len (args.mm) == 3 else [args.mm[0]] * 3)
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
            mm = {mm}
            overwrite = {overwrite}
            max_workers = {max_workers}
            quiet = {quiet}
        '''))

    resample_mp(
        images_path=images_path,
        output_images_path=output_images_path,
        prefix=prefix,
        image_suffix=image_suffix,
        output_image_suffix=output_image_suffix,
        mm=mm,
        overwrite=overwrite,
        max_workers=max_workers,
        quiet=quiet,
    )

def resample_mp(
        images_path,
        output_images_path,
        prefix='',
        image_suffix='_0000',
        output_image_suffix='_0000',
        mm=(1.0, 1.0, 1.0),
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
            _resample,
            mm=mm,
            overwrite=overwrite,
        ),
        image_path_list,
        output_image_path_list,
        max_workers=max_workers,
        chunksize=1,
        disable=quiet,
    )

def _resample(
        image_path,
        output_image_path,
        mm=(1.0, 1.0, 1.0),
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

    output_image = resample(image, mm)

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

def resample(
        image,
        mm = (1.0, 1.0, 1.0),
    ):
    '''
    Resample the image to the target voxel size in mm.

    Parameters
    ----------
    image : nibabel.Nifti1Image
        The input image to resample.
    mm : tuple[float, float, float]
        The target voxel size in mm.

    Returns
    -------
    nibabel.Nifti1Image
        The resampled image.
    '''
    image_data = np.asanyarray(image.dataobj).astype(np.float64)

    # Create result
    subject = tio.Resample(mm)(tio.Subject(
        image=tio.ScalarImage(tensor=image_data[None, ...], affine=image.affine),
    ))
    output_image_data = subject.image.data.numpy()[0, ...].astype(np.float64)

    output_image = nib.Nifti1Image(output_image_data, subject.image.affine, image.header)
    # Set qform twice to ensure consistent header information
    # This addresses an issue where the second call to set_qform may alter output_image.header['pixdim']
    # Applying it here prevents inconsistencies that could arise from later image edits
    output_image.set_qform(output_image.affine)
    output_image.set_qform(output_image.affine)

    return output_image

if __name__ == '__main__':
    main()