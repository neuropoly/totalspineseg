import sys, argparse, textwrap
import multiprocessing as mp
from functools import partial
from tqdm.contrib.concurrent import process_map
from pathlib import Path
import warnings
from totalspineseg.utils.image import Image

warnings.filterwarnings("ignore")


def main():

    # Description and arguments
    parser = argparse.ArgumentParser(
        description=textwrap.dedent(f'''
        This script processes NIfTI (Neuroimaging Informatics Technology Initiative) images
        and reorient them to a specific orientation.'''
        ),
        epilog=textwrap.dedent('''
            Examples:
            generate_resampled_images -i images -o images
        '''),
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--images-dir', '-i', type=Path, required=True,
        help='The folder where input NIfTI images files are located (required).'
    )
    parser.add_argument(
        '--output-images-dir', '-o', type=Path, required=True,
        help='The folder where the output images will be saved. (required).'
    )
    parser.add_argument(
        '--orientation', type=str, default='LPI',
        help='The target orientation of the output image. Default is LPI.'
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
    orientation = args.orientation
    max_workers = args.max_workers
    verbose = args.verbose

    # Print the argument values if verbose is enabled
    if verbose:
        print(textwrap.dedent(f'''
            Running {Path(__file__).stem} with the following params:
            images_path = "{images_path}"
            output_images_path = "{output_images_path}"
            orientation = "{orientation}"
            max_workers = {max_workers}
            verbose = {verbose}
        '''))

    glob_pattern = f'*.nii.gz'

    # Process the NIfTI image and segmentation files
    images_path_list = list(images_path.glob(glob_pattern))

    # Create a partially-applied function with the extra arguments
    partial_reorient_images = partial(
        reorient_images,
        images_path=images_path,
        output_images_path=output_images_path,
        orientation=orientation,
        verbose=verbose
    )

    with mp.Pool() as pool:
        process_map(partial_reorient_images, images_path_list, max_workers=max_workers)


def reorient_images(
        image_path,
        images_path,
        output_images_path,
        orientation,
        verbose
    ):

    # Load image
    image = Image(str(image_path)).change_orientation(orientation)

    # Make sure output directory exists and save with original header image dtype
    output_image_path = output_images_path / image_path.relative_to(images_path).parent / image_path.name
    output_image_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_image_path, verbose=verbose)

if __name__ == '__main__':
    main()