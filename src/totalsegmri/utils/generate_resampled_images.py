import sys, argparse, textwrap
import multiprocessing as mp
from functools import partial
from tqdm.contrib.concurrent import process_map
from pathlib import Path
import nibabel as nib
import numpy as np
import torchio as tio
import gryds
import warnings

warnings.filterwarnings("ignore")


def main():

    # Description and arguments
    parser = argparse.ArgumentParser(
        description=textwrap.dedent(f'''
        This script processes NIfTI (Neuroimaging Informatics Technology Initiative) image and segmentation files.
        It transform image and segmentations into mXmXm mm resolution.'''
        ),
        epilog=textwrap.dedent('''
            Examples:
            generate_resampled_images -i images -s labels -o images -g labels -m 1
            For BIDS:
            generate_resampled_images -i . -s derivatives/labels -o . -g derivatives/labels --image-suffix "" --output-image-suffix "" --seg-suffix "_seg" --output-seg-suffix "_seg" -d "sub-" -u "anat" -m 1
        '''),
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--images-dir', '-i', type=Path, required=True,
        help='The folder where input NIfTI images files are located (required).'
    )
    parser.add_argument(
        '--segs-dir', '-s', type=Path, default=None,
        help='The folder where input NIfTI segmentation files are located (required).'
    )
    parser.add_argument(
        '--output-images-dir', '-o', type=Path, required=True,
        help='The folder where output augmented images will be saved with _a1, _a2 etc. suffixes (required).'
    )
    parser.add_argument(
        '--output-segs-dir', '-g', type=Path, default=None,
        help='The folder where output augmented segmentation will be saved with _a1, _a2 etc. suffixes (required).'
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
        '--seg-suffix', type=str, default='',
        help='Segmentation suffix, defaults to "".'
    )
    parser.add_argument(
        '--output-image-suffix', type=str, default='_0000',
        help='Image suffix for output, defaults to "_0000".'
    )
    parser.add_argument(
        '--output-seg-suffix', type=str, default='',
        help='Segmentation suffix for output, defaults to "".'
    )
    parser.add_argument(
        '--mm', '-m', type=int, nargs='+', default=[1],
        help='The target voxel size in mm. Can accept 1 or 3 parameters for x, y, z. Default is 1mm.'
    )
    parser.add_argument(
        '--max-workers', '-w', type=int, default=min(32, mp.cpu_count() + 4),
        help='Max worker to run in parallel proccess, defaults to min(32, mp.cpu_count() + 4).'
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
    segs_path = args.segs_dir
    output_images_path = args.output_images_dir
    output_segs_path = args.output_segs_dir
    subject_dir = args.subject_dir
    subject_subdir = args.subject_subdir
    prefix = args.prefix
    image_suffix = args.image_suffix
    seg_suffix = args.seg_suffix
    output_image_suffix = args.output_image_suffix
    output_seg_suffix = args.output_seg_suffix
    mm = tuple(args.mm if len (args.mm) == 3 else [args.mm[0]] * 3)
    max_workers = args.max_workers
    verbose = args.verbose
    
    # Print the argument values if verbose is enabled
    if verbose:
        print(textwrap.dedent(f''' 
            Running {Path(__file__).stem} with the following params:
            images_path = "{images_path}"
            segs_path = "{segs_path}"
            output_images_path = "{output_images_path}"
            output_segs_path = "{output_segs_path}"
            subject_dir = "{subject_dir}"
            subject_subdir = "{subject_subdir}"
            prefix = "{prefix}"
            image_suffix = "{image_suffix}"
            seg_suffix = "{seg_suffix}"
            output_image_suffix = "{output_image_suffix}"
            output_seg_suffix = "{output_seg_suffix}"
            mm = "{mm}"
            max_workers = "{max_workers}"
            verbose = "{verbose}"
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
    partial_generate_resampled_images = partial(
        generate_resampled_images,
        images_path=images_path,
        segs_path=segs_path,
        output_images_path=output_images_path,
        output_segs_path=output_segs_path,
        image_suffix=image_suffix,
        seg_suffix=seg_suffix,
        output_image_suffix=output_image_suffix,
        output_seg_suffix=output_seg_suffix,
        mm=mm,
    )

    with mp.Pool() as pool:
        process_map(partial_generate_resampled_images, images_path_list, max_workers=max_workers)


def generate_resampled_images(
        image_path,
        images_path,
        segs_path,
        output_images_path,
        output_segs_path,
        image_suffix,
        seg_suffix,
        output_image_suffix,
        output_seg_suffix,
        mm,
    ):
    
    
    if segs_path:
        seg_path = segs_path / image_path.relative_to(images_path).parent /  image_path.name.replace(f'{image_suffix}.nii.gz', f'{seg_suffix}.nii.gz')
        
        # Create result
        subject = tio.Resample(mm)(tio.Subject(
            image=tio.ScalarImage(image_path),
            seg=tio.LabelMap(seg_path),
        ))
        output_image, output_seg = subject.image, subject.seg
        
        output_seg_path = output_segs_path / image_path.relative_to(images_path).parent / seg_path.name.replace(f'{seg_suffix}.nii.gz', f'{output_seg_suffix}.nii.gz')

        output_seg_path.parent.mkdir(parents=True, exist_ok=True)
        output_seg.save(output_seg_path)

    else:
        # Create result
        subject = tio.Resample(mm)(tio.Subject(
            image=tio.ScalarImage(image_path),
        ))
        output_image = subject.image

    output_image_path = output_images_path / image_path.relative_to(images_path).parent / image_path.name.replace(f'{image_suffix}.nii.gz', f'{output_image_suffix}.nii.gz')

    # Make sure output directory exists
    output_image_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save mapped segmentation
    output_image.save(output_image_path)

if __name__ == '__main__':
    main()