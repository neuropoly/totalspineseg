import sys, argparse, textwrap, tempfile, shutil, subprocess
from pathlib import Path
import multiprocessing as mp
from functools import partial
from tqdm.contrib.concurrent import process_map
import nibabel as nib
import numpy as np
from totalsegmri.utils.dirpath import DirPath


def main():

    # Description and arguments
    parser = argparse.ArgumentParser(
        description=textwrap.dedent(f'''
        This script processes NIfTI (Neuroimaging Informatics Technology Initiative) image and segmentation files.
        It crop images and segmentations in the most anteior voxel the lowest vertebrae in the image or at the lowest voxel of T12-L1 IVD.'''
        ),
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--images-dir', '-i', type=DirPath(), required=True,
        help='The folder where input NIfTI images files are located (required).'
    )
    parser.add_argument(
        '--segs-dir', '-s', type=DirPath(), required=True,
        help='The folder where input NIfTI segmentation files are located (required).'
    )
    parser.add_argument(
        '--output-images-dir', '-o', type=DirPath(True), required=True,
        help='The folder where output combined JPG images will be saved (required).'
    )
    parser.add_argument(
        '--output-segs-dir', '-g', type=DirPath(True), required=True,
        help='The folder where output combined JPG images will be saved (required).'
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
        '--from-bottom', action="store_true", default=False,
        help='Crop at the lowest voxel of T12-L1 IVD. Default: Crop in the most anteior voxel the lowest vertebrae in the image'
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
    from_bottom = args.from_bottom
    max_workers = args.max_workers
    verbose = args.verbose
    
    # Print the argument values if verbose is enabled
    if verbose:
        print(textwrap.dedent(f''' 
            Running with the following arguments:
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
            from_bottom = "{from_bottom}"
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
    partial_generate_croped_images = partial(
        generate_croped_images,
        images_path=images_path,
        segs_path=segs_path,
        output_images_path=output_images_path,
        output_segs_path=output_segs_path,
        image_suffix=image_suffix,
        seg_suffix=seg_suffix,
        output_image_suffix=output_image_suffix,
        output_seg_suffix=output_seg_suffix,
        from_bottom=from_bottom,
    )

    with mp.Pool() as pool:
        process_map(partial_generate_croped_images, images_path_list, max_workers=max_workers)


def generate_croped_images(
        image_path,
        images_path,
        segs_path,
        output_images_path,
        output_segs_path,
        image_suffix,
        seg_suffix,
        output_image_suffix,
        output_seg_suffix,
        from_bottom,
    ):
    
    seg_path = segs_path / image_path.relative_to(images_path).parent /  image_path.name.replace(f'{image_suffix}.nii.gz', f'{seg_suffix}.nii.gz')
    
    if not seg_path.is_file():
        print(f'Segmentation file not found: {seg_path}')
        return

    output_image_path = output_images_path / image_path.relative_to(images_path).parent / image_path.name.replace(f'{image_suffix}.nii.gz', f'{output_image_suffix}.nii.gz')
    output_seg_path = output_segs_path / image_path.relative_to(images_path).parent / seg_path.name.replace(f'{seg_suffix}.nii.gz', f'{output_seg_suffix}.nii.gz')

    temp_path = Path(tempfile.mkdtemp())
    output_image_path.parent.mkdir(parents=True, exist_ok=True)
    output_seg_path.parent.mkdir(parents=True, exist_ok=True)

    try:

        # Copy files to tmp in canonial orientation
        run_command(f'sct_image -i {image_path} -setorient LPI -o {temp_path}/img.nii.gz')
        run_command(f'sct_image -i {seg_path} -setorient LPI -o {temp_path}/seg.nii.gz')

        # Ensure img and seg in the same space
        run_command(f'sct_register_multimodal -i {temp_path}/seg.nii.gz -d {temp_path}/img.nii.gz -identity 1 -x nn -o {temp_path}/seg.nii.gz')
        
        seg = nib.load(f'{temp_path}/seg.nii.gz')
        seg_data = seg.get_fdata().astype(np.uint8)
        
        # Create an array of z indices
        z_indices = np.tile(np.arange(seg_data.shape[2]), (seg_data.shape[0], seg_data.shape[1], 1))
        # Create an array of y indices
        y_indices = np.broadcast_to(np.arange(seg_data.shape[1])[..., np.newaxis], seg_data.shape)

        if not from_bottom:
            # Cut at the z of the most anteior voxel the lowest vertebrae in the image
            last_vert = seg_data[(18 <= seg_data) & (seg_data <= 41)].min()
            zmin = -1
            # Get the z - loop to fine the most inferior possible z with spinal canal (label 201). 
            while zmin == -1 or 201 not in seg_data[..., zmin]:
                zmin = z_indices[(seg_data == last_vert) & (y_indices == y_indices[seg_data == last_vert].max())].min()
                last_vert += 1
            run_command(f'sct_crop_image -i {temp_path}/img.nii.gz -zmin {zmin} -o {temp_path}/img.nii.gz')
            run_command(f'sct_crop_image -i {temp_path}/seg.nii.gz -zmin {zmin} -o {temp_path}/seg.nii.gz')
        elif 207 in seg_data:
            # Cut at the lowest voxel of T12-L1 IVD
            zmax = z_indices[seg_data == 207].min()
            run_command(f'sct_crop_image -i {temp_path}/img.nii.gz -zmax {zmax} -o {temp_path}/img.nii.gz')
            run_command(f'sct_crop_image -i {temp_path}/seg.nii.gz -zmax {zmax} -o {temp_path}/seg.nii.gz')

        # Copy files from tmp to output destination
        shutil.copy(str(temp_path / 'img.nii.gz'), str(output_image_path))
        shutil.copy(str(temp_path / 'seg.nii.gz'), str(output_seg_path))
    finally:
        shutil.rmtree(temp_path)

def run_command(command):
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    # print(result.stdout)
    # print(result.stderr)

if __name__ == '__main__':
    main()