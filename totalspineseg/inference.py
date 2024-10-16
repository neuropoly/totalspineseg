import os, argparse, warnings, subprocess, textwrap, torch, psutil, shutil
from fnmatch import fnmatch
from pathlib import Path
from importlib.metadata import metadata
from tqdm import tqdm
from totalspineseg import *

warnings.filterwarnings("ignore")

def main():
    # Description and arguments
    parser = argparse.ArgumentParser(
        description=textwrap.dedent('''
            This script runs inference using the trained TotalSpineSeg nnUNet model.
            If not already installed, the script will download the pretrained models from the GitHub releases.
        '''),
        epilog=textwrap.dedent('''
            Examples:
            totalspineseg input.nii.gz output_folder
            totalspineseg input.nii.gz output_folder --loc output_folder_loc/step2_output/localizer.nii.gz
            totalspineseg input_folder output_folder
            totalspineseg input_folder output_folder --step1
            totalspineseg input_folder output_folder --loc output_folder_loc/step2_output
            totalspineseg input_folder output_folder --loc output_folder_loc/step2_output --loc-suffix _loc
            totalspineseg input_folder output_folder --loc output_folder_loc/step2_output --suffix _T1w _T2w --loc-suffix _T2w_loc
        '''),
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        'input', type=Path,
        help='The input folder containing the .nii.gz images to run the model on, or a single .nii.gz image.'
    )
    parser.add_argument(
        'output', type=Path,
        help='The output folder where the model outputs will be stored.'
    )
    parser.add_argument(
        '--iso', action="store_true", default=False,
        help='Use isotropic output as output by the model instead of resampling output to the input, defaults to false.'
    )
    parser.add_argument(
        '--loc', '-l', type=Path, default=None,
        help=' '.join(f'''
            Folder containing localizers segmentations or a single .nii.gz localizer segmentation to use for detecting first vertebrae and disc if C1 and C2-C3 disc or the Sacrum and L5-S disc not found in the image, Optional.
            This is the output of the model applied on localizer images. It can be the output of step 2, or step 1 if you only want to run step 1 (step1 flag).
            The algorithm will use the localizers' segmentations to detect the matching vertebrae and discs. The localizer and the image must be aligned.
            Matching will based on the majority of the voxels of the first vertebra or disc in the localizer, that intersect with image.
            The file names should be in match with the image file names, or you can use the --suffix and --loc-suffix to match the files.
        '''.split())
    )
    parser.add_argument(
        '--suffix', '-s', type=str, nargs='+', default=[''],
        help='Suffix to use for the input images, defaults to "".'
    )
    parser.add_argument(
        '--loc-suffix', '-ls', type=str, default='',
        help='Suffix to use for the localizers labels, defaults to "".'
    )
    parser.add_argument(
        '--step1', action='store_true',
        help='Run only step 1 of the inference process.'
    )
    parser.add_argument(
        '--data-dir', '-d', type=Path, default=Path(os.environ.get('TOTALSPINESEG_DATA', '')), required='TOTALSPINESEG_DATA' not in os.environ,
        help=' '.join(f'''
            The path to store the nnUNet data, defaults to the TOTALSPINESEG_DATA environment variable if set.
            If the TOTALSPINESEG_DATA environment variable is not set, the path must be provided.
        '''.split())
    )
    parser.add_argument(
        '--max-workers', '-w', type=int, default=os.cpu_count(),
        help=f'Max worker to run in parallel proccess, defaults to numer of available cores'
    )
    parser.add_argument(
        '--max-workers-nnunet', type=int, default=int(max(min(os.cpu_count(), psutil.virtual_memory().total / 2**30 // 8), 1)),
        help='Max worker to run in parallel proccess for nnUNet, defaults to min(numer of available cores, Memory in GB / 8).'
    )
    parser.add_argument(
        '--device', type=str, choices=['cuda', 'cpu'], default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to run the nnUNet model on, defaults to "cuda" if available, otherwise "cpu".'
    )
    parser.add_argument(
        '--quiet', '-q', action="store_true", default=False,
        help='Do not display inputs and progress bar, defaults to false (display).'
    )

    # Parse the command-line arguments
    args = parser.parse_args()

    # Get the command-line argument values
    input_path = args.input
    output_path = args.output
    output_iso = args.iso
    loc_path = args.loc
    suffix = args.suffix
    loc_suffix = args.loc_suffix
    step1_only = args.step1
    data_path = args.data_dir
    max_workers = args.max_workers
    max_workers_nnunet = min(args.max_workers_nnunet, max_workers)
    device = args.device
    quiet = args.quiet

    # Check if the data folder exists
    if not data_path.exists():
        raise FileNotFoundError(' '.join(f'''
            The totalspineseg data folder does not exist at {data_path},
            if it is not the correct path, please set the TOTALSPINESEG_DATA environment variable to the correct path,
            or use the --data-dir argument to specify the correct path.
        '''.split()))

    # Datasets data
    step1_dataset = 'Dataset101_TotalSpineSeg_step1'
    step2_dataset = 'Dataset102_TotalSpineSeg_step2'

    # Read urls from 'pyproject.toml'
    step1_zip_url = dict([_.split(', ') for _ in metadata('totalspineseg').get_all('Project-URL')])[step1_dataset]
    step2_zip_url = dict([_.split(', ') for _ in metadata('totalspineseg').get_all('Project-URL')])[step2_dataset]

    fold = 0

    # Set nnUNet paths
    nnUNet_raw = data_path / 'nnUNet' / 'raw'
    nnUNet_preprocessed = data_path / 'nnUNet' / 'preprocessed'
    nnUNet_results = data_path / 'nnUNet' / 'results'
    nnUNet_exports = data_path / 'nnUNet' / 'exports'

    # If not both steps models are installed, use the release subfolder
    if not (nnUNet_results / step1_dataset).is_dir() or not (nnUNet_results / step2_dataset).is_dir():
        # TODO Think of better way to get the release
        release = step1_zip_url.split('/')[-2]
        nnUNet_results = nnUNet_results / release

    # Create the nnUNet directories if they do not exist
    nnUNet_raw.mkdir(parents=True, exist_ok=True)
    nnUNet_preprocessed.mkdir(parents=True, exist_ok=True)
    nnUNet_results.mkdir(parents=True, exist_ok=True)
    nnUNet_exports.mkdir(parents=True, exist_ok=True)

    # Set nnUNet environment variables
    os.environ['nnUNet_def_n_proc'] = str(max_workers_nnunet)
    os.environ['nnUNet_n_proc_DA'] = str(max_workers_nnunet)
    os.environ['nnUNet_raw'] = str(nnUNet_raw)
    os.environ['nnUNet_preprocessed'] = str(nnUNet_preprocessed)
    os.environ['nnUNet_results'] = str(nnUNet_results)

    # Print the argument values if not quiet
    if not quiet:
        print(textwrap.dedent(f'''
            Running TotalSpineSeg with the following parameters:
            input = "{input_path}"
            output = "{output_path}"
            iso = {output_iso}
            loc = "{loc_path}"
            suffix = {suffix}
            loc_suffix = "{loc_suffix}"
            step1_only = {step1_only}
            data_dir = "{data_path}"
            max_workers = {max_workers}
            max_workers_nnunet = {max_workers_nnunet}
            device = "{device}"
        '''))

    # Installing the pretrained models if not already installed
    for dataset, zip_url in [(step1_dataset, step1_zip_url), (step2_dataset, step2_zip_url)]:
        install_weights(
            nnunet_dataset=dataset,
            zip_url=zip_url,
            results_folder=nnUNet_results,
            exports_folder=nnUNet_exports,
            quiet=quiet
        )

    if not quiet: print('\n' 'Making input dir with _0000 suffix:')
    if input_path.name.endswith('.nii.gz'):
        # If the input is a single file, copy it to the input_raw folder
        (output_path / 'input_raw').mkdir(parents=True, exist_ok=True)
        shutil.copy(input_path, output_path / 'input_raw' / input_path.name.replace('.nii.gz', '_0000.nii.gz'))
    else:
        # If the input is a folder, copy the files to the input_raw folder
        cpdir_mp(
            input_path,
            output_path / 'input_raw',
            pattern=sum([[f'*{s}.nii.gz', f'sub-*/anat/*{s}.nii.gz'] for s in suffix], []),
            flat=True,
            replace={'.nii.gz': '_0000.nii.gz'},
            overwrite=True,
            max_workers=max_workers,
            quiet=quiet,
        )

    if not quiet: print('\n' 'Copying the input images to the input folder for processing:')
    cpdir_mp(
        output_path / 'input_raw',
        output_path / 'input',
        overwrite=True,
        max_workers=max_workers,
        quiet=quiet,
    )

    if loc_path is not None:
        if not quiet: print('\n' 'Copying localizers to the output folder:')

        # Create the localizers folder
        (output_path / 'localizers').mkdir(parents=True, exist_ok=True)

        # List all localizers in the localizers folder
        locs = list(loc_path.glob(f'*{loc_suffix}.nii.gz')) + list(loc_path.glob(f'sub-*/anat/*{loc_suffix}.nii.gz'))

        # Copy the localizers to the output folder
        images = list((output_path / 'input').glob('*_0000.nii.gz'))
        for image in tqdm(images, disable=quiet):
            if loc_path.name.endswith('.nii.gz'):
                # If the localizers are in a single file, copy it to the localizers folder
                loc = loc_path
            else:
                # If the localizers are in a folder, find the matching localizer for the image
                image_suffix = next((_ for _ in suffix if fnmatch(image.name, f'*{_}_0000.nii.gz')), '')
                loc = next((_ for _ in locs if fnmatch(image.name, _.name.replace(f'{loc_suffix}.nii.gz', f'{image_suffix}_0000.nii.gz'))), None)
            if loc:
                shutil.copy(loc, output_path / 'localizers' / image.name.replace('_0000.nii.gz', f'.nii.gz'))

        if not quiet: print('\n' 'Generating preview images for the localizers:')
        preview_jpg_mp(
            output_path / 'input',
            output_path / 'preview',
            segs_path=output_path / 'localizers',
            output_suffix='_loc',
            overwrite=True,
            max_workers=max_workers,
            quiet=quiet,
        )

        if not quiet: print('\n' 'Generating preview images for the localizers with tags:')
        preview_jpg_mp(
            output_path / 'input',
            output_path / 'preview',
            segs_path=output_path / 'localizers',
            output_suffix='_loc_tags',
            overwrite=True,
            max_workers=max_workers,
            quiet=quiet,
            label_texts_right={
                11: 'C1', 12: 'C2', 13: 'C3', 14: 'C4', 15: 'C5', 16: 'C6', 17: 'C7',
                21: 'T1', 22: 'T2', 23: 'T3', 24: 'T4', 25: 'T5', 26: 'T6', 27: 'T7',
                28: 'T8', 29: 'T9', 30: 'T10', 31: 'T11', 32: 'T12',
                41: 'L1', 42: 'L2', 43: 'L3', 44: 'L4', 45: 'L5', 46: 'L6',
            },
            label_texts_left={
                50: 'Sacrum', 63: 'C2C3', 64: 'C3C4', 65: 'C4C5', 66: 'C5C6', 67: 'C6C7',
                71: 'C7T1', 72: 'T1T2', 73: 'T2T3', 74: 'T3T4', 75: 'T4T5', 76: 'T5T6', 77: 'T6T7',
                78: 'T7T8', 79: 'T8T9', 80: 'T9T10', 81: 'T10T11', 82: 'T11T12',
                91: 'T12L1', 92: 'L1L2', 93: 'L2L3', 94: 'L3L4', 95: 'L4L5', 96: 'L5L6', 100: 'L5S'
            },
        )

    if not quiet: print('\n' 'Converting 4D images to 3D:')
    average4d_mp(
        output_path / 'input',
        output_path / 'input',
        image_suffix='',
        output_image_suffix='',
        overwrite=True,
        max_workers=max_workers,
        quiet=quiet,
    )

    if not quiet: print('\n' 'Transforming images to canonical space:')
    reorient_canonical_mp(
        output_path / 'input',
        output_path / 'input',
        overwrite=True,
        max_workers=max_workers,
        quiet=quiet,
    )

    if not quiet: print('\n' 'Resampling images to 1x1x1mm:')
    resample_mp(
        output_path / 'input',
        output_path / 'input',
        image_suffix='',
        output_image_suffix='',
        overwrite=True,
        max_workers=max_workers,
        quiet=quiet,
    )

    if not quiet: print('\n' 'Generating preview images for input:')
    preview_jpg_mp(
        output_path / 'input',
        output_path / 'preview',
        output_suffix='_input',
        overwrite=True,
        max_workers=max_workers,
        quiet=quiet,
    )

    # Get the nnUNet parameters from the results folder
    nnUNetTrainer, nnUNetPlans, configuration = next((nnUNet_results / step1_dataset).glob('*/fold_*')).parent.name.split('__')
    # Check if the final checkpoint exists, if not use the latest checkpoint
    checkpoint = 'checkpoint_final.pth' if (nnUNet_results / step1_dataset / f'{nnUNetTrainer}__{nnUNetPlans}__{configuration}' / f'fold_{fold}' / 'checkpoint_final.pth').is_file() else 'checkpoint_latest.pth'

    if not quiet: print('\n' 'Running step 1 model:')
    subprocess.run([
        'nnUNetv2_predict',
            '-d', step1_dataset,
            '-i', str(output_path / 'input'),
            '-o', str(output_path / 'step1_raw'),
            '-f', str(fold),
            '-c', configuration,
            '-p', nnUNetPlans,
            '-tr', nnUNetTrainer,
            '-npp', str(max_workers_nnunet),
            '-nps', str(max_workers_nnunet),
            '-chk', checkpoint,
            '-device', device,
            '--save_probabilities',
    ])

    # Remove unnecessary files from output folder
    (output_path / 'step1_raw' / 'dataset.json').unlink(missing_ok=True)
    (output_path / 'step1_raw' / 'plans.json').unlink(missing_ok=True)
    (output_path / 'step1_raw' / 'predict_from_raw_data_args.json').unlink(missing_ok=True)
    for f in (output_path / 'step1_raw').glob('*.pkl'):
        f.unlink(missing_ok=True)

    if not quiet: print('\n' 'Generating preview images for step 1:')
    preview_jpg_mp(
        output_path / 'input',
        output_path / 'preview',
        segs_path=output_path / 'step1_raw',
        output_suffix='_step1_raw',
        overwrite=True,
        max_workers=max_workers,
        quiet=quiet,
    )

    if not quiet: print('\n' 'Extracting the largest connected component:')
    largest_component_mp(
        output_path / 'step1_raw',
        output_path / 'step1_output',
        binarize=True,
        dilate=5,
        overwrite=True,
        max_workers=max_workers,
        quiet=quiet,
    )

    if not quiet: print('\n' 'Using an iterative algorithm to label IVDs with the definite labels:')
    # Labeling is based on the C2-C3, C7-T1 and L5-S1 IVD labels output by step 1 model.
    if loc_path is None:
        iterative_label_mp(
            output_path / 'step1_output',
            output_path / 'step1_output',
            selected_disc_landmarks=[2, 5, 3, 4],
            disc_labels=[1, 2, 3, 4, 5],
            disc_landmark_labels=[2, 3, 4, 5],
            disc_landmark_output_labels=[63, 71, 91, 100],
            canal_labels=[8],
            canal_output_label=2,
            cord_labels=[9],
            cord_output_label=1,
            sacrum_labels=[6],
            sacrum_output_label=50,
            map_input_dict={7: 11},
            overwrite=True,
            max_workers=max_workers,
            quiet=quiet,
        )
    else:
        iterative_label_mp(
            output_path / 'step1_output',
            output_path / 'step1_output',
            locs_path=output_path / 'localizers',
            selected_disc_landmarks=[2, 5],
            disc_labels=[1, 2, 3, 4, 5],
            disc_landmark_labels=[2, 3, 4, 5],
            disc_landmark_output_labels=[63, 71, 91, 100],
            loc_disc_labels=list(range(63, 101)),
            canal_labels=[8],
            canal_output_label=2,
            cord_labels=[9],
            cord_output_label=1,
            sacrum_labels=[6],
            sacrum_output_label=50,
            map_input_dict={7: 11},
            overwrite=True,
            max_workers=max_workers,
            quiet=quiet,
        )

    if not quiet: print('\n' 'Filling spinal cancal label to include all non cord spinal canal:')
    # This will put the spinal canal label in all the voxels between the canal and the cord.
    fill_canal_mp(
        output_path / 'step1_output',
        output_path / 'step1_output',
        canal_label=2,
        cord_label=1,
        largest_canal=True,
        largest_cord=True,
        overwrite=True,
        max_workers=max_workers,
        quiet=quiet,
    )

    if not quiet: print('\n' 'Transforming labels to input images space:')
    transform_seg2image_mp(
        output_path / 'input',
        output_path / 'step1_output',
        output_path / 'step1_output',
        overwrite=True,
        max_workers=max_workers,
        quiet=quiet,
    )

    if not quiet: print('\n' 'Generating preview images for the step 1 labeled images:')
    preview_jpg_mp(
        output_path / 'input',
        output_path / 'preview',
        segs_path=output_path / 'step1_output',
        output_suffix='_step1_output',
        overwrite=True,
        max_workers=max_workers,
        quiet=quiet,
    )

    if not quiet: print('\n' 'Generating preview images for the step 1 labeled images with tags:')
    preview_jpg_mp(
        output_path / 'input',
        output_path / 'preview',
        segs_path=output_path / 'step1_output',
        output_suffix='_step1_output_tags',
        overwrite=True,
        max_workers=max_workers,
        quiet=quiet,
        label_texts_left={
            63: 'C2C3', 64: 'C3C4', 65: 'C4C5', 66: 'C5C6', 67: 'C6C7',
            71: 'C7T1', 72: 'T1T2', 73: 'T2T3', 74: 'T3T4', 75: 'T4T5', 76: 'T5T6', 77: 'T6T7',
            78: 'T7T8', 79: 'T8T9', 80: 'T9T10', 81: 'T10T11', 82: 'T11T12',
            91: 'T12L1', 92: 'L1L2', 93: 'L2L3', 94: 'L3L4', 95: 'L4L5', 96: 'L5L6', 100: 'L5S'
        },
    )

    if not quiet: print('\n' 'Extracting spinal canal soft segmentation from step 1 model output:')
    extract_soft_mp(
        output_path / 'step1_raw',
        output_path / 'step1_output',
        output_path / 'step1_canal',
        label=8,
        seg_labels=[1, 2],
        dilate=1,
        overwrite=True,
        max_workers=max_workers,
        quiet=quiet,
    )

    if not quiet: print('\n' 'Extracting spinal cord soft segmentation from step 1 model output:')
    extract_soft_mp(
        output_path / 'step1_raw',
        output_path / 'step1_output',
        output_path / 'step1_cord',
        label=9,
        seg_labels=[1],
        dilate=1,
        overwrite=True,
        max_workers=max_workers,
        quiet=quiet,
    )

    if not quiet: print('\n' 'Removing the raw files from step 1 to save space...')
    for f in (output_path / 'step1_raw').glob('*.npz'):
        f.unlink(missing_ok=True)

    if not quiet: print('\n' 'Extracting the levels of the vertebrae and IVDs from the step 1 model output:')
    extract_levels_mp(
        output_path / 'step1_output',
        output_path / 'step1_levels',
        canal_labels=[1, 2],
        disc_labels=list(range(63, 68)) + list(range(71, 83)) + list(range(91, 96)) + [100],
        overwrite=True,
        max_workers=max_workers,
        quiet=quiet,
    )

    if not step1_only:
        if not quiet: print('\n' 'Copying the original images into step 2 input folder:')
        cpdir_mp(
            output_path / 'input',
            output_path / 'step2_input',
            pattern=['*_0000.nii.gz'],
            overwrite=True,
            max_workers=max_workers,
            quiet=quiet,
        )

        if not quiet: print('\n' 'Cropping the images to the bounding box of step 1 segmentation:')
        # This will also delete images without segmentation
        crop_image2seg_mp(
            output_path / 'step2_input',
            output_path / 'step1_output',
            output_path / 'step2_input',
            margin=10,
            overwrite=True,
            max_workers=max_workers,
            quiet=quiet,
        )

        if not quiet: print('\n' 'Transform step 1 segmentation to the cropped images space:')
        transform_seg2image_mp(
            output_path / 'step2_input',
            output_path / 'step1_output',
            output_path / 'step2_input',
            output_seg_suffix='_0001',
            overwrite=True,
            max_workers=max_workers,
            quiet=quiet,
        )

        if not quiet: print('\n' 'Mapping the IVDs labels from the step1 model output to the odd IVDs:')
        # This will also delete labels without odd IVDs
        extract_alternate_mp(
            output_path / 'step2_input',
            output_path / 'step2_input',
            seg_suffix='_0001',
            output_seg_suffix='_0001',
            labels=list(range(63, 101)),
            overwrite=True,
            max_workers=max_workers,
            quiet=quiet,
        )

        if not quiet: print('\n' 'Generating preview images for step 2 input:')
        preview_jpg_mp(
            output_path / 'input',
            output_path / 'preview',
            segs_path=output_path / 'step2_input',
            seg_suffix='_0001',
            output_suffix='_step2_input',
            overwrite=True,
            max_workers=max_workers,
            quiet=quiet,
        )

        # Remove images without the 2'nd channel
        for f in (output_path / 'step2_input').glob('*_0000.nii.gz'):
            if not f.with_name(f.name.replace('_0000.nii.gz', '_0001.nii.gz')).exists():
                f.unlink(missing_ok=True)

        # Get the nnUNet parameters from the results folder
        nnUNetTrainer, nnUNetPlans, configuration = next((nnUNet_results / step2_dataset).glob('*/fold_*')).parent.name.split('__')
        # Check if the final checkpoint exists, if not use the latest checkpoint
        checkpoint = 'checkpoint_final.pth' if (nnUNet_results / step2_dataset / f'{nnUNetTrainer}__{nnUNetPlans}__{configuration}' / f'fold_{fold}' / 'checkpoint_final.pth').is_file() else 'checkpoint_latest.pth'

        if not quiet: print('\n' 'Running step 2 model:')
        subprocess.run([
            'nnUNetv2_predict',
                '-d', step2_dataset,
                '-i', str(output_path / 'step2_input'),
                '-o', str(output_path / 'step2_raw'),
                '-f', str(fold),
                '-c', configuration,
                '-p', nnUNetPlans,
                '-tr', nnUNetTrainer,
                '-npp', str(max_workers_nnunet),
                '-nps', str(max_workers_nnunet),
                '-chk', checkpoint,
                '-device', device
        ])

        # Remove unnecessary files from output folder
        (output_path / 'step2_raw' / 'dataset.json').unlink(missing_ok=True)
        (output_path / 'step2_raw' / 'plans.json').unlink(missing_ok=True)
        (output_path / 'step2_raw' / 'predict_from_raw_data_args.json').unlink(missing_ok=True)

        # Remove the raw files from step 2 to save space
        for f in (output_path / 'step2_input').glob('*_0000.nii.gz'):
            f.unlink(missing_ok=True)

        if not quiet: print('\n' 'Generating preview images for step 2:')
        preview_jpg_mp(
            output_path / 'input',
            output_path / 'preview',
            segs_path=output_path / 'step2_raw',
            output_suffix='_step2_raw',
            overwrite=True,
            max_workers=max_workers,
            quiet=quiet,
        )

        if not quiet: print('\n' 'Extracting the largest connected component:')
        largest_component_mp(
            output_path / 'step2_raw',
            output_path / 'step2_output',
            binarize=True,
            dilate=5,
            overwrite=True,
            max_workers=max_workers,
            quiet=quiet,
        )

        if not quiet: print('\n' 'Using an iterative algorithm to label vertebrae and IVDs:')
        if loc_path is None:
            iterative_label_mp(
                output_path / 'step2_output',
                output_path / 'step2_output',
                selected_disc_landmarks=[2, 5, 3, 4],
                disc_labels=[1, 2, 3, 4, 5],
                disc_landmark_labels=[2, 3, 4, 5],
                disc_landmark_output_labels=[63, 71, 91, 100],
                vertebrae_labels=[7, 8, 9],
                vertebrae_landmark_output_labels=[13, 21, 41, 50],
                vertebrae_extra_labels=[6],
                canal_labels=[10],
                canal_output_label=2,
                cord_labels=[11],
                cord_output_label=1,
                sacrum_labels=[9],
                sacrum_output_label=50,
                overwrite=True,
                max_workers=max_workers,
                quiet=quiet,
            )
        else:
            iterative_label_mp(
                output_path / 'step2_output',
                output_path / 'step2_output',
                locs_path=output_path / 'localizers',
                selected_disc_landmarks=[2, 5],
                disc_labels=[1, 2, 3, 4, 5],
                disc_landmark_labels=[2, 3, 4, 5],
                disc_landmark_output_labels=[63, 71, 91, 100],
                vertebrae_labels=[7, 8, 9],
                vertebrae_landmark_output_labels=[13, 21, 41, 50],
                vertebrae_extra_labels=[6],
                loc_disc_labels=list(range(63, 101)),
                canal_labels=[10],
                canal_output_label=2,
                cord_labels=[11],
                cord_output_label=1,
                sacrum_labels=[9],
                sacrum_output_label=50,
                overwrite=True,
                max_workers=max_workers,
                quiet=quiet,
            )

        if not quiet: print('\n' 'Filling spinal cancal label to include all non cord spinal canal:')
        # This will put the spinal canal label in all the voxels between the canal and the cord.
        fill_canal_mp(
            output_path / 'step2_output',
            output_path / 'step2_output',
            canal_label=2,
            cord_label=1,
            largest_canal=True,
            largest_cord=True,
            overwrite=True,
            max_workers=max_workers,
            quiet=quiet,
        )

        if not quiet: print('\n' 'Transforming labels to input images space:')
        transform_seg2image_mp(
            output_path / 'input',
            output_path / 'step2_output',
            output_path / 'step2_output',
            overwrite=True,
            max_workers=max_workers,
            quiet=quiet,
        )

        if not quiet: print('\n' 'Generating preview images for the final output:')
        preview_jpg_mp(
            output_path / 'input',
            output_path / 'preview',
            segs_path=output_path / 'step2_output',
            output_suffix='_step2_output',
            overwrite=True,
            max_workers=max_workers,
            quiet=quiet,
        )

        if not quiet: print('\n' 'Generating preview images for the final output with tags:')
        preview_jpg_mp(
            output_path / 'input',
            output_path / 'preview',
            segs_path=output_path / 'step2_output',
            output_suffix='_step2_output_tags',
            overwrite=True,
            max_workers=max_workers,
            quiet=quiet,
            label_texts_right={
                11: 'C1', 12: 'C2', 13: 'C3', 14: 'C4', 15: 'C5', 16: 'C6', 17: 'C7',
                21: 'T1', 22: 'T2', 23: 'T3', 24: 'T4', 25: 'T5', 26: 'T6', 27: 'T7',
                28: 'T8', 29: 'T9', 30: 'T10', 31: 'T11', 32: 'T12',
                41: 'L1', 42: 'L2', 43: 'L3', 44: 'L4', 45: 'L5', 46: 'L6',
            },
            label_texts_left={
                50: 'Sacrum', 63: 'C2C3', 64: 'C3C4', 65: 'C4C5', 66: 'C5C6', 67: 'C6C7',
                71: 'C7T1', 72: 'T1T2', 73: 'T2T3', 74: 'T3T4', 75: 'T4T5', 76: 'T5T6', 77: 'T6T7',
                78: 'T7T8', 79: 'T8T9', 80: 'T9T10', 81: 'T10T11', 82: 'T11T12',
                91: 'T12L1', 92: 'L1L2', 93: 'L2L3', 94: 'L3L4', 95: 'L4L5', 96: 'L5L6', 100: 'L5S'
            },
        )

    if not output_iso:
        if not quiet: print('\n' 'Resampling step1_output to the input images space:')
        transform_seg2image_mp(
            output_path / 'input_raw',
            output_path / 'step1_output',
            output_path / 'step1_output',
            overwrite=True,
            max_workers=max_workers,
            quiet=quiet,
        )
        if not quiet: print('\n' 'Resampling step1_cord to the input images space:')
        transform_seg2image_mp(
            output_path / 'input_raw',
            output_path / 'step1_cord',
            output_path / 'step1_cord',
            interpolation = 'linear',
            overwrite=True,
            max_workers=max_workers,
            quiet=quiet,
        )
        if not quiet: print('\n' 'Resampling step1_canal to the input images space:')
        transform_seg2image_mp(
            output_path / 'input_raw',
            output_path / 'step1_canal',
            output_path / 'step1_canal',
            interpolation = 'linear',
            overwrite=True,
            max_workers=max_workers,
            quiet=quiet,
        )
        if not quiet: print('\n' 'Resampling step1_levels to the input images space:')
        transform_seg2image_mp(
            output_path / 'input_raw',
            output_path / 'step1_levels',
            output_path / 'step1_levels',
            interpolation = 'label',
            overwrite=True,
            max_workers=max_workers,
            quiet=quiet,
        )
        if not step1_only:
            if not quiet: print('\n' 'Resampling step2_output to the input images space:')
            transform_seg2image_mp(
                output_path / 'input_raw',
                output_path / 'step2_output',
                output_path / 'step2_output',
                overwrite=True,
                max_workers=max_workers,
                quiet=quiet,
            )

    # Remove the input_raw folder
    shutil.rmtree(output_path / 'input_raw', ignore_errors=True)

if __name__ == '__main__':
    main()