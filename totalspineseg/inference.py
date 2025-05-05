import os, argparse, warnings, textwrap, torch, psutil, shutil
from fnmatch import fnmatch
import multiprocessing as mp
import nibabel as nib
from pathlib import Path
import importlib.resources
from tqdm import tqdm
from totalspineseg import *
from totalspineseg.init_inference import init_inference

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
            totalspineseg input.nii output_folder
            totalspineseg input.nii output_folder --loc output_folder_loc/step2_output/localizer.nii
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
        help='The input folder containing the .nii.gz (or .nii) images to run the model on, or a single .nii.gz (or .nii) image.'
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
            Folder containing localizers segmentations or a single .nii.gz (or .nii) localizer segmentation to use for detecting first vertebrae and disc if C1 and C2-C3 disc or the Sacrum and L5-S disc not found in the image, Optional.
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
        '--keep-only', '-k', type=str, nargs='+', default=[''],
        help='Specify only the output folders you want, the rest will be deleted, by default everything is kept.'
    )
    parser.add_argument(
        '--data-dir', '-d', type=Path, default=None,
        help=' '.join(f'''
            The path to store the nnUNet data.
        '''.split())
    )
    parser.add_argument(
        '--no-stalling', action="store_true", default=False,
        help='Set multiprocessing method to "forkserver" to avoid deadlock issues, default to False.'
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
    keep_only = args.keep_only
    max_workers = args.max_workers
    max_workers_nnunet = min(args.max_workers_nnunet, max_workers)
    device = args.device
    quiet = args.quiet

    # Init data_path
    if not args.data_dir is None:
        data_path = args.data_dir
    elif 'TOTALSPINESEG_DATA' in os.environ:
        data_path = Path(os.environ.get('TOTALSPINESEG_DATA', ''))
    else:
        data_path = importlib.resources.files(models)
    
    # Change multiprocessing method if specified
    if args.no_stalling:
        mp.set_start_method('forkserver', force=True)
    
    # Default release to use
    default_release = list(ZIP_URLS.values())[0].split('/')[-2]

    # Install weights if not present
    init_inference(
        data_path=data_path,
        dict_urls=ZIP_URLS,
        quiet=quiet
        )
    
    # Run inference
    inference(
        input_path=input_path,
        output_path=output_path,
        data_path=data_path,
        default_release=default_release,
        output_iso=output_iso,
        loc_path=loc_path,
        suffix=suffix,
        loc_suffix=loc_suffix,
        step1_only=step1_only,
        keep_only=keep_only,
        max_workers=max_workers,
        max_workers_nnunet=max_workers_nnunet,
        device=device,
        quiet=quiet
    )


def inference(
        input_path,
        output_path,
        data_path,
        default_release,
        output_iso=False,
        loc_path=None,
        suffix=[''],
        loc_suffix='',
        step1_only=False,
        keep_only=[''],
        max_workers=os.cpu_count(),
        max_workers_nnunet=int(max(min(os.cpu_count(), psutil.virtual_memory().total / 2**30 // 8), 1)),
        device='cuda',
        quiet=False
    ):
    '''
    Inference function

    Parameters
    ----------
    input_path : pathlib.Path or string
        The input folder path containing the niftii images.
    output_path : pathlib.Path or string
        The output folder path that will contain the predictions.
    data_path : pathlib.Path or string
        Folder path containing the network weights.
    default_release : string
        Default release used for inference.
    output_iso : bool
        If False, output predictions will be resampled to the original space.
    loc_path : None or pathlib.Path/string
        The localizer folder path containing the niftii predictions of the localizer.
    suffix : string
        Suffix to use for the input images
    loc_suffix : string
        Suffix to use for the localizer images
    step1_only : bool
        If True only the prediction of the first model will be computed.
    keep_only : list of string
        If not empty, only the folders listed will be kept and some functions won't be computed.
    max_workers : int
        Max worker to run in parallel proccess, defaults to numer of available cores
    max_workers_nnunet : int
        Max worker to run in parallel proccess for nnUNet
    device : 'cuda' or 'cpu'
        Device to run the nnUNet model on
    quiet : bool
        If True, will reduce the amount of displayed information

    Returns
    -------
    list of string
        List of output folders.
    '''
    # Convert paths to Path like objects
    if isinstance(input_path, str):
        input_path = Path(input_path)
    else:
        if not isinstance(input_path, Path):
            raise ValueError('input_path should be a Path object from pathlib or a string')

    if isinstance(output_path, str):
        output_path = Path(output_path)
    else:
        if not isinstance(output_path, Path):
            raise ValueError('output_path should be a Path object from pathlib or a string')

    if isinstance(data_path, str):
        data_path = Path(data_path)
    else:
        if not isinstance(data_path, Path):
            raise ValueError('data_path should be a Path object from pathlib or a string')
    
    # Check if the data folder exists
    if not data_path.exists():
        raise FileNotFoundError(f"The totalspineseg data folder does not exist at {data_path}.")

    # Datasets data
    step1_dataset = 'Dataset101_TotalSpineSeg_step1'
    step2_dataset = 'Dataset102_TotalSpineSeg_step2'

    fold = 0

    # Set nnUNet results path
    nnUNet_results = data_path / 'nnUNet' / 'results'

    # If not both steps models are installed, use the default release subfolder
    if not (nnUNet_results / step1_dataset).is_dir() or not (nnUNet_results / step2_dataset).is_dir():
        nnUNet_results = nnUNet_results / default_release
        # Check if weights are available
        if not (nnUNet_results / step1_dataset).is_dir() or not (nnUNet_results / step2_dataset).is_dir():
            raise FileNotFoundError('Model weights are missing.')

    # Load device
    if isinstance(device, str):
        assert device in ['cpu', 'cuda', 'mps'], f'-device must be either cpu, mps or cuda. Other devices are not tested/supported. Got: {device}.'
        if device == 'cpu':
            # let's allow torch to use hella threads
            import multiprocessing
            torch.set_num_threads(multiprocessing.cpu_count())
            device = torch.device('cpu')
        elif device == 'cuda':
            # multithreading in torch doesn't help nnU-Net if run on GPU
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
            device = torch.device('cuda')
        else:
            device = torch.device('mps')
    else:
        assert isinstance(device, torch.device)

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
            keep_only = {keep_only}
            data_dir = "{data_path}"
            max_workers = {max_workers}
            max_workers_nnunet = {max_workers_nnunet}
            device = "{device.type}"
        '''))

    if not quiet: print('\n' 'Making input dir with _0000 suffix:')
    if not input_path.is_dir():
        # If the input is a single file, copy it to the input_raw folder
        (output_path / 'input_raw').mkdir(parents=True, exist_ok=True)

        # Check suffixes
        if input_path.name.endswith(".nii.gz"):
            # Copy file
            dst_path = output_path / 'input_raw' / input_path.name.replace('.nii.gz', '_0000.nii.gz')
            shutil.copy(input_path, dst_path)
        elif input_path.suffix == ".nii":
            # Compress file                    
            src_img = nib.load(input_path)
            dst_path = output_path / 'input_raw' / input_path.name.replace('.nii', '_0000.nii.gz')
            nib.save(src_img, dst_path)
        else:
            raise ValueError(f"Unknown file type: {''.join(input_path.suffixes)}, please use niftii files")
    else:
        # If the input is a folder, copy the files to the input_raw folder
        cpdir_mp(
            input_path,
            output_path / 'input_raw',
            pattern=sum([[f'*{s}.nii.gz', f'sub-*/anat/*{s}.nii.gz', f'*{s}.nii', f'sub-*/anat/*{s}.nii'] for s in suffix], []),
            flat=True,
            replace={'.nii.gz': '_0000.nii.gz'},
            compress=True,
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
        locs = list(loc_path.glob(f'*{loc_suffix}.nii.gz')) + list(loc_path.glob(f'*{loc_suffix}.nii')) \
        + list(loc_path.glob(f'sub-*/anat/*{loc_suffix}.nii.gz')) + list(loc_path.glob(f'sub-*/anat/*{loc_suffix}.nii'))

        # Copy the localizers to the output folder
        images = list((output_path / 'input').glob('*_0000.nii.gz'))
        for image in tqdm(images, disable=quiet):
            if '.nii' in loc_path.suffixes:
                # If the localizers are in a single file, copy it to the localizers folder
                loc = loc_path
            else:
                # If the localizers are in a folder, find the matching localizer for the image
                image_suffix = next((_ for _ in suffix if fnmatch(image.name, f'*{_}_0000.nii.gz')), '')
                loc = next((_ for _ in locs if fnmatch(image.name, _.name.replace(f'{loc_suffix}.nii', f'{image_suffix}_0000.nii'))), None)
            if loc:
                dst_loc = output_path / 'localizers' / image.name.replace('_0000.nii.gz', '.nii.gz')
                if "".join(loc.suffixes) == ".nii":
                    # Compress loc                    
                    src_loc = nib.load(loc)
                    nib.save(src_loc, dst_loc)
                else:
                    # Copy loc
                    shutil.copy(loc, dst_loc)
        
        if not keep_only[0] or 'preview' in keep_only:
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
        if not keep_only[0] or 'preview' in keep_only:
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

    if not quiet: print('\n' 'Preprocessing images:')
    average4d_mp(
        output_path / 'input',
        output_path / 'input',
        image_suffix='',
        output_image_suffix='',
        overwrite=True,
        max_workers=max_workers,
        quiet=quiet,
    )

    if not quiet: print('\n' 'Reorienting images to LPI(-):')
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

    if not keep_only[0] or 'preview' in keep_only:
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
    checkpoint = 'checkpoint_final.pth' if (nnUNet_results / step1_dataset / f'{nnUNetTrainer}__{nnUNetPlans}__{configuration}' / f'fold_{fold}' / 'checkpoint_final.pth').is_file() else 'checkpoint_best.pth'

    # Construct step 1 model folder
    model_folder_step1 = nnUNet_results / step1_dataset / f'{nnUNetTrainer}__{nnUNetPlans}__{configuration}'
    
    if not quiet: print('\n' 'Running step 1 model:')
    predict_nnunet(
        model_folder=model_folder_step1,
        images_dir=output_path / 'input',
        output_dir=output_path / 'step1_raw',
        folds = str(fold),
        save_probabilities = True,
        checkpoint = checkpoint,
        npp = max_workers_nnunet,
        nps = max_workers_nnunet,
        device = device
    )

    # Remove unnecessary files from output folder
    (output_path / 'step1_raw' / 'dataset.json').unlink(missing_ok=True)
    (output_path / 'step1_raw' / 'plans.json').unlink(missing_ok=True)
    (output_path / 'step1_raw' / 'predict_from_raw_data_args.json').unlink(missing_ok=True)
    for f in (output_path / 'step1_raw').glob('*.pkl'):
        f.unlink(missing_ok=True)

    if not keep_only[0] or 'preview' in keep_only:
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

    if not quiet: print('\n' 'Filling spinal canal label to include all non cord spinal canal:')
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

    if not keep_only[0] or 'preview' in keep_only:
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

    if not keep_only[0] or 'preview' in keep_only:
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

    if not keep_only[0] or 'step1_canal' in keep_only:
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

    if not keep_only[0] or 'step1_cord' in keep_only:
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

    if not keep_only[0] or 'step1_levels' in keep_only:
        if not quiet: print('\n' 'Extracting the levels of the vertebrae and IVDs from the step 1 model output:')
        extract_levels_mp(
            output_path / 'step1_output',
            output_path / 'step1_levels',
            canal_labels=[1, 2],
            disc_labels=list(range(63, 68)) + list(range(71, 83)) + list(range(91, 96)) + [100],
            c1_label=11,
            c2_label=50,
            overwrite=True,
            max_workers=max_workers,
            quiet=quiet,
        )

    if not keep_only[0] or 'preview' in keep_only:
        if not quiet: print('\n' 'Generating preview images for the step 1 levels:')
        preview_jpg_mp(
            output_path / 'input',
            output_path / 'preview',
            segs_path=output_path / 'step1_levels',
            output_suffix='_step1_levels_tags',
            levels=True,
            overwrite=True,
            max_workers=max_workers,
            quiet=quiet,
            label_texts_right={i: f'{i}' for i in range(1, 31)},
        )

    if not step1_only:
        if not keep_only[0] or 'step2_output' in keep_only:
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

            if not keep_only[0] or 'preview' in keep_only:
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
            checkpoint = 'checkpoint_final.pth' if (nnUNet_results / step2_dataset / f'{nnUNetTrainer}__{nnUNetPlans}__{configuration}' / f'fold_{fold}' / 'checkpoint_final.pth').is_file() else 'checkpoint_best.pth'

            # Construct step 2 model folder
            model_folder_step2 = nnUNet_results / step2_dataset / f'{nnUNetTrainer}__{nnUNetPlans}__{configuration}'

            if not quiet: print('\n' 'Running step 2 model:')
            predict_nnunet(
                model_folder=model_folder_step2,
                images_dir=output_path / 'step2_input',
                output_dir=output_path / 'step2_raw',
                folds = str(fold),
                checkpoint = checkpoint,
                npp = max_workers_nnunet,
                nps = max_workers_nnunet,
                device = device
            )

            # Remove unnecessary files from output folder
            (output_path / 'step2_raw' / 'dataset.json').unlink(missing_ok=True)
            (output_path / 'step2_raw' / 'plans.json').unlink(missing_ok=True)
            (output_path / 'step2_raw' / 'predict_from_raw_data_args.json').unlink(missing_ok=True)

            # Remove the raw files from step 2 to save space
            for f in (output_path / 'step2_input').glob('*_0000.nii.gz'):
                f.unlink(missing_ok=True)

            if not keep_only[0] or 'preview' in keep_only:
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

            if not quiet: print('\n' 'Filling spinal canal label to include all non cord spinal canal:')
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

            if not keep_only[0] or 'preview' in keep_only:
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

            if not keep_only[0] or 'preview' in keep_only:
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

    # Keep and resample output data
    folder_list = [f for f in os.listdir(str(output_path)) if not f.startswith('input')]
    folder_dict = {
        'step1_output':{'interpolation':'nearest', 'description':'Step 1 output'},
        'step1_cord':{'interpolation':'linear', 'description':'Spinal cord soft segmentations'},
        'step1_canal':{'interpolation':'linear', 'description':'Spinal canal soft segmentations'},
        'step1_levels':{'interpolation':'label', 'description':'Single voxels at the posterior tip of discs'},
        'step2_output':{'interpolation':'nearest', 'description':'Segmentation and labeling of the vertebrae, discs, spinal cord and spinal canal'}
    }
    for folder in folder_list:
        if not keep_only[0] or folder in keep_only:
            if folder in folder_dict.keys():
                if not output_iso:
                    if not quiet: print('\n' f'Resampling {folder} to the input space:')
                    transform_seg2image_mp(
                        output_path / 'input_raw',
                        output_path / folder,
                        output_path / folder,
                        interpolation=folder_dict[folder]['interpolation'],
                        overwrite=True,
                        max_workers=max_workers,
                        quiet=quiet,
                    )
                    if not quiet: print(f'\n{folder_dict[folder]["description"]}:')
                    if not quiet: print(f'{str(output_path)}/{folder}')

        else:
            # Remove the folder
            if not quiet: print('\n' f'Removing {folder}')
            shutil.rmtree(output_path / folder, ignore_errors=True)

    # Remove the input_raw folder
    shutil.rmtree(output_path / 'input_raw', ignore_errors=True)

    # Remove the input folder
    if not output_iso:
        shutil.rmtree(output_path / 'input', ignore_errors=True)
    
    # Return list of output paths
    return [str(output_path / folder) for folder in os.listdir(str(output_path))]

if __name__ == '__main__':
    main()
