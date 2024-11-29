import os, argparse, textwrap
import zipfile
from pathlib import Path
from urllib.request import urlretrieve
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")


def main():

    # Description and arguments
    parser = argparse.ArgumentParser(
        description=' '.join(f'''
            This script download nnunet weights zip file from url into exports folder and install it into results folder.
        '''.split()),
        epilog=textwrap.dedent('''
            Examples:
            install_weights --nnunet-dataset Dataset101_TotalSpineSeg_step1 --zip-url https://github.com/neuropoly/totalspineseg/releases/download/r20241005/Dataset101_TotalSpineSeg_step1_r20241005.zip --results-folder results --exports-folder exports
        '''),
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--nnunet-dataset', type=str, required=True,
        help='Name of the nnUNet dataset to install (required).'
    )
    parser.add_argument(
        '--zip-url', type=str, required=True,
        help='URL of the zip file to download (required).'
    )
    parser.add_argument(
        '--results-folder', type=Path, required=True,
        help='Results folder where the weights will be stored (Required).'
    )
    parser.add_argument(
        '--exports-folder', type=Path, required=True,
        help='Exports folder where the zipped weights will be dowloaded (Required).'
    )
    parser.add_argument(
        '--store-export', type=bool, default=True,
        help='Store exported zip file, default to true.'
    )
    parser.add_argument(
        '--quiet', '-q', action="store_true", default=False,
        help='Do not display inputs and progress bar, default to false. (display)'
    )

    # Parse the command-line arguments
    args = parser.parse_args()

    # Get the command-line argument values
    nnunet_dataset = args.nnunet_dataset
    zip_url = args.zip_url
    results_folder = args.results_folder
    exports_folder = args.exports_folder
    store_export = args.store_export
    quiet = args.quiet

    # Print the argument values if not quiet
    if not quiet:
        print(textwrap.dedent(f'''
            Running {Path(__file__).stem} with the following params:
            nnunet_dataset = "{nnunet_dataset}"
            zip_url = "{zip_url}"
            results_folder = "{results_folder}"
            exports_folder = "{exports_folder}"
            store_export = "{store_export}"
            quiet = {quiet}
        '''))

    install_weights(
        nnunet_dataset=nnunet_dataset,
        zip_url=zip_url,
        results_folder=results_folder,
        exports_folder=exports_folder,
        store_export=store_export,
        quiet=quiet,
    )

def install_weights(
        nnunet_dataset,
        zip_url,
        results_folder,
        exports_folder,
        store_export=True,
        quiet=False,
    ):
    '''
    Download nnunet weights from url.
    '''
    results_folder = Path(results_folder)
    exports_folder = Path(exports_folder)

    # Create the download and export folder if they do not exist
    results_folder.mkdir(parents=True, exist_ok=True)
    exports_folder.mkdir(parents=True, exist_ok=True)

    # Installing the pretrained models if not already installed
    if not (results_folder / nnunet_dataset).is_dir():
        # Get the zip file name and path
        zip_name = zip_url.split('/')[-1]
        zip_file = exports_folder / zip_name

        # Check if the zip file exists
        if not zip_file.is_file():
            # If the zip file is not found, download it from the releases
            if not quiet: print(f'Downloading the pretrained model from {zip_url}...')
            with tqdm(unit='B', unit_scale=True, miniters=1, unit_divisor=1024, disable=quiet) as pbar:
                urlretrieve(
                    zip_url,
                    zip_file,
                    lambda b, bsize, tsize=None: (pbar.total == tsize or pbar.reset(tsize)) and pbar.update(b * bsize - pbar.n),
                )

        if not zip_file.is_file():
            raise FileNotFoundError(f'Could not download the pretrained model for {nnunet_dataset}.')

        # If the pretrained model is not installed, install it from zip
        if not quiet: print(f'Installing the pretrained model from {zip_file}...')
        # Install the pretrained model from the zip file
        install_model_from_zip_file(str(zip_file), extract_folder=str(results_folder))

        # Remove export
        if not store_export:
            os.remove(str(zip_file))


def install_model_from_zip_file(zip_file: str, extract_folder):
    '''
    Based on https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2/model_sharing/model_import.py
    '''
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(extract_folder)


if __name__ == '__main__':
    main()