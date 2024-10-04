import os
import argparse
from pathlib import Path
import subprocess
from importlib.metadata import metadata
from urllib.request import urlretrieve
from tqdm import tqdm

def main():
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Download nnunet weights from totalspineseg repository.')
    parser.add_argument('--nnunet-dataset', required=True, type=str, help='Name of the nnUNet dataset present in the pyproject.toml (Required)')
    parser.add_argument('--zip-url', required=True, type=str, help='URL of the weights contained inside the pyproject.toml (Required)')
    parser.add_argument('--results-folder', required=True, type=Path, help='Results folder where the weights will be stored (Required)')
    parser.add_argument('--exports-folder', required=True, type=Path, help='Exports folder where the zipped weights will be dowloaded (Required)')
    parser.add_argument('--quiet', '-q', action="store_true", default=False, help='Do not display inputs and progress bar, defaults to false (Default=False).')
    args = parser.parse_args()

    # Datasets data
    nnunet_dataset = args.nnunet_dataset
    zip_url=args.zip_url
    results_folder = args.results_folder
    exports_folder = args.exports_folder
    quiet = args.quiet

    # Install nnUNet weight
    install_weight(
        nnunet_dataset=nnunet_dataset,
        zip_url=zip_url,
        results_folder=results_folder,
        exports_folder=exports_folder,
        quiet=quiet
    )

def install_weight(
    nnunet_dataset,
    zip_url,
    results_folder,
    exports_folder,
    quiet
):

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
        os.environ['nnUNet_results'] = str(results_folder)
        subprocess.run(['nnUNetv2_install_pretrained_model_from_zip', str(zip_file)])


if __name__ == '__main__':
    main()