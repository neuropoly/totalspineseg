import argparse, textwrap, os
from pathlib import Path
from importlib.metadata import metadata
import importlib.resources
from totalspineseg import models, install_weights


def main():
    parser = argparse.ArgumentParser(
        description=textwrap.dedent('''
            This script downloads the pretrained models from the GitHub releases.
        '''),
        epilog=textwrap.dedent('''
            Examples:
            totalspineseg_init
            totalspineseg_init --quiet
        '''),
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--quiet', '-q', action="store_true",
        help='Do not display inputs and progress bar, defaults to false (display).'
    )

    # Parse the command-line arguments
    args = parser.parse_args()

    # Get the command-line argument values
    quiet = args.quiet

    # Init data_path
    if 'TOTALSPINESEG_DATA' in os.environ:
        data_path = Path(os.environ.get('TOTALSPINESEG_DATA', ''))
    else:
        data_path = importlib.resources.files(models)

    # Initialize inference
    init_inference(
        data_path=data_path,
        quiet=quiet
        )


def init_inference(data_path, quiet):
    # Datasets data
    step1_dataset = 'Dataset101_TotalSpineSeg_step1'
    step2_dataset = 'Dataset102_TotalSpineSeg_step2'

    # Read urls from 'pyproject.toml'
    step1_zip_url = dict([_.split(', ') for _ in metadata('totalspineseg').get_all('Project-URL')])[step1_dataset]
    step2_zip_url = dict([_.split(', ') for _ in metadata('totalspineseg').get_all('Project-URL')])[step2_dataset]
    
    # Set nnUNet paths
    nnUNet_results = data_path / 'nnUNet' / 'results'
    nnUNet_exports = data_path / 'nnUNet' / 'exports'

    # If not both steps models are installed, use the release subfolder
    if not (nnUNet_results / step1_dataset).is_dir() or not (nnUNet_results / step2_dataset).is_dir():
        # TODO Think of better way to get the release
        weights_release = step1_zip_url.split('/')[-2]
        nnUNet_results = nnUNet_results / weights_release

    # Installing the pretrained models if not already installed
    for dataset, zip_url in [(step1_dataset, step1_zip_url), (step2_dataset, step2_zip_url)]:
        install_weights(
            nnunet_dataset=dataset,
            zip_url=zip_url,
            results_folder=nnUNet_results,
            exports_folder=nnUNet_exports,
            quiet=quiet
        )


if __name__=='__main__':
    main()