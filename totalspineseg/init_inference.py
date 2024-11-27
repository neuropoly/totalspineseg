import argparse, textwrap, os
from pathlib import Path
import importlib.resources
from totalspineseg import models, install_weights
from totalspineseg.utils.utils import ZIP_URLS

# This is just to silence nnUNet warnings. These variables should have no purpose/effect.
# There are sadly no other workarounds at the moment, see:
# https://github.com/MIC-DKFZ/nnUNet/blob/227d68e77f00ec8792405bc1c62a88ddca714697/nnunetv2/paths.py#L21
os.environ['nnUNet_raw'] = "./nnUNet_raw"
os.environ['nnUNet_preprocessed'] = "./nnUNet_preprocessed"
os.environ['nnUNet_results'] = "./nnUNet_results"

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
        '--store-export', action="store_false",
        help='Store exported zip file, default to true.'
    )
    parser.add_argument(
        '--quiet', '-q', action="store_true",
        help='Do not display inputs and progress bar, defaults to false (display).'
    )

    # Parse the command-line arguments
    args = parser.parse_args()

    # Get the command-line argument values
    store_export = args.store_export
    quiet = args.quiet

    # Init data_path
    if 'TOTALSPINESEG_DATA' in os.environ:
        data_path = Path(os.environ.get('TOTALSPINESEG_DATA', ''))
    else:
        data_path = importlib.resources.files(models)

    # Initialize inference
    init_inference(
        data_path=data_path,
        dict_urls=ZIP_URLS,
        store_export=store_export,
        quiet=quiet
        )


def init_inference(
        data_path,
        dict_urls=ZIP_URLS,
        store_export=True,
        quiet=False
    ):
    '''
    Function used to download and install nnUNetV2 weights

    Parameters
    ----------
    data_path : pathlib.Path or string
        Folder path containing the network weights.
    dict_urls : dictionary
        Url dictionary containing all the weights that need to be downloaded.
    quiet : bool
        If True, will reduce the amount of displayed information

    Returns
    -------
    list of string
        List of output folders.
    '''
    # Convert data_path to Path like object
    if isinstance(data_path, str):
        data_path = Path(data_path)
    else:
        if not isinstance(data_path, Path):
            raise ValueError('data_path should be a Path object from pathlib or a string')
    
    # Set nnUNet paths
    nnUNet_results = data_path / 'nnUNet' / 'results'
    nnUNet_exports = data_path / 'nnUNet' / 'exports'

    # If not both steps models are installed, use the release subfolder
    if not any([(nnUNet_results / dataset).is_dir() for dataset in dict_urls.keys()]):
        # TODO Think of better way to get the release
        weights_release = list(dict_urls.values())[0].split('/')[-2]
        nnUNet_results = nnUNet_results / weights_release

    # Installing the pretrained models if not already installed
    for dataset, zip_url in dict_urls.items():
        install_weights(
            nnunet_dataset=dataset,
            zip_url=zip_url,
            results_folder=nnUNet_results,
            exports_folder=nnUNet_exports,
            store_export=store_export,
            quiet=quiet
        )


if __name__=='__main__':
    main()