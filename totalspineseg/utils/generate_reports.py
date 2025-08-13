import sys, os, argparse, textwrap

def main():

    # Description and arguments
    parser = argparse.ArgumentParser(
        description=' '.join(f'''
            This script processes the outputs of totalspineseg/utils/measure_seg.py to generate a reports.
        '''.split()),
        epilog=textwrap.dedent('''
            Examples:
            totalspineseg_generate_reports -m metrics_folder -o reports
        '''),
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--metrics-dir', '-m', type=Path, required=True,
        help='The folder where computed metrics are located (required).'
    )
    parser.add_argument(
        '--ofolder', '-o', type=Path, required=True,
        help='The folder where reports will be saved (required).'
    )
    parser.add_argument(
        '--quiet', '-q', action="store_true", default=False,
        help='Do not display inputs and progress bar, defaults to false (display).'
    )

    # Parse the command-line arguments
    args = parser.parse_args()

    # Get the command-line argument values
    metrics_path = args.metrics_dir
    ofolder = args.ofolder
    quiet = args.quiet

    # Print the argument values if not quiet
    if not quiet:
        print(textwrap.dedent(f'''
            Running {Path(__file__).stem} with the following params:
            metrics_path = "{metrics_path}"
            ofolder = "{ofolder}"
            quiet = {quiet}
        '''))

    generate_reports(
        images_path=images_path,
        ofolder_path=ofolder,
        quiet=quiet,
    )

def generate_reports(
        images_path,
        ofolder_path,
        quiet=False
    ):

    return