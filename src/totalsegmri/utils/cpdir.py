import sys, argparse, textwrap
import multiprocessing as mp
from pathlib import Path
from tqdm.contrib.concurrent import process_map
from functools import partial
import shutil


def main():

    # Description and arguments
    parser = argparse.ArgumentParser(
        description=textwrap.dedent(f'''
        This script copy files from one folder to another, the destination folder will be created if not exists.'''
        ),
        epilog=textwrap.dedent('''
            Examples:
            cpdir src dst
        '''),
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        'src', type=Path,
        help='The source folder (required).'
    )
    parser.add_argument(
        'dst', type=Path,
        help='The destnation folder, will be created if not exist (required).'
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
    src = args.src
    dst = args.dst
    max_workers = args.max_workers
    verbose = args.verbose
    
    # Print the argument values if verbose is enabled
    if verbose:
        print(textwrap.dedent(f''' 
            Running {Path(__file__).stem} with the following params:
            src = "{src}"
            dst = "{dst}"
            max_workers = "{max_workers}"
            verbose = "{verbose}"
        '''))

    # Process the files
    files_path_list = list(src.glob('*'))

    # Create a partially-applied function with the extra arguments
    partial_cpdir = partial(
        cpdir,
        dst=dst,
    )

    with mp.Pool() as pool:
        process_map(partial_cpdir, files_path_list, max_workers=max_workers)
    

def cpdir(file_path, dst):
    
    # Make sure dst directory exists and copy
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(file_path, dst)


if __name__ == '__main__':
    main()
