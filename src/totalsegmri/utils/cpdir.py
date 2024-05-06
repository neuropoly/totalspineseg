import sys, argparse, textwrap, re
import multiprocessing as mp
from pathlib import Path
from tqdm.contrib.concurrent import process_map
from functools import partial
import shutil
import warnings

warnings.filterwarnings("ignore")


def main():

    # Description and arguments
    parser = argparse.ArgumentParser(
        description=textwrap.dedent(f'''
        This script copy files from one folder to another, the destination folder will be created if not exists.'''
        ),
        epilog=textwrap.dedent('''
            Examples:
            cpdir src dst
            cpdir src dst -p "*w_0000.nii.gz" "*w.nii.gz" -f
            cpdir src dst -p "*"
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
        '--pattern', '-p', type=str, nargs='+', default=['**/*'],
        help='The pattern to use for glob (default "**/*").'
    )
    parser.add_argument(
        '--flat', '-f', action="store_true", default=False,
        help='Put files in destination folder directly without keeping the source subfolders, defaults to false (keep source subfolders).'
    )
    parser.add_argument(
        '--replace', '-r', type=lambda x:x.split(':'), nargs='+', default=[],
        help=textwrap.dedent('''
            Replace string in the destination path befor copying using regex. (e.g. -r "_w.nii.gz:_w_0001.nii.gz").
            Notic that the replacment is done on the full path.
        ''')
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
    pattern = args.pattern
    flat = args.flat
    replace = dict(args.replace)
    max_workers = args.max_workers
    verbose = args.verbose

    # Print the argument values if verbose is enabled
    if verbose:
        print(textwrap.dedent(f''' 
            Running {Path(__file__).stem} with the following params:
            src = "{src}"
            dst = "{dst}"
            pattern = {pattern}
            flat = {flat}
            replace = {replace}
            max_workers = {max_workers}
            verbose = {verbose}
        '''))

    # Process the files
    files_path_list = [_ for __ in [list(src.glob(p)) for p in pattern] for _ in __ if _.is_file()]

    # Create a partially-applied function with the extra arguments
    partial_cpdir = partial(
        cpdir,
        src=src,
        dst=dst,
        flat=flat,
        replace=replace,
    )

    with mp.Pool() as pool:
        process_map(partial_cpdir, files_path_list, max_workers=max_workers)


def cpdir(file_path, src, dst, flat, replace):

    if flat:
        dst_file_path = dst / file_path.name
    else:
        dst_file_path = dst / file_path.relative_to(src)
    # Replace all regex patterns in replace list
    for k, v in replace.items():
        dst_file_path = Path(re.sub(k, v, str(dst_file_path)))
    # Make sure dst directory exists and copy
    dst_file_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(file_path, dst_file_path)


if __name__ == '__main__':
    main()
