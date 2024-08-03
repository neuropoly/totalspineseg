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
            cpdir src dst -p "*w_0000.nii.gz" "*w.nii.gz" -f -t sub-:sub-SINGLE .nii.gz:_0000.nii.gz
            cpdir src dst -p "*" -r
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
        '--replace', '-t', type=lambda x:x.split(':'), nargs='+', default=[],
        help=textwrap.dedent('''
            Replace string in the destination path befor copying using regex. (e.g. -r "_w.nii.gz:_w_0001.nii.gz").
            Notice that the replacment is done on the full path.
        ''')
    )
    parser.add_argument(
        '--override', '-r', action="store_true", default=False,
        help='Override existing output files, defaults to false (Do not override).'
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
    override = args.override
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
            override = {override}
            max_workers = {max_workers}
            verbose = {verbose}
        '''))

    cpdir_mp(
        src_path=src,
        dst_path=dst,
        pattern=pattern,
        flat=flat,
        replace=replace,
        override=override,
        max_workers=max_workers
    )

def cpdir_mp(
        src_path,
        dst_path,
        pattern=['**/*'],
        flat=False,
        replace={},
        override=False,
        max_workers=mp.cpu_count()
    ):
    src_path = Path(src_path)
    dst_path = Path(dst_path)

    # Process the files
    src_path_list = [_ for __ in [list(src_path.glob(p)) for p in pattern] for _ in __ if _.is_file()]

    if flat:
        dst_path_list = [dst_path / _.name for _ in src_path_list]
    else:
        dst_path_list = [dst_path / _.relative_to(src_path) for _ in src_path_list]

    # Replace all regex patterns in replace list
    for p, r in replace.items():
        dst_path_list = [Path(re.sub(p, r, str(_))) for _ in dst_path_list]

    process_map(
        partial(
            _cpdir,
            override=override,
        ),
        src_path_list,
        dst_path_list,
        max_workers=max_workers,
    )

def _cpdir(
        src_path,
        dst_path,
        override=False,
    ):

    # If the output already exists and we are not overriding it, return
    if not override and dst_path.exists():
        return

    # Make sure dst directory exists and copy
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, dst_path)

if __name__ == '__main__':
    main()
