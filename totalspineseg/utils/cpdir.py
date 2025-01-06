import sys, argparse, textwrap, re
import multiprocessing as mp
import nibabel as nib
from pathlib import Path
from tqdm.contrib.concurrent import process_map
from functools import partial
import shutil
import warnings

warnings.filterwarnings("ignore")


def main():

    # Description and arguments
    parser = argparse.ArgumentParser(
        description=' '.join(f'''
            This script copy files from one folder to another, the destination folder will be created if not exists.
        '''.split()),
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
        help='The destination folder, will be created if not exist (required).'
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
        help=' '.join(f'''
            Replace string in the destination path before copying using regex. (e.g. -r "_w.nii.gz:_w_0001.nii.gz").
            Notice that the replacement is done on the full path.
        '''.split())
    )
    parser.add_argument(
        '--compress', '-c', type=bool, default=False,
        help=' '.join(f'''
            Compress files into gzip in destination folder (default is false).
        '''.split())
    )
    parser.add_argument(
        '--tries', '-n', type=int, default=10,
        help='Number of times to try copying the file in case of failure (default is 10).'
    )
    parser.add_argument(
        '--overwrite', '-r', action="store_true", default=False,
        help='Overwrite existing output files, defaults to false (Do not overwrite).'
    )
    parser.add_argument(
        '--max-workers', '-w', type=int, default=mp.cpu_count(),
        help='Max workers to run in parallel processes, defaults to multiprocessing.cpu_count().'
    )
    parser.add_argument(
        '--quiet', '-q', action="store_true", default=False,
        help='Do not display inputs and progress bar, defaults to false (display).'
    )

    # Parse the command-line arguments
    args = parser.parse_args()

    # Get the command-line argument values
    src = args.src
    dst = args.dst
    pattern = args.pattern
    flat = args.flat
    replace = dict(args.replace)
    compress = args.compress
    num_tries = args.tries
    overwrite = args.overwrite
    max_workers = args.max_workers
    quiet = args.quiet

    # Print the argument values if not quiet
    if not quiet:
        print(textwrap.dedent(f'''
            Running {Path(__file__).stem} with the following params:
            src = "{src}"
            dst = "{dst}"
            pattern = {pattern}
            flat = {flat}
            replace = {replace}
            compress = {compress}
            tries = {num_tries}
            overwrite = {overwrite}
            max_workers = {max_workers}
            quiet = {quiet}
        '''))

    cpdir_mp(
        src_path=src,
        dst_path=dst,
        pattern=pattern,
        flat=flat,
        replace=replace,
        compress=compress,
        num_tries=num_tries,
        overwrite=overwrite,
        max_workers=max_workers,
        quiet=quiet,
    )

def cpdir_mp(
        src_path,
        dst_path,
        pattern=['**/*'],
        flat=False,
        replace={},
        compress=False,
        num_tries=1,
        overwrite=False,
        max_workers=mp.cpu_count(),
        quiet=False,
    ):
    '''
    Wrapper function to handle multiprocessing.
    '''
    src_path = Path(src_path)
    dst_path = Path(dst_path)

    # Process the files
    src_path_list = [_ for __ in [list(src_path.glob(p)) for p in pattern] for _ in __ if _.is_file()]

    if flat:
        dst_path_list = [dst_path / _.name.replace('.nii', '.nii.gz') if _.name.endswith('.nii') and compress else dst_path / _.name for _ in src_path_list]
    else:
        dst_path_list = [dst_path / _.relative_to(src_path).replace('.nii', '.nii.gz') if _.name.endswith('.nii') and compress else dst_path / _.relative_to(src_path) for _ in src_path_list]

    # Replace all regex patterns in replace list
    for p, r in replace.items():
        dst_path_list = [Path(re.sub(p, r, str(_))) for _ in dst_path_list]

    process_map(
        partial(
            _cpdir,
            num_tries=num_tries,
            overwrite=overwrite,
        ),
        src_path_list,
        dst_path_list,
        max_workers=max_workers,
        chunksize=1,
        disable=quiet,
    )

def _cpdir(
        src_path,
        dst_path,
        num_tries=1,
        overwrite=False,
    ):
    '''
    Copy file from src to dst.
    '''
    # If the output already exists and we are not overriding it, return
    if not overwrite and dst_path.exists():
        return

    # Make sure dst directory exists
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    # Try copying up to num_tries times
    for attempt in range(num_tries):
        try:
            if "".join(dst_path.suffixes) != "".join(src_path.suffixes):
                # Compressing the destination file                    
                src_img = nib.load(src_path)
                nib.save(src_img, dst_path)
                assert dst_path.exists(), 'Error during gzip compression'
            else:
                # Copy the file
                shutil.copy2(src_path, dst_path)
                # Check if destination file exists before accessing its size
                if dst_path.exists() and src_path.stat().st_size == dst_path.stat().st_size:
                    # Successful copy
                    break
                else:
                    # File sizes do not match or dst_path does not exist
                    if attempt == num_tries - 1:
                        print(f"Warning: File sizes do not match after copying {src_path} to {dst_path}")
                    else:
                        # Delete the destination file before retrying
                        dst_path.unlink(missing_ok=True)
        except OSError as e:
            # Catch OSError
            if attempt == num_tries - 1:
                print(f"Error copying {src_path} to {dst_path}: {e}")
            else:
                # Delete the destination file before retrying
                dst_path.unlink(missing_ok=True)

if __name__ == '__main__':
    main()
