

def main():

    # Description and arguments
    parser = argparse.ArgumentParser(
        description=' '.join(f'''
            This script processes NIfTI (Neuroimaging Informatics Technology Initiative) image and segmentation files.
            It uses MRI scans and totalspineseg segmentations to extract metrics from the canal, the discs and vertebrae.
        '''.split()),
        epilog=textwrap.dedent('''
            Examples:
            totalspineseg_measure_seg -i images -s segmentations -o metrics
        '''),
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--images-dir', '-i', type=Path, required=True,
        help='The folder where input NIfTI images files are located (required).'
    )
    parser.add_argument(
        '--segs-dir', '-s', type=Path, required=True,
        help='The folder where input NIfTI segmentation files are located (required).'
    )
    parser.add_argument(
        '--ofolder', '-o', type=Path, required=True,
        help='The folder where output metrics will be saved (required).'
    )
    parser.add_argument(
        '--prefix', '-p', type=str, default='',
        help='File prefix to work on.'
    )
    parser.add_argument(
        '--image-suffix', type=str, default='_0000',
        help='Image suffix, defaults to "_0000".'
    )
    parser.add_argument(
        '--seg-suffix', type=str, default='',
        help='Segmentation suffix, defaults to "".'
    )
    parser.add_argument(
        '--max-workers', '-w', type=int, default=mp.cpu_count(),
        help='Max worker to run in parallel proccess, defaults to multiprocessing.cpu_count().'
    )
    parser.add_argument(
        '--quiet', '-q', action="store_true", default=False,
        help='Do not display inputs and progress bar, defaults to false (display).'
    )

    # Parse the command-line arguments
    args = parser.parse_args()

    # Get the command-line argument values
    images_path = args.images_dir
    segs_path = args.segs_dir
    ofolder = args.ofolder
    prefix = args.prefix
    image_suffix = args.image_suffix
    seg_suffix = args.seg_suffix
    max_workers = args.max_workers
    quiet = args.quiet

    # Print the argument values if not quiet
    if not quiet:
        print(textwrap.dedent(f'''
            Running {Path(__file__).stem} with the following params:
            images_path = "{images_path}"
            segs_path = "{segs_path}"
            ofolder = "{ofolder}"
            prefix = "{prefix}"
            image_suffix = "{image_suffix}"
            seg_suffix = "{seg_suffix}"
            max_workers = {max_workers}
            quiet = {quiet}
        '''))

    measure_seg_mp(
        images_path=images_path,
        segs_path=segs_path,
        ofolder_path=ofolder,
        prefix=prefix,
        image_suffix=image_suffix,
        seg_suffix=seg_suffix,
        max_workers=max_workers,
        quiet=quiet,
    )