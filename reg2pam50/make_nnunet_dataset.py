import os
from pathlib import Path
import tempfile
import shutil
import nibabel as nib
import numpy as np
import subprocess

def run_command(command):
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    print(result.stdout)
    print(result.stderr)

# Settings

path = Path(r'')
nnraw_path = Path(r'')

datasets = (
    ('single', 'data-single-subject', '_totalseg', True),
    ('multi', 'data-multi-subject', '_totalseg', True),
    ('whole', 'whole-spine', '_totalseg', True),
    ('spider', 'SPIDER', '_totalseg', False),
)

def main():

    nnraw_path.mkdir(parents=True, exist_ok=True)

    for dsid, dsname, segs_suffix, from_top in datasets:
        dspath = path / dsname
        # Process the NIfTI image and segmentation files
        subject_dir_list = list(dspath.glob('sub-*'))
        for i, subject_dir in enumerate(subject_dir_list):
            subject = subject_dir.name
            print(f'Working on {subject} ({i + 1}/{len(subject_dir_list)})')
            if not (subject_dir / 'anat').is_dir():
                print(f'Warning: "anat" folder not found for {subject}')
                continue
            for image_path in (subject_dir / 'anat').glob('*.nii.gz'):
                image_name = image_path.name.replace('.nii.gz', '')
                seg_path = dspath / 'derivatives' / 'labels' / subject / 'anat' / f'{image_name}{segs_suffix}.nii.gz'
                if not seg_path.is_file():
                    print(f'Warning: Segmentation file not found segmentation file for {image_path}')
                    continue
                print(f"Processing {image_path.name} and {seg_path.name}...")

                make_nnunt_image_seg(
                    image_path,
                    seg_path,
                    nnraw_path / 'imagesTr' / f'{dsid}_{image_name.replace("sub-", "")}_0000.nii.gz',
                    nnraw_path / 'labelsTr' / f'{dsid}_{image_name.replace("sub-", "")}.nii.gz',
                    from_top
                )
                print(f"Done")

def make_nnunt_image_seg(in_img, in_seg, out_img, out_seg, from_top):

    in_img, in_seg, out_img, out_seg = Path(in_img), Path(in_seg), Path(out_img), Path(out_seg)

    temp_path = Path(tempfile.mkdtemp())
    out_img.parent.mkdir(parents=True, exist_ok=True)
    out_seg.parent.mkdir(parents=True, exist_ok=True)

    try:

        # Copy files to tmp in canonial orientation
        run_command(f'sct_image -i {in_img} -setorient LPI -o {temp_path}/img.nii.gz')
        run_command(f'sct_image -i {in_seg} -setorient LPI -o {temp_path}/seg.nii.gz')

        # Ensure img and seg in the same space
        run_command(f'sct_register_multimodal -i {temp_path}/seg.nii.gz -d {temp_path}/img.nii.gz -identity 1 -x nn -o {temp_path}/seg.nii.gz')
        
        img = nib.load(f'{temp_path}/seg.nii.gz')
        data = img.get_fdata()
        
        # Create an array of z indices
        z_indices = np.tile(np.arange(data.shape[2]), (data.shape[0], data.shape[1], 1))
        # Create an array of y indices
        y_indices = np.broadcast_to(np.arange(data.shape[1])[..., np.newaxis], data.shape)

        if from_top:
            # Cut at the z of the most anteior voxel the lowest vertebrae in the image
            last_vert = data[(data >= 1) & (data <= 24)].max()
            zmin = -1
            while zmin == -1 or 49 not in data[..., zmin]:
                zmin = z_indices[(data == last_vert) & (y_indices == y_indices[data == last_vert].max())].min()
                last_vert -= 1
            run_command(f'sct_crop_image -i {temp_path}/img.nii.gz -zmin {zmin} -o {out_img}')
            run_command(f'sct_crop_image -i {temp_path}/seg.nii.gz -zmin {zmin} -o {out_seg}')
        elif 42 in data:
            # Cut at the lowest voxel of T12-L1 IVD
            zmax = z_indices[data == 207].min()
            run_command(f'sct_crop_image -i {temp_path}/img.nii.gz -zmax {zmax} -o {out_img}')
            run_command(f'sct_crop_image -i {temp_path}/seg.nii.gz -zmax {zmax} -o {out_seg}')
    finally:
        shutil.rmtree(temp_path)

if __name__ == '__main__':
    main()