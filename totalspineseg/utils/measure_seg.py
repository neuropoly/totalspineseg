import json, os, textwrap, argparse
import multiprocessing as mp
import importlib
from tqdm.contrib.concurrent import process_map
from functools import partial
from pathlib import Path
import numpy as np
from skimage import measure, transform, io, morphology, draw
from scipy import ndimage as ndi
import math
from scipy import interpolate
import scipy
import platform
import csv
import warnings
import cv2

from totalspineseg.utils.image import Image, resample_nib, zeros_like
import pyvista as pv
from skimage.morphology import ball, binary_dilation
import totalspineseg.resources as ressources

warnings.filterwarnings("ignore")


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
        '--labels-dir', '-l', type=Path, required=True,
        help='The folder where input NIfTI labels (at the posterior tip of the discs) files are located (required).'
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
        '--label-suffix', type=str, default='',
        help='Label suffix, defaults to "".'
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
    labels_path = args.labels_dir
    ofolder = args.ofolder
    prefix = args.prefix
    image_suffix = args.image_suffix
    seg_suffix = args.seg_suffix
    label_suffix = args.label_suffix
    max_workers = args.max_workers
    quiet = args.quiet

    # Use default mapping path
    ressources_path = importlib.resources.files(ressources)
    mapping_path = os.path.join(ressources_path, 'labels_maps/tss_map.json')

    # Print the argument values if not quiet
    if not quiet:
        print(textwrap.dedent(f'''
            Running {Path(__file__).stem} with the following params:
            images_path = "{images_path}"
            segs_path = "{segs_path}"
            labels_path = "{labels_path}"
            ofolder = "{ofolder}"
            prefix = "{prefix}"
            image_suffix = "{image_suffix}"
            seg_suffix = "{seg_suffix}"
            label_suffix = "{label_suffix}"
            mapping_path = "{mapping_path}"
            max_workers = {max_workers}
            quiet = {quiet}
        '''))

    measure_seg_mp(
        images_path=images_path,
        segs_path=segs_path,
        labels_path=labels_path,
        ofolder_path=ofolder,
        prefix=prefix,
        image_suffix=image_suffix,
        seg_suffix=seg_suffix,
        label_suffix=label_suffix,
        mapping_path=mapping_path,
        max_workers=max_workers,
        quiet=quiet,
    )

def measure_seg_mp(
        images_path,
        segs_path,
        labels_path,
        ofolder_path,
        prefix='',
        image_suffix='_0000',
        seg_suffix='',
        label_suffix='',
        mapping_path='',
        max_workers=mp.cpu_count(),
        quiet=False,
    ):
    '''
    Wrapper function to handle multiprocessing.
    '''
    images_path = Path(images_path)
    segs_path = Path(segs_path)
    labels_path = Path(labels_path)
    ofolder_path = Path(ofolder_path)

    glob_pattern = f'{prefix}*{image_suffix}.nii.gz'

    # Process the NIfTI image and segmentation files
    image_path_list = list(images_path.glob(glob_pattern))
    seg_path_list = [segs_path / image_path.relative_to(images_path).parent / image_path.name.replace(f'{image_suffix}.nii.gz', f'{seg_suffix}.nii.gz') for image_path in image_path_list]
    labels_path_list = [labels_path / image_path.relative_to(images_path).parent / image_path.name.replace(f'{image_suffix}.nii.gz', f'{label_suffix}.nii.gz') for image_path in image_path_list]

    # Load mapping
    with open(mapping_path, 'r') as file:
        mapping = json.load(file)

    process_map(
        partial(
            _measure_seg,
            ofolder_path=ofolder_path,
            mapping=mapping,
        ),
        image_path_list,
        seg_path_list,
        labels_path_list,
        max_workers=max_workers,
        chunksize=1,
        disable=quiet,
    )

def _measure_seg(
        img_path,
        seg_path,
        label_path,
        ofolder_path,
        mapping
    ):
    '''
    Wrapper function to handle IO.
    '''
    # Load image and segmentation
    img = Image(str(img_path)).change_orientation('RPI')
    seg = Image(str(seg_path)).change_orientation('RPI')
    label = Image(str(label_path)).change_orientation('RPI')

    try:
        metrics, imgs = measure_seg(
            img=img,
            seg=seg,
            label=label,
            mapping=mapping,
        )
    except ValueError as e:
        print(f'Error: {seg_path}, {e}')
        return
    
    # Create output folders if does not exists
    img_name=Path(str(seg_path)).name.replace('.nii.gz', '')
    ofolder_path = Path(os.path.join(ofolder_path, img_name))
    csv_folder_path = ofolder_path / 'csv'
    imgs_folder_path = ofolder_path / 'imgs'
    csv_folder_path.mkdir(parents=True, exist_ok=True)
    imgs_folder_path.mkdir(parents=True, exist_ok=True)

    # Save csv files
    for struc in metrics.keys():
        csv_name = f'{struc}.csv'
        csv_path = csv_folder_path / csv_name
        fieldnames=list(metrics[struc][0].keys())
        with open(str(csv_path), mode='w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in metrics[struc]:
                writer.writerow(row)
    
    # Save images
    for name, img in imgs.items():
        img_name = f'{name}.png'
        img_path = imgs_folder_path / img_name
        if len(img.shape) == 3:
            save_isometric_png(img, img_path)
        else:
            if 'foramen' in img_name:
                cv2.imwrite(img_path, img*125)
            else:
                cv2.imwrite(img_path, img*255)
    
def measure_seg(img, seg, label, mapping):
    '''
    Compute morphometric measurements of the spinal canal, the intervertebral discs and the neural foramen
    '''
    # Fetch discs label coordinates
    discs_label = label.getNonZeroCoordinates(sorting='z')

    # Fetch unique segmentation values
    unique_seg = np.unique(seg.data)

    # Resample image to isotropic resolution
    nx, ny, nz, nt, px, py, pz, pt = seg.dim
    pr = min([px, py, pz])
    seg = resample_nib(seg, new_size=[pr, pr, pr], new_size_type='mm', interpolation='nn', verbose=False)
    img = resample_nib(img, new_size=[pr, pr, pr], new_size_type='mm', interpolation='linear', verbose=False)

    # Normalize image intensity
    img.data = (img.data - np.mean(img.data)) / np.std(img.data) # Normalize with mean and std
    img.data = (img.data - img.data.min()) / (img.data.max() - img.data.min() + 1e-8)

    # Create dict with z-slice and values for discs posterior tip
    disc_slices = {}
    for x, y, z, v in discs_label:
        # Rescale z base on image resolution
        z_rescaled = int(round(z * pz / pr))
        disc_slices[z_rescaled] = v

    # Extract spinal canal from segmentation (CSF + SC)
    seg_canal = zeros_like(seg)
    seg_canal.data[seg.data == mapping['SC']] = 1
    seg_canal.data[seg.data == mapping['CSF']] = 2

    # Extract canal centerline
    centerline = get_centerline(seg_canal)

    # Init output dictionaries with metrics
    metrics = {}
    imgs = {}

    # Measure CSF signal
    seg_csf_data = (seg.data == mapping['CSF']).astype(int)
    properties = measure_csf(img.data, seg_csf_data)
    median_csf_signal = np.median(list(properties['slice_signal'].values()))

    rows = []
    for i, (k, v) in enumerate(properties['slice_signal'].items()):
        row = {
            "structure": "csf",
            "index": i,
            "slice_nb": k,
            "disc_level": disc_slices[k] if k in disc_slices else None,
            "slice_signal": v
            }

        rows.append(row)
    metrics['csf'] = rows

    # Compute metrics onto intervertebral discs
    rows = []
    for struc in mapping.keys():
        if mapping[struc] in unique_seg and '-' in struc: # Intervertbral disc in segmentation
            seg_disc_data = (seg.data == mapping[struc]).astype(int)
            # Check if disc is more than one slice
            if (seg_disc_data.sum(axis=0).sum(axis=0)).astype(bool).sum() > 1:
                properties, img_dict = measure_disc(img_data=img.data, seg_disc_data=seg_disc_data, median_csf_signal=median_csf_signal, pr=pr)

                # Save image
                imgs[f'discs_{struc}_seg'] = img_dict['seg']
                imgs[f'discs_{struc}_img'] = img_dict['img']

                # Create a row
                row = {
                    "structure": "disc",
                    "name": struc,
                    "eccentricity": properties['eccentricity'],
                    "solidity": properties['solidity'],
                    "median_thickness": properties['median_thickness'],
                    "intensity_counts": properties['intensity_counts'],
                    "intensity_bins": properties['intensity_bins'],
                    "center": properties['center'],
                    "volume": properties['volume']
                }
                rows.append(row)
    metrics['discs'] = rows
    
    # Compute metrics onto vertebrae and foramens
    foramens_rows = []
    vertebrae_rows = []
    vert_list = []
    seg_bin = zeros_like(seg) # Segmentation of the vertebral bodies and the discs
    for i, struc in enumerate(mapping.keys()):
        if mapping[struc] in unique_seg  and (10 < mapping[struc] < 50): # Vertebrae
            vert_value = int(struc[1:])
            if struc.startswith('C'):
                if vert_value == 7:
                    next_vert = 'T1'
                else:
                    next_vert = f'C{vert_value+1}'
            if struc.startswith('T'):
                if vert_value == 12:
                    next_vert = 'L1'
                else:
                    next_vert = f'T{vert_value+1}'
            if struc.startswith('L'):
                next_vert = f'L{vert_value+1}'
            if mapping[next_vert] in unique_seg: # two adjacent vertebrae
                # Fetch vertebrae names
                top_vert = struc
                bottom_vert = next_vert
                foramens_name = f'foramens_{top_vert}-{bottom_vert}'

                # Init foramen segmentation
                disc_mask = (seg.data == mapping[f'{top_vert}-{bottom_vert}'])
                seg_foramen_data = disc_mask.astype(int) * 2 # Set disc value to 2
                seg_bin.data[disc_mask] = 1

                # Compute vertebrae properties
                for vert in [top_vert, bottom_vert]:
                    seg_vert_data = (seg.data == mapping[vert]).astype(int)
                    # Check if vertebra is more than one slice
                    if (seg_vert_data.sum(axis=0).sum(axis=0)).astype(bool).sum() > 1:
                        if not vert in vert_list:
                            properties, img_dict, body_array = measure_vertebra(img_data=img.data, seg_vert_data=seg_vert_data, seg_canal_data=seg_canal.data, canal_centerline=centerline, pr=pr)
                            # Save image
                            imgs[f'vertebrae_{vert}_seg'] = img_dict['seg']
                            imgs[f'vertebrae_{vert}_img'] = img_dict['img']

                            # Add vertebral bodies
                            seg_bin.data[body_array.astype(bool)] = 1

                            # Create a row per position/thickness point
                            vertebrae_row = {
                                "structure": "vertebra",
                                "name": vert,
                                "AP_thickness": properties['AP_thickness'],
                                "median_thickness": properties['median_thickness'],
                                "center": properties['center'],
                                "volume": properties['volume']
                            }
                            vertebrae_rows.append(vertebrae_row)
                            vert_list.append(vert)
                    seg_foramen_data += seg_vert_data

                # Compute foramens properties
                foramens_areas, foramens_imgs = measure_foramens(seg_foramen_data=seg_foramen_data, canal_centerline=centerline, pr=pr)
                
                # Save image
                for side,image in foramens_imgs.items():
                    imgs[f'{foramens_name}_{side}'] = image
                
                # Save foramen metrics
                foramens_row = {
                    "structure": "foramen",
                    "name": foramens_name,
                    "right_surface": foramens_areas['right'],
                    "left_surface": foramens_areas['left']
                }
                foramens_rows.append(foramens_row)
    metrics['vertebrae'] = vertebrae_rows
    metrics['foramens'] = foramens_rows
    
    # Create spine centerline using vertebral bodies and discs
    if 50 in unique_seg: # Add sacrum
        seg_bin.data[seg.data == 50] = 1
    dilation_radius = 1.5//pr  # 1.5 mm
    seg_bin.data = binary_dilation(seg_bin.data, ball(dilation_radius)) # Dilate seg_bin to remove gap between discs and vertebrae
    spine_centerline = get_centerline(seg_bin)

    # Compute metrics onto canal segmentation
    properties = measure_canal(seg_canal, centerline, spine_centerline)
    rows = []
    for i in range(len(properties[list(properties.keys())[0]])):
        slice_nb = list(properties[list(properties.keys())[0]].keys())[i]
        row = {
            "structure": "canal",
            "index": i,
            "slice_nb": slice_nb,
            "disc_level": disc_slices[slice_nb] if slice_nb in disc_slices else None,
            }
        for key in properties.keys():
            row[key] = properties[key][slice_nb]
        rows.append(row)
    metrics['canal'] = rows
    return metrics, imgs

def measure_disc(img_data, seg_disc_data, median_csf_signal, pr):
    '''
    Calculate metrics from binary disc segmentation
    '''
    # Fetch coords from image
    coords = np.argwhere(seg_disc_data > 0)
    values = np.array([img_data[c[0], c[1], c[2]] for c in coords])

    # Normalize disc intensity using median CSF signal
    values = values / median_csf_signal

    # Identify the smallest elipsoid that can fit the disc
    ellipsoid = fit_ellipsoid(coords)

    # Extract intensity histogram
    intensity_counts, intensity_bins = np.histogram(values, range=(0, 2.5), bins=100)

    # Extract SI thickness
    bin_size = max(2//pr, 1) # Put 1 bin per 2 mm
    median_thickness = compute_thickness_profile(coords, ellipsoid['rotation_matrix'], bin_size=bin_size)

    # Extract disc volume
    voxel_volume = pr**3
    volume = ellipsoid['volume']*voxel_volume # mm3

    properties = {
        'center': np.round(ellipsoid['center']),
        'median_thickness': median_thickness*pr,
        'intensity_counts': intensity_counts.tolist(),
        'intensity_bins': intensity_bins.tolist(),
        'volume': volume,
        'eccentricity': ellipsoid['eccentricity'],
        'solidity': ellipsoid['solidity']
    }

    # Recreate volume for visualization
    disc_seg, (xmin, xmax, ymin, ymax, zmin, zmax) = crop_around_binary(seg_disc_data)

    # Normalize image intensity
    p10 = np.percentile(img_data, 5)
    p90 = np.percentile(img_data, 95)
    img_disc = (img_data - p10) / (p90 - p10 + 1e-8)

    # Extract 2D cut of disc image
    padding = 8
    ymax, zmax = [v + padding if v + padding < img_disc.shape[1+i] else img_disc.shape[1+i]-1 for i, v in enumerate((ymax, zmax))]
    ymin, zmin = [v - padding if v - padding >= 0 else 0 for v in (ymin, zmin)]
    disc_img = img_disc[xmin:xmax, ymin:ymax, zmin:zmax]
    disc_img = disc_img[int((xmax-xmin)//2)]

    img_dict = {'seg':disc_seg, 'img':disc_img}
    return properties, img_dict

def measure_csf(img_data, seg_csf_data):
    '''
    Extract signal from cerebro spinal fluid (CSF)
    '''
    # Extract min and max index in Z direction
    X, Y, Z = seg_csf_data.nonzero()
    min_z_index, max_z_index = min(Z), max(Z)

    # Loop across z axis
    properties = {
        'slice_signal':{}
    }
    for iz in range(min_z_index, max_z_index + 1):
        # Extract csf coordinates in the slice
        slice_csf = seg_csf_data[:, :, iz].astype(bool)

        # Extract images values using segmentation
        slice_values = img_data[slice_csf]

        # Extract most represented value
        # Take multiple maximums
        hist, bin_edges = np.histogram(slice_values, bins=100)
        signal = np.median(bin_edges[np.argsort(hist)[-10:]])

        # Save values
        properties['slice_signal'][iz] = signal
    return properties

def measure_canal(seg_canal, centerline, spine_centerline):
    '''
    Based on https://github.com/spinalcordtoolbox/spinalcordtoolbox/blob/master/spinalcordtoolbox/process_seg.py

    Expected orientation is RPI

    Extract canal metrics using:
    - canal segmentation
    - spine segmentation
    '''
    # List properties
    property_list = [
        'area_canal',
        'area_spinalcord',
        'diameter_AP_canal',
        'diameter_AP_spinalcord',
        'diameter_RL_canal',
        'diameter_RL_spinalcord',
        'canal_centroid',
        'eccentricity_canal',
        'eccentricity_spinalcord',
        'orientation',
        'solidity_canal',
        'solidity_spinalcord',
        'angle_AP',
        'angle_RL',
        'length'
    ]

    # Fetch dimensions from image.
    nx, ny, nz, nt, px, py, pz, pt = seg_canal.dim

    # Extract min and max index in Z direction
    deriv = {int(z_ref): centerline["derivative"][:2, index] for index, z_ref in enumerate(centerline["position"][2])}
    pos_spine = {int(z_ref): np.round(spine_centerline["position"][:2, index]).astype(int) for index, z_ref in enumerate(spine_centerline["position"][2])}

    min_z_index = max(min(deriv.keys()), min(pos_spine.keys()))
    max_z_index = min(max(deriv.keys()), max(pos_spine.keys()))
    # Loop across the S-I slices
    shape_properties = {key: {} for key in property_list}
    for iz in range(min_z_index, max_z_index + 1):
        patch_canal = (seg_canal.data[:, :, iz] > 0).astype(int)
        patch_sc = (seg_canal.data[:, :, iz] == 1).astype(int)
        # Extract tangent vector to the centerline (i.e. its derivative)
        tangent_vect = np.array([deriv[iz][0] * px, deriv[iz][1] * py, pz])
        # Compute the angle about AP axis between the centerline and the normal vector to the slice
        angle_AP_rad = math.atan2(tangent_vect[0], tangent_vect[2])
        # Compute the angle about RL axis between the centerline and the normal vector to the slice
        angle_RL_rad = math.atan2(tangent_vect[1], tangent_vect[2])
        # Apply affine transformation to account for the angle between the centerline and the normal to the patch
        tform = transform.AffineTransform(scale=(np.cos(angle_RL_rad), np.cos(angle_AP_rad)))
        # Convert to float64, to avoid problems in image indexation causing issues when applying transform.warp
        patch_canal = patch_canal.astype(np.float64)
        patch_sc = patch_sc.astype(np.float64)
        # Create a circle centered on the spine centerline
        patch_centerline = np.zeros_like(patch_canal)
        rr, cc = draw.disk((pos_spine[iz][0], pos_spine[iz][1]), radius=8, shape=patch_canal.shape)
        patch_centerline[rr, cc] = 1.0

        patch_canal_scaled = transform.warp(
            patch_canal,
            tform.inverse,
            output_shape=patch_canal.shape,
            order=1,
        )
        patch_centerline_scaled = transform.warp(
            patch_centerline,
            tform.inverse,
            output_shape=patch_centerline.shape,
            order=1,
        )
        patch_sc_scaled = transform.warp(
            patch_sc,
            tform.inverse,
            output_shape=patch_sc.shape,
            order=1,
        )
        # Calculate shape metrics
        shape_property = _properties2d(patch_canal_scaled, patch_sc_scaled, patch_centerline_scaled, [px, py])

        if shape_property is not None:
            # Add custom fields
            shape_property['angle_AP'] = angle_AP_rad * 180.0 / math.pi
            shape_property['angle_RL'] = angle_RL_rad * 180.0 / math.pi
            shape_property['length'] = pz / (np.cos(angle_AP_rad) * np.cos(angle_RL_rad))
            # Loop across properties and assign values for function output
            for property_name in property_list:
                shape_properties[property_name][iz] = shape_property[property_name]
        else:
            print(f'Warning: error with slice {iz}.')

    return shape_properties

def measure_vertebra(img_data, seg_vert_data, seg_canal_data, canal_centerline, pr):
    '''
    Returns:
        properties: python dict
        image: 3D numpy array representing the structure of interest
    '''
    # Extract vertebra coords
    coords = np.argwhere(seg_vert_data > 0)

    # Extract z position (SI) of the vertebra
    vert_pos = np.mean(coords,axis=0)
    z_mean = vert_pos[-1]

    # Find closest point and derivative onto the canal centerline
    closest_canal_idx = np.argmin(abs(canal_centerline['position'][2]-z_mean))
    canal_pos = canal_centerline['position'][:,closest_canal_idx]
    canal_deriv = canal_centerline['derivative'][:,closest_canal_idx]

    # Create two perpendicular vectors u1 and u2
    v = canal_deriv
    tmp = np.array([1, 0, 0]) # Init temporary non colinear vector
    u1 = np.cross(v, tmp)
    u1 /= np.linalg.norm(u1)
    u2 = np.cross(v, u1)
    u2 /= np.linalg.norm(u2)

    # Define vector w with angle theta in u1u2 plane
    def w(u1, u2, theta): 
        return np.cos(theta) * u1 + np.sin(theta) * u2
    
    def cutting_plane(theta):
        # Init normal vector of the plane
        n = np.cross(v, w(u1, u2, theta))
        n_norm = np.linalg.norm(n)
        if n_norm == 0:
            raise ValueError("Normal vector has zero norm.")
        n = n/n_norm

        dot_product = np.dot(coords-canal_pos, n)
        pos = np.sum(dot_product > 0)
        neg = len(coords) - pos
        proportion = pos/(pos+neg)
        return proportion - 0.5

    # Find function zero between 0 and pi
    res = scipy.optimize.root_scalar(cutting_plane, bracket=[0, np.pi], method='brentq')
    if not res.converged:
        raise ValueError('Did not find a cutting plane for vertebra')
    best_theta = res.root
    if (vert_pos[1]-canal_pos[1])*w(u1, u2, best_theta)[1] < 0:
        # Orient vector from canal to body
        best_theta += np.pi
    u = np.cross(v, w(u1, u2, best_theta)) # create last vector

    # Find canal distance to vertebral body
    canal_slice_coords = np.argwhere(seg_canal_data[:,:,int(np.round(z_mean))]>0)
    projections = np.dot(canal_slice_coords-canal_pos[:2], w(u1, u2, best_theta)[:2])
    AP_radius = projections.max()

    # Isolate vertebral body
    anterior_pos = np.array([canal_pos[0], canal_pos[1]+AP_radius, canal_pos[2]])
    projections = np.dot(coords-anterior_pos, w(u1, u2, best_theta))
    body_coords = coords[projections>0]
    body_pos = np.mean(body_coords,axis=0)

    # Fetch AP thickness
    AP_thickness = np.max(projections)

    # Compute thickness profile vertebral body
    coordinate_system = np.stack((u, w(u1, u2, best_theta), v), axis=0)
    values = np.array([img_data[c[0], c[1], c[2]] for c in body_coords])
    bin_size = max(2//pr, 1) # Put 1 bin per 2 mm
    median_thickness = compute_thickness_profile(body_coords, coordinate_system, bin_size=bin_size)
    
    # Extract vertebral body volume
    voxel_volume = pr**3
    volume = body_coords.shape[0]*voxel_volume # mm3

    properties = {
        'center': np.round(body_pos),
        'median_thickness': median_thickness*pr,
        'AP_thickness': AP_thickness*pr,
        'volume': volume,
    }

    # Recreate volume for visualization
    rotate_inv = np.linalg.inv(coordinate_system)
    rot_coords = coords @ rotate_inv
    rot_coords = rot_coords - np.min(rot_coords, axis=0)
    vert_seg = np.zeros((int(np.round(np.max(rot_coords[:,0]))), int(np.round(np.max(rot_coords[:,1]))), int(np.round(np.max(rot_coords[:,2])))))
    for i, coord in enumerate(rot_coords):
        if projections[i]>0: # vertebral body
            vert_seg[int(np.round(coord[0]-1)), int(np.round(coord[1]-1)), int(np.round(coord[2]-1))]=2
        else:
            vert_seg[int(np.round(coord[0]-1)), int(np.round(coord[1]-1)), int(np.round(coord[2]-1))]=1
    
    # Recreate body volume without rotation
    body_array = np.zeros_like(seg_vert_data)
    for coord in body_coords:
        body_array[coord[0], coord[1], coord[2]]=1

    # Recreate volume for visualization
    _, (xmin, xmax, ymin, ymax, zmin, zmax) = crop_around_binary(body_array)
    
    # Normalize image intensity
    p10 = np.percentile(img_data, 5)
    p90 = np.percentile(img_data, 95)
    img_vert = (img_data - p10) / (p90 - p10 + 1e-8)

    # Extract 2D cut of disc image
    padding = 8
    ymax, zmax = [v + padding if v + padding < img_vert.shape[1+i] else img_vert.shape[1+i]-1 for i, v in enumerate((ymax, zmax))]
    ymin, zmin = [v - padding if v - padding >= 0 else 0 for v in (ymin, zmin)]
    vert_img = img_vert[xmin:xmax, ymin:ymax, zmin:zmax]
    vert_img = vert_img[int((xmax-xmin)//2)]

    img_dict = {'seg':vert_seg, 'img':vert_img}

    return properties, img_dict, body_array

def measure_foramens(seg_foramen_data, canal_centerline, pr):
    '''
    This function measures the surface of the left and right neural foramen formed by 2 vertebrae and a disc

    Parameters:
        seg_foramen_data: 3D numpy array containing
            - a segmentation of the top and bottom vertebrae equal to 1
            - a segmentation of the intervertebral discs in between equal to 2
        canal_centerline: python dict

    Returns:
        foramens_areas: python dict
            left and right surface of the foramina
        foramens_imgs:
            left and right image of the foramina
    '''
    # Extract vertebrae and disc coords
    coords = np.argwhere(seg_foramen_data > 0)

    # Extract z position (SI) of the disc center of mass
    disc_coords = np.argwhere(seg_foramen_data == 2)
    disc_pos = np.mean(coords,axis=0)
    z_mean = disc_pos[-1]

    # Find closest point and derivative onto the canal centerline
    closest_canal_idx = np.argmin(abs(canal_centerline['position'][2]-z_mean))
    canal_pos, canal_deriv = canal_centerline['position'][:,closest_canal_idx], canal_centerline['derivative'][:,closest_canal_idx]
    
    # Create two perpendicular vectors u1 and u2
    v = canal_deriv
    tmp = np.array([1, 0, 0]) # Init temporary non colinear vector
    u1 = np.cross(v, tmp)
    u1 /= np.linalg.norm(u1)
    u2 = np.cross(v, u1)
    u2 /= np.linalg.norm(u2)

    # Define vector w with angle theta in u1u2 plane
    def w(u1, u2, theta): 
        return np.cos(theta) * u1 + np.sin(theta) * u2

    # Use dichotomie to construct plane formed by v and w for different theta
    # to find the most optimal plane that cut the segmentation in half
    theta_min = 0
    theta_max = np.pi
    theta = theta_min
    proportion = {}
    incr=0
    nb_incr = 10
    while incr < nb_incr:
        n = np.cross(v, w(u1, u2, theta)) # normal vector of the plane
        n_norm = np.linalg.norm(n)
        if n_norm == 0:
            raise ValueError("Normal vector has zero norm.")
        n = n/n_norm
        
        # Count point on positive side of plane
        dot_product = np.dot(coords-canal_pos, n)
        pos = np.sum(dot_product > 0)
        neg = len(coords) - pos
        proportion[theta] = pos/(pos+neg)

        # Use dichotomy
        if len(proportion.keys()) == 1:
            theta = theta_max
        elif len(proportion.keys()) == 2:
            theta = (theta_min + theta_max)/2
        else:
            if (proportion[theta] - 0.5)*(proportion[theta_min] - 0.5)<=0:
                theta_max = theta
            else:
                theta_min = theta
            theta = (theta_min + theta_max)/2
        incr+=1
    
    # Use best plane to cut the segmentation in two halfs
    best_theta = list(proportion.keys())[np.argmin(np.abs(np.array(list(proportion.values()))-0.5))]
    if (disc_pos[1]-canal_pos[1])*w(u1, u2, best_theta)[1] < 0:
        # Orient vector from canal to disc
        best_theta += np.pi

    n = np.cross(v, w(u1, u2, best_theta)) # normal vector of the plane
    n /= np.linalg.norm(n)
    dot_product = np.dot(coords-canal_pos, n)

    # Distinguish left-from-right
    pos_coords = dot_product>0
    if n[0] > 0: # Oriented from right to left RPI
        halfs = {"left": coords[pos_coords], "right":coords[~pos_coords]}
    else:
        halfs = {"right": coords[pos_coords], "left":coords[~pos_coords]}

    # Project foramens
    foramens_areas = {}
    foramens_imgs = {}
    for side, coords in halfs.items():
        # Project coords in vw plane
        x_coords = np.dot(coords, v)
        y_coords = np.dot(coords, w(u1, u2, best_theta))

        # Center the image onto the segmentation
        x_coords = x_coords - np.min(x_coords)
        y_coords = y_coords - np.min(y_coords)

        # Round coordinates
        x_coords = np.round(x_coords).astype(int)
        y_coords = np.round(y_coords).astype(int)

        # Create image
        img = np.zeros((np.max(x_coords), np.max(y_coords)))
        for x, y in zip(x_coords, y_coords):
            img[x-1, y-1]=1

        # Inverse image
        labeled_bg = morphology.remove_small_objects(~img.astype(bool), min_size=64)

        # Padd image to connect exterior components
        labeled_bg = np.pad(labeled_bg, pad_width=(5,5), mode='constant', constant_values=1)

        # Label all component and extract regions
        labeled_img, _ = ndi.label(labeled_bg)
        regions = measure.regionprops(labeled_img)
        
        # Save foramens
        areas = [region.area for region in regions]
        if len(areas) > 1:
            # Select second biggest region assuming it is the foramen
            foramen_region = regions[np.argsort(areas)[-2]]
            foramen_mask = labeled_img == foramen_region.label
            foramens_imgs[side] = labeled_bg + foramen_mask.astype(int)
            
            # Calculate foramen area
            pixel_surface = pr**2
            foramen_area = np.argwhere(foramen_mask > 0).shape[0]*pixel_surface #mm2
            foramens_areas[side] = foramen_area
        else:
            print('Error with foramen, possibly not closed shape.')
            foramens_areas[side] = -1
            foramens_imgs[side] = labeled_bg
    return foramens_areas, foramens_imgs

def fit_ellipsoid(coords):
    # Compute the center of mass of the disc
    center = coords.mean(axis=0)

    # Center the coordinates
    coords_centered = coords - center

    # Compute covariance matrix
    cov = np.cov(coords_centered, rowvar=False)

    # Eigen-decomposition
    eigvals, eigvecs = np.linalg.eigh(cov)  # Use eigh for symmetric matrix

    # Reorder vectors to be close to default coordinate system
    new_order = np.argmax(abs(eigvecs), axis=1)
    if len(new_order) != len(set(new_order)): # Check if no axis are overwritten
        raise ValueError("One eigen vector was overwritten")
    eigvecs = eigvecs[:,new_order]
    eigvals = eigvals[new_order]

    # Extract axis length from eigen values
    radii = np.sqrt(eigvals) * 2  # Multiply by 2 for full axis length

    # Compute eccentricity in the RL-AP plane
    eccentricity = np.sqrt(1 - (np.min(eigvals[:2]) / np.max(eigvals[:2]))) if np.max(eigvals[:2]) > 0 else 0

    # Compute solidity = volume / convex_hull_volume
    volume = coords.shape[0]  # Ellipsoid volume
    try:
        hull = scipy.spatial.ConvexHull(coords)
        solidity = volume / hull.volume if hull.volume > 0 else 0
    except:
        solidity = -1  # Fallback if Qhull fails

    # Results
    ellipsoid = {
        'center': center,
        'axes_lengths': radii,
        'rotation_matrix': eigvecs,  # columns = directions of axes
        'eccentricity': eccentricity,
        'solidity': solidity,
        'volume': volume
    }
    return ellipsoid

def compute_thickness_profile(coords, rotation_matrix, bin_size=1.0):
    """
    Measure thickness profile of the segmentation by splitting RL-AP plane into bins.
    
    Parameters:
        coords: (N, 3) array of 3D points of the segmentations
        rotation_matrix: (3, 3) array corresponding to new coordinate system
        bin_size: RL-AP plane resolution for thickness extraction (in voxels)

    Returns:
        median thicknesses: thickness in the RL-AP plane
    """
    # Project voxel coordinates onto the axis
    center = np.mean(coords,axis=0) 
    coords_centered = coords - center

    # Rotate coords_centered
    eigvecs = rotation_matrix
    rot_coords = coords_centered @ eigvecs

    # Find min and max dimensions of the disc in the RL-AP plane
    min_RL, max_RL = rot_coords[:,0].min(), rot_coords[:,0].max()
    min_AP, max_AP = rot_coords[:,1].min(), rot_coords[:,1].max()

    # Pad min and max to reduce effect of discs edges
    min_RL += bin_size
    max_RL -= bin_size
    min_AP += bin_size
    max_AP -= bin_size

    # Create bin matrix along RL and AP dimension
    bins_RL = np.arange(min_RL, max_RL + bin_size, bin_size)
    bin_indices_RL = np.digitize(rot_coords[:,0], bins_RL) - 1
    bins_AP = np.arange(min_AP, max_AP + bin_size, bin_size)
    bin_indices_AP = np.digitize(rot_coords[:,1], bins_AP) - 1

    # Fetch median thickness
    thicknesses = []
    for x in range(len(bins_RL) - 1):
        slice_mask_RL = bin_indices_RL == x
        for y in range(len(bins_AP) - 1):
            slice_mask_AP = bin_indices_AP == y
            slice_mask = slice_mask_RL*slice_mask_AP
            if any(slice_mask):
                # Extract voxels in the square
                slice_coords = rot_coords[slice_mask]

                # Find max and minimum in square
                min_SI, max_SI = slice_coords[:,2].min(), slice_coords[:,2].max()

                # Extract thickness
                thicknesses.append(max_SI-min_SI)
    return np.median(np.array(thicknesses))

def get_centerline(seg):
    '''
    Based on https://github.com/spinalcordtoolbox/spinalcordtoolbox/blob/master/spinalcordtoolbox/centerline/core.py

    Extract centerline from canal segmentation using center of mass and interpolate with bspline
    Expect orientation RPI
    '''
    arr = np.array(np.where(seg.data))
    # Loop across SI axis and average coordinates within duplicate SI values
    sorted_avg = []
    for i_si in np.unique(arr[2]):
        sorted_avg.append(arr[:, arr[2] == i_si].mean(axis=1))
    x_mean, y_mean, z_mean = np.array(sorted_avg).T
    z_ref = np.array(range(z_mean.min().astype(int), z_mean.max().astype(int) + 1))

    # Interpolate centerline
    px, py, pz = seg.dim[4:7]
    x_centerline_fit, x_centerline_deriv = bspline(z_mean, x_mean, z_ref, smooth=8000, pz=pz) # Increase smoothing...
    y_centerline_fit, y_centerline_deriv = bspline(z_mean, y_mean, z_ref, smooth=8000, pz=pz) # ...a lot

    # Construct output
    arr_ctl = np.array([x_centerline_fit, y_centerline_fit, z_ref])
    arr_ctl_der = np.array([x_centerline_deriv, y_centerline_deriv, np.ones_like(z_ref)])

    # Create centerline dictionary
    centerline = {"position": arr_ctl, "derivative": arr_ctl_der}
    return centerline

def bspline(x, y, xref, smooth, deg_bspline=3, pz=1):
    """
    Copied from https://github.com/spinalcordtoolbox/spinalcordtoolbox/blob/master/spinalcordtoolbox/centerline/curve_fitting.py
    Bspline interpolation.

    The smoothing factor (s) is calculated based on an empirical formula (made by JCA, based on
    preliminary results) and is a function of pz, density of points and an input smoothing parameter (smooth). The
    formula is adjusted such that the parameter (smooth) produces similar smoothing results than a Hanning window with
    length smooth, as implemented in linear().

    :param x:
    :param y:
    :param xref:
    :param smooth: float: Smoothing factor. 0: no smoothing, 5: moderate smoothing, 50: large smoothing
    :param deg_bspline: int: Degree of spline
    :param pz: float: dimension of pixel along superior-inferior direction (z, assuming RPI orientation)
    :return:
    """
    if len(x) <= deg_bspline:
        deg_bspline -= 2
    density = (float(len(x)) / len(xref)) ** 2
    s = density * smooth * pz / float(3)
    # Then, run bspline interpolation
    tck = interpolate.splrep(x, y, s=s, k=deg_bspline)
    y_fit = interpolate.splev(xref, tck, der=0)
    y_fit_der = interpolate.splev(xref, tck, der=1)
    return y_fit, y_fit_der

def _properties2d(canal, spinalcord, spine_centerline, dim):
    """
    Compute shape property of the input 2D image. Accounts for partial volume information.
    :param canal: 2D input canal image in uint8 or float (weighted for partial volume) that has a single object.
    :param spinalcord: 2D input spinal cord image in uint8 or float (weighted for partial volume).
    :param spine_centerline: 2D input spine centerline image in uint8 or float (weighted for partial volume).
    :param dim: [px, py]: Physical dimension of the image (in mm). X,Y respectively correspond to AP,RL.
    :return:
    """
    # Check if slice is empty
    if np.all(canal < 1e-6):
        print('The slice is empty.')
        return None
    
    # Normalize between 0 and 1 (also check if slice is empty)
    canal_norm = (canal - canal.min()) / (canal.max() - canal.min())

    # Fetch spine position
    spine_pos = np.nonzero(spine_centerline)
    spine_pos = np.array([np.round(np.mean(spine_pos[0])), np.round(np.mean(spine_pos[1]))])

    # Binarize canal using threshold at 0.5
    canal_bin = np.array(canal_norm > 0.5, dtype='uint8')

    # Extract canal slice center of mass
    canal_coords = np.nonzero(canal_bin)
    canal_pos = np.array([np.round(np.mean(canal_coords[0])), np.round(np.mean(canal_coords[1]))])

    # Create vector v from canal_pos to spine pos and normalize it
    v = spine_pos - canal_pos
    v = v / np.linalg.norm(v)

    # Create w an orthogonal vector to v
    w = np.array([-v[1], v[0]])

    # Compute AP diameter along v 
    v_mask = cylindrical_mask(shape=canal_bin.shape, p0=canal_pos, v=v, radius=4) # Create cylindrical mask along v
    AP_mask = v_mask*canal_bin
    AP_coords = np.argwhere(AP_mask)
    projections = np.dot(AP_coords, v)  # Project onto vector
    diameter_AP_canal = (projections.max() - projections.min())*dim[0] # AP length = max - min projection

    # Compute RL diameter along w
    w_mask = cylindrical_mask(shape=canal_bin.shape, p0=canal_pos, v=w, radius=4) # Create cylindrical mask along w
    RL_mask = w_mask*canal_bin
    RL_coords = np.argwhere(RL_mask)
    projections = np.dot(RL_coords, w)  # Project onto vector
    diameter_RL_canal = (projections.max() - projections.min())*dim[1] # RL length = max - min projection

    # Compute area
    area_canal = np.sum(canal_bin) * dim[0] * dim[1]

    # Compute eccentricity
    max_diameter_canal = max(diameter_AP_canal, diameter_RL_canal)
    min_diameter_canal = min(diameter_AP_canal, diameter_RL_canal)
    eccentricity_canal = np.sqrt(1 - min_diameter_canal**2/max_diameter_canal**2) if max_diameter_canal > 0 else 0

    # Compute angle between AP and patient AP
    u2 = np.array([0, 1])
    angle = angle_between(u2, v) # rad
    angle = 360*angle/(2*np.pi)

    # Deal with https://github.com/spinalcordtoolbox/spinalcordtoolbox/issues/2307
    if any(x in platform.platform() for x in ['Darwin-15', 'Darwin-16']):
        solidity_canal = -1
    else:
        solidity_canal = compute_solidity_2d(canal_bin)

    # Compute spinal cord metrics if not empty else set metrics to -1
    if not np.all(spinalcord < 1e-6):
        # Normalize between 0 and 1 (also check if slice is empty)
        spinalcord_norm = (spinalcord - spinalcord.min()) / (spinalcord.max() - spinalcord.min())

        # Binarize canal using threshold at 0.5 Necessary input for measure.regionprops
        spinalcord_bin = np.array(spinalcord_norm > 0.5, dtype='uint8')

        # Extract spinalcord slice center of mass
        spinalcord_coords = np.nonzero(spinalcord_bin)
        spinalcord_pos = np.array([np.round(np.mean(spinalcord_coords[0])), np.round(np.mean(spinalcord_coords[1]))])

        # Compute AP diameter along v
        v_mask = cylindrical_mask(shape=spinalcord_bin.shape, p0=spinalcord_pos, v=v, radius=4) # Create cylindrical mask along v
        AP_mask = v_mask*spinalcord_bin
        AP_coords = np.argwhere(AP_mask)
        projections = np.dot(AP_coords, v)  # Project onto vector
        diameter_AP_spinalcord = (projections.max() - projections.min())*dim[0] # AP length = max - min projection

        # Compute RL diameter along w
        w_mask = cylindrical_mask(shape=spinalcord_bin.shape, p0=spinalcord_pos, v=w, radius=4) # Create cylindrical mask along w
        RL_mask = w_mask*spinalcord_bin
        RL_coords = np.argwhere(RL_mask)
        projections = np.dot(RL_coords, w)  # Project onto vector
        diameter_RL_spinalcord = (projections.max() - projections.min())*dim[1] # RL length = max - min projection

        # Compute area 
        area_spinalcord = np.sum(spinalcord_bin) * dim[0] * dim[1]

        # Compute eccentricity 
        max_diameter_spinalcord = max(diameter_AP_spinalcord, diameter_RL_spinalcord)
        min_diameter_spinalcord = min(diameter_AP_spinalcord, diameter_RL_spinalcord)
        eccentricity_spinalcord = np.sqrt(1 - min_diameter_spinalcord**2/max_diameter_spinalcord**2) if max_diameter_spinalcord > 0 else 0

        # Deal with https://github.com/spinalcordtoolbox/spinalcordtoolbox/issues/2307
        if any(x in platform.platform() for x in ['Darwin-15', 'Darwin-16']):
            solidity_spinalcord = -1
        else:
            solidity_spinalcord = compute_solidity_2d(spinalcord_bin)
    else:
        area_spinalcord = -1
        diameter_AP_spinalcord = -1
        diameter_RL_spinalcord = -1
        eccentricity_spinalcord = -1
        solidity_spinalcord = -1
    
    # Fill up dictionary
    properties = {
        'area_canal': area_canal,
        'area_spinalcord': area_spinalcord,
        'diameter_AP_canal': diameter_AP_canal,
        'diameter_AP_spinalcord': diameter_AP_spinalcord,
        'diameter_RL_canal': diameter_RL_canal,
        'diameter_RL_spinalcord': diameter_RL_spinalcord,
        'canal_centroid': canal_pos,
        'eccentricity_canal': eccentricity_canal,
        'eccentricity_spinalcord': eccentricity_spinalcord,
        'orientation': angle,
        'solidity_canal': solidity_canal,  # convexity measure
        'solidity_spinalcord': solidity_spinalcord,  # convexity measure
    }
    return properties

def angle_between(u, v):
    u = np.array(u)
    v = np.array(v)
    dot_product = np.dot(u, v)
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    # Clip to avoid numerical issues (e.g. arccos(1.0000001))
    cos_theta = np.clip(dot_product / (norm_u * norm_v), -1.0, 1.0)
    angle = np.arccos(cos_theta)  # in radians
    return angle

def cylindrical_mask(shape, p0, v, radius):
    """
    Create a 2D binary mask of a 'cylinder' (thick line) along vector `v` passing through `p0`.
    
    Args:
        shape (tuple): Shape of the 2D image (height, width)
        p0 (np.array): A point [y, x] the vector passes through
        v (np.array): Direction vector [vy, vx]
        radius (float): Cylinder radius (in pixels)

    Returns:
        mask (2D np.array): Binary mask with True inside the cylinder
    """
    h, w = shape
    Y, X = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    
    # Shift grid by point
    dy = Y - p0[0]
    dx = X - p0[1]
    
    # Normalize direction vector
    v = v / np.linalg.norm(v)
    
    # Compute perpendicular distance to the line (vector projection method)
    # Distance = ||(point - p0) - ((point - p0) · v) * v||
    dot = dx * v[1] + dy * v[0]
    proj_x = dot * v[1]
    proj_y = dot * v[0]
    perp_x = dx - proj_x
    perp_y = dy - proj_y
    dist = np.sqrt(perp_x**2 + perp_y**2)
    
    # Inside mask if distance < radius
    mask = dist < radius
    return mask

def compute_solidity_2d(mask):
    labeled = measure.label(mask)
    props = measure.regionprops(labeled)[0]
    return props.solidity

def save_isometric_png(volume, filename):
    volume, _ = crop_around_binary(volume)
    plotter = pv.Plotter(off_screen=True, border=False)
    plotter.add_volume(pv.wrap(volume), cmap="viridis", opacity="linear", shade=True)
    # # See top view
    # plotter.view_xy()
    plotter.remove_scalar_bar()
    plotter.camera.zoom(1.3)
    plotter.show(screenshot=filename)

def crop_around_binary(volume):
    """
    Crop a 3D numpy array around the non-zero region and return the cropped size.

    Args:
        volume : np.ndarray
            3D binary numpy array (bool or 0/1 values).

    Returns:
        cropped : np.ndarray
            Cropped 3D volume.
        bbox : tuple
            Bounding box coordinates.
    """
    assert volume.ndim == 3, "Input must be a 3D array"

    # Find non-zero coordinates
    coords = np.argwhere(volume)
    if coords.size == 0:
        return (volume.copy(), None)

    xmin, ymin, zmin = coords.min(axis=0)
    xmax, ymax, zmax = coords.max(axis=0)
    cropped = volume[xmin:xmax+1, ymin:ymax+1, zmin:zmax+1]

    return cropped, (xmin, xmax, ymin, ymax, zmin, zmax)

if __name__ == '__main__':
    # img_path = '/home/GRAMES.POLYMTL.CA/p118739/data_nvme_p118739/data/datasets/test-tss/out/input/sub-001_ses-A_acq-isotropic_T2w_0000.nii.gz'
    # seg_path = '/home/GRAMES.POLYMTL.CA/p118739/data_nvme_p118739/data/datasets/test-tss/out/step2_output/sub-001_ses-A_acq-isotropic_T2w.nii.gz'
    # label_path = '/home/GRAMES.POLYMTL.CA/p118739/data_nvme_p118739/data/datasets/test-tss/out/step1_levels/sub-001_ses-A_acq-isotropic_T2w.nii.gz'

    img_path = '/home/GRAMES.POLYMTL.CA/p118739/data_nvme_p118739/data/datasets/test-tss/spider_output/input/sub-039_acq-lowresSag_T2w_0000.nii.gz'
    seg_path = '/home/GRAMES.POLYMTL.CA/p118739/data_nvme_p118739/data/datasets/test-tss/spider_output/step2_output/sub-039_acq-lowresSag_T2w.nii.gz'
    label_path = '/home/GRAMES.POLYMTL.CA/p118739/data_nvme_p118739/data/datasets/test-tss/spider_output/step1_levels/sub-039_acq-lowresSag_T2w.nii.gz'

    ofolder_path = 'test'

    # Load totalspineseg mapping
    with open('totalspineseg/resources/labels_maps/tss_map.json', 'r') as file:
        mapping = json.load(file)
    
    # Run measure_seg
    _measure_seg(img_path, seg_path, label_path, ofolder_path, mapping)