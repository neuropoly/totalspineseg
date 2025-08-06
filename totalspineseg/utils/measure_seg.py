import json, os
import multiprocessing as mp
from tqdm.contrib.concurrent import process_map
from functools import partial
from pathlib import Path
import numpy as np
from skimage import measure, transform, io, morphology
from scipy import ndimage as ndi
import math
from scipy import interpolate
import scipy
import platform
import csv
import warnings

from totalspineseg.utils.image import Image, resample_nib, zeros_like

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

def measure_seg_mp(
        images_path,
        segs_path,
        ofolder_path,
        prefix='',
        image_suffix='_0000',
        seg_suffix='',
        mapping_path='',
        max_workers=mp.cpu_count(),
        quiet=False,
    ):
    '''
    Wrapper function to handle multiprocessing.
    '''
    images_path = Path(images_path)
    segs_path = Path(segs_path)
    ofolder_path = Path(ofolder_path)

    glob_pattern = f'{prefix}*{image_suffix}.nii.gz'

    # Process the NIfTI image and segmentation files
    image_path_list = list(images_path.glob(glob_pattern))
    seg_path_list = [segs_path / _.relative_to(images_path).parent / _.name.replace(f'{image_suffix}.nii.gz', f'{seg_suffix}.nii.gz') for _ in image_path_list]
    
    # Load mapping 
    with open(mapping_path, 'r') as file:
        mapping = json.load(file)

    process_map(
        partial(
            _measure_seg,
            ofolder_path=ofolder_path,
            mapping=mapping,
        ),
        seg_path_list,
        output_seg_path_list,
        max_workers=max_workers,
        chunksize=1,
        disable=quiet,
    )

def _measure_seg(
        img_path,
        seg_path,
        ofolder_path,
        mapping
    ):
    '''
    Wrapper function to handle IO.
    '''
    # Load image and segmentation
    img = Image(str(img_path)).change_orientation('RPI')
    seg = Image(str(seg_path)).change_orientation('RPI')

    try:
        metrics = measure_seg(
            img=img,
            seg=seg,
            mapping=mapping,
        )
    except ValueError as e:
        print(f'Error: {seg_path}, {e}')
        return
    
    # Create output folder if does not exists
    ofolder_path = Path(ofolder_path)
    ofolder_path.mkdir(parents=True, exist_ok=True)

    # Save csv files
    img_name=Path(str(seg_path)).name.replace('.nii.gz', '')
    for struc in metrics.keys():
        csv_name = f'{img_name}_{struc}.csv'
        csv_path = ofolder_path / csv_name
        fieldnames=list(metrics[struc][0].keys())
        with open(str(csv_path), mode='w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in metrics[struc]:
                writer.writerow(row)

def measure_seg(img, seg, mapping):
    '''
    Compute morphometric measurements of the spinal canal, the intervertebral discs and the neural foramen
    '''
    # Create reverse mapping:
    rev_mapping = {v:k for k,v in mapping.items()}

    # Fetch unique segmentation values
    unique_seg = np.unique(seg.data)

    # Resample image to isotropic resolution
    nx, ny, nz, nt, px, py, pz, pt = seg.dim
    pr = min([px, py, pz])
    seg = resample_nib(seg, new_size=[pr, pr, pr], new_size_type='mm', interpolation='nn')

    # Extract spinal canal from segmentation (CSF + SC)
    seg_canal = zeros_like(seg)
    seg_canal.data = np.isin(seg.data, [mapping['CSF'], mapping['SC']]).astype(int)

    # Extract binary segmentation
    seg_bin = zeros_like(seg)
    seg_bin.data = seg.data != 0

    # Init dictionary with metrics
    metrics = {'canal':{}, 'discs':[], 'foramens':[]}

    # Compute metrics onto canal segmentation
    properties, centerline = measure_canal(seg_canal, seg_bin)
    rows = []
    for i in range(len(properties[list(properties.keys())[0]])):
        row = {
            "structure": "canal",
            "index": i
            }
        for key in properties.keys():
            row[key] = properties[key][i]
        rows.append(row)
    metrics['canal'] = rows

    # Compute metrics onto intervertebral discs
    rows = []
    for struc in mapping.keys():
        if mapping[struc] in unique_seg and '-' in struc: # Intervertbral disc in segmentation
            seg_disc = zeros_like(seg)
            seg_disc.data = (seg.data == mapping[struc]).astype(int)
            properties = measure_disc(seg_disc=seg_disc, pr=pr)

            # Create a row per position/thickness point
            for i, (pos, thick) in enumerate(zip(properties['position'], properties['thickness'])):
                row = {
                    "structure": "disc",
                    "name": struc,
                    "index": i,
                    "position_x": pos[0],
                    "position_y": pos[1],
                    "thickness": thick,
                    "volume": properties['volume'] if i == 0 else ""  # only once per disc
                }
                rows.append(row)
    metrics['discs'] = rows
    
    # Compute metrics onto vertebrae foramens
    rows = []
    for struc in mapping.keys():
        if mapping[struc] in unique_seg  and (10 < mapping[struc] < 50): # Vertebrae
            if mapping[struc]+1 in unique_seg: # two adjacent vertebrae
                # Fetch vertebrae names
                top_vert = struc
                bottom_vert = rev_mapping[mapping[struc]+1]
                structure_name = f'foramens_{top_vert}-{bottom_vert}'

                # Merge top vert, bottom vert and intervetebral disc segmentations
                seg_foramen = zeros_like(seg)
                seg_foramen.data = np.isin(seg.data, [mapping[struc], mapping[struc]+1]).astype(int)
                seg_foramen.data[seg.data == mapping[f'{top_vert}-{bottom_vert}']] = 2 # Set disc value to 2

                # Compute properties
                properties = measure_foramens(seg_foramen=seg_foramen, canal_centerline=centerline, pr=pr)
                row = {
                    "structure": "foramen",
                    "name": structure_name,
                    "right_surface": properties['areas']['right'],
                    "left_surface": properties['areas']['left']
                }
                rows.append(row)
    metrics['foramens'] = rows
    return metrics

def measure_disc(seg_disc, pr):
    '''
    Calculate metrics from binary disc segmentation
    '''
    # Fetch coords from image
    coords = np.argwhere(seg_disc.data > 0)

    # Identify the smallest elipsoid that can fit the disc
    ellipsoid = fit_ellipsoid(coords)

    # Measure thickness profile along the rotated SI axis
    bin_size = max(2//pr, 1) # Put 1 bin per 2 mm
    position, thickness = compute_thickness_profile(coords, ellipsoid, bin_size=bin_size)

    # Extract disc volume
    voxel_volume = pr**3
    volume = coords.shape[0]*voxel_volume # mm3

    properties = {
        'position': position,
        'thickness': thickness,
        'volume': volume
    }
    return properties

def measure_canal(seg_canal, seg_bin):
    '''
    Based on https://github.com/spinalcordtoolbox/spinalcordtoolbox/blob/master/spinalcordtoolbox/process_seg.py

    Expected orientation is RPI

    Extract canal metrics using:
    - canal segmentation
    - spine segmentation
    '''
    # List properties
    property_list = [
        'area',
        'angle_AP',
        'angle_RL',
        'diameter_AP',
        'diameter_RL',
        'eccentricity',
        'orientation',
        'solidity',
        'length'
    ]

    # Fetch dimensions from image.
    nx, ny, nz, nt, px, py, pz, pt = seg_canal.dim

    # Extract min and max index in Z direction
    X, Y, Z = seg_canal.data.nonzero()
    min_z_index, max_z_index = min(Z), max(Z)

    # Extract derivative of centerline for angle correction
    arr_ctl, arr_ctl_der = get_centerline(seg_canal)
    deriv = {int(z_ref): arr_ctl_der[:2, index] for index, z_ref in enumerate(arr_ctl[2])}

    centerline = {"position":arr_ctl, "derivative":arr_ctl_der}

    # Loop across the S-I slices
    shape_properties = {key: np.full(nz, np.nan, dtype=np.double) for key in property_list}
    for iz in range(min_z_index, max_z_index + 1):
        patch_canal = seg_canal.data[:, :, iz]
        patch_spine = seg_bin.data[:, :, iz]
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
        patch_spine = patch_spine.astype(np.float64)
        patch_canal_scaled = transform.warp(
            patch_canal,
            tform.inverse,
            output_shape=patch_canal.shape,
            order=1,
        )
        patch_spine_scaled = transform.warp(
            patch_spine,
            tform.inverse,
            output_shape=patch_spine.shape,
            order=1,
        )
        # Calculate shape metrics
        shape_property = _properties2d(patch_canal_scaled, patch_spine_scaled, [px, py])

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

    return shape_properties, centerline

def measure_foramens(seg_foramen, canal_centerline, pr):
    '''
    seg_foramen contains:
    - a segmentation of the top and bottom vertebrae equal to 1
    - a segmentation of the intervertebral discs in between equal to 2
    '''
    # Extract vertebrae and disc coords
    coords = np.argwhere(seg_foramen.data > 0)

    # Extract z position (SI) of the disc center of mass
    disc_coords = np.argwhere(seg_foramen.data == 2)
    z_mean = np.mean(disc_coords[:,2])

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
    foramen_areas = {}
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
        else:
            raise ValueError('Error with foramen, possibly not closed shape. See foramen_error.png')

        # Calculate foramen area
        pixel_surface = pr**2
        foramen_area = np.argwhere(foramen_mask > 0).shape[0]*pixel_surface #mm2
        foramen_areas[side] = foramen_area

        properties = {
            "areas":foramen_areas
        }
    return properties

def grade_discs():
    return

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

    # Results
    ellipsoid = {
        'center': center,
        'axes_lengths': radii,
        'rotation_matrix': eigvecs  # columns = directions of axes
    }
    return ellipsoid

def compute_thickness_profile(coords, ellipsoid, bin_size=1.0):
    """
    Measure thickness profile of the segmentation by splitting RL-AP plane into bins.
    
    Parameters:
        coords: (N, 3) array of 3D points of the segmentations
        ellipsoid: dict with 'center' and 'rotation_matrix'
        bin_size: RL-AP plane resolution for thickness extraction (in voxels)

    Returns:
        positions: (RL_coords, AP_coords) for each point
        thicknesses: thickness for each point in the RL-AP plane
    """
    center = ellipsoid['center']
    eigvecs = ellipsoid['rotation_matrix']
    axis_main = eigvecs[:, 2]  # Main axis (e.g., head-foot)

    # Project voxel coordinates onto the axis
    coords_centered = coords - center

    # Rotate coords_centered
    rot_coords = coords_centered @ eigvecs

    # Find min and max dimensions of the disc in the RL-AP plane
    min_RL, max_RL = rot_coords[:,0].min(), rot_coords[:,0].max()
    min_AP, max_AP = rot_coords[:,1].min(), rot_coords[:,1].max()

    # Create bin matrix along RL and AP dimension
    bins_RL = np.arange(min_RL, max_RL + bin_size, bin_size)
    bin_indices_RL = np.digitize(rot_coords[:,0], bins_RL) - 1
    bins_AP = np.arange(min_AP, max_AP + bin_size, bin_size)
    bin_indices_AP = np.digitize(rot_coords[:,1], bins_AP) - 1

    thicknesses = []
    positions = []

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

                # Extract thickness and position
                thicknesses.append(max_SI-min_SI)
                positions.append([(bins_RL[x] + bins_RL[x+1]) / 2, (bins_AP[y] + bins_AP[y+1]) / 2])

    return np.array(positions), np.array(thicknesses)

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
    x_centerline_fit, x_centerline_deriv = bspline(z_mean, x_mean, z_ref, smooth=20, pz=pz)
    y_centerline_fit, y_centerline_deriv = bspline(z_mean, y_mean, z_ref, smooth=20, pz=pz)

    # Construct output
    arr_ctl = np.array([x_centerline_fit, y_centerline_fit, z_ref])
    arr_ctl_der = np.array([x_centerline_deriv, y_centerline_deriv, np.ones_like(z_ref)])

    return arr_ctl, arr_ctl_der

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

def _properties2d(canal, spine, dim):
    """
    Compute shape property of the input 2D image. Accounts for partial volume information.
    :param canal: 2D input canal image in uint8 or float (weighted for partial volume) that has a single object.
    :param spine: 2D input spine/canal image in uint8 or float (weighted for partial volume) that has a single object.
    :param dim: [px, py]: Physical dimension of the image (in mm). X,Y respectively correspond to AP,RL.
    :return:
    """
    upscale = 5  # upscale factor for resampling the input image (for better precision)
    pad = 3  # padding used for cropping
    # Check if slice is empty
    if np.all(canal < 1e-6):
        print('The slice is empty.')
        return None
    # Normalize between 0 and 1 (also check if slice is empty)
    canal_norm = (canal - canal.min()) / (canal.max() - canal.min())
    spine_norm = (spine - spine.min()) / (spine.max() - spine.min())

    # Convert to float64
    canal_norm = canal_norm.astype(np.float64)
    spine_norm = spine_norm.astype(np.float64)

    # Binarize canal using threshold at 0.5 Necessary input for measure.regionprops
    canal_bin = np.array(canal_norm > 0.5, dtype='uint8')
    spine_bin = np.array(spine_norm > 0.5, dtype='uint8')

    # Extract canal slice center of mass
    canal_coords = np.nonzero(canal_bin)
    x_mean = np.mean(canal_coords[0])
    y_mean = np.mean(canal_coords[1])
    canal_pos = np.array([np.round(x_mean), np.round(y_mean)])

    # Take two perpendicular vectors u1 and u2
    u1 = np.array([1, 0])
    u2 = np.array([0, 1])
    
    # Define a rotating vector with an angle theta
    def v(u1, u2, theta):
        return np.cos(theta) * u1 + np.sin(theta) * u2
    
    # Count pixels on the positive side of the vector
    spine_coords = np.argwhere(spine_bin > 0)
    total_pixels = np.sum(spine_bin)
    def proportion(theta):
        return np.sum(np.dot(spine_coords-canal_pos, v(u1, u2, theta))>0)/total_pixels

    # Find function maximum
    res = scipy.optimize.minimize_scalar(lambda theta: -proportion(theta), bounds=(0, 2 * np.pi), method='bounded')
    theta_max = res.x

    # Compute AP diameter along v
    v_mask = cylindrical_mask(shape=canal_bin.shape, p0=canal_pos, v=v(u1,u2,theta_max), radius=4) # Create cylindrical mask along v
    AP_mask = v_mask*canal_bin
    AP_coords = np.argwhere(AP_mask)
    projections = np.dot(AP_coords, v(u1,u2,theta_max))  # Project onto vector
    diameter_AP = (projections.max() - projections.min())*dim[0] # AP length = max - min projection
    
    # Compute RL diameter along w 
    def w(u1, u2, theta): 
        return np.cos(theta) * u2 - np.sin(theta) * u1
    w_mask = cylindrical_mask(shape=canal_bin.shape, p0=canal_pos, v=w(u1,u2,theta_max), radius=4) # Create cylindrical mask along v
    RL_mask = w_mask*canal_bin
    RL_coords = np.argwhere(RL_mask)
    projections = np.dot(RL_coords, w(u1,u2,theta_max))  # Project onto vector
    diameter_RL = (projections.max() - projections.min())*dim[1] # RL length = max - min projection
    
    # Compute area
    area = np.sum(canal_bin) * dim[0] * dim[1]

    # Compute eccentricity
    eccentricity = np.sqrt(1 - AP_length**2/RL_length**2)

    # Compute solidity
    solidity = compute_solidity(canal_bin)

    # Compute angle between AP and patient AP
    angle = angle_between(u2, v(u1,u2,theta_max)) # rad
    angle = 360*angle/(2*np.pi)

    # Deal with https://github.com/spinalcordtoolbox/spinalcordtoolbox/issues/2307
    if any(x in platform.platform() for x in ['Darwin-15', 'Darwin-16']):
        solidity = np.nan
    else:
        solidity = compute_solidity(canal_bin)
    
    # Fill up dictionary
    properties = {
        'area': area,
        'diameter_AP': diameter_AP,
        'diameter_RL': diameter_RL,
        'centroid': canal_pos,
        'eccentricity': eccentricity,
        'orientation': angle,
        'solidity': solidity,  # convexity measure
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

def compute_solidity(mask):
    labeled = measure.label(mask)
    props = measure.regionprops(labeled)[0]
    return props.solidity

if __name__ == '__main__':
    img_path = '/home/GRAMES.POLYMTL.CA/p118739/data_nvme_p118739/data/datasets/measure-discs/img/sub-016_acq-isotropic_T2w.nii.gz'
    seg_path = '/home/GRAMES.POLYMTL.CA/p118739/data_nvme_p118739/data/datasets/measure-discs/out/step2_output/sub-016_acq-isotropic_T2w.nii.gz'
    ofolder_path = 'test'

    # Load totalspineseg mapping
    with open('totalspineseg/resources/labels_maps/tss_map.json', 'r') as file:
        mapping = json.load(file)
    
    # Run measure_seg
    _measure_seg(img_path, seg_path, ofolder_path, mapping)