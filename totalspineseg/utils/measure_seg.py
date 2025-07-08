import json
import multiprocessing as mp
from tqdm.contrib.concurrent import process_map
from functools import partial
import numpy as np
from skimage import measure, transform, io, morphology
from scipy import ndimage as ndi
import math
from scipy import interpolate
import platform

from totalspineseg.utils.image import Image, resample_nib, zeros_like


# def extract_levels_mp(
#         segs_path,
#         output_segs_path,
#         prefix='',
#         seg_suffix='',
#         mapping_path='',
#         overwrite=False,
#         max_workers=mp.cpu_count(),
#         quiet=False,
#     ):
#     '''
#     Wrapper function to handle multiprocessing.
#     '''
#     segs_path = Path(segs_path)
#     output_segs_path = Path(output_segs_path)

#     glob_pattern = f'{prefix}*{seg_suffix}.nii.gz'

#     # Process the NIfTI image and segmentation files
#     seg_path_list = list(segs_path.glob(glob_pattern))
#     output_seg_path_list = [output_segs_path / _.relative_to(segs_path).parent / _.name.replace(f'{seg_suffix}.nii.gz', f'{output_seg_suffix}.nii.gz') for _ in seg_path_list]

#     process_map(
#         partial(
#             _extract_levels,
#             canal_labels=canal_labels,
#             disc_labels=disc_labels,
#             c1_label=c1_label,
#             c2_label=c2_label,
#             overwrite=overwrite,
#         ),
#         seg_path_list,
#         output_seg_path_list,
#         max_workers=max_workers,
#         chunksize=1,
#         disable=quiet,
#     )

# def _extract_levels(
#         seg_path,
#         output_seg_path,
#         canal_labels=[],
#         disc_labels=[],
#         c1_label=0,
#         c2_label=0,
#         overwrite=False,
#     ):
#     '''
#     Wrapper function to handle IO.
#     '''
#     seg_path = Path(seg_path)
#     output_seg_path = Path(output_seg_path)

#     # If the output image already exists and we are not overriding it, return
#     if not overwrite and output_seg_path.exists():
#         return

#     # Load segmentation
#     seg = nib.load(seg_path)

#     try:
#         output_seg = extract_levels(
#             seg,
#             canal_labels=canal_labels,
#             disc_labels=disc_labels,
#             c1_label=c1_label,
#             c2_label=c2_label,
#         )
#     except ValueError as e:
#         output_seg_path.is_file() and output_seg_path.unlink()
#         print(f'Error: {seg_path}, {e}')
#         return

#     # Ensure correct segmentation dtype, affine and header
#     output_seg = nib.Nifti1Image(
#         np.asanyarray(output_seg.dataobj).round().astype(np.uint8),
#         output_seg.affine, output_seg.header
#     )
#     output_seg.set_data_dtype(np.uint8)
#     output_seg.set_qform(output_seg.affine)
#     output_seg.set_sform(output_seg.affine)

#     # Make sure output directory exists and save the segmentation
#     output_seg_path.parent.mkdir(parents=True, exist_ok=True)
#     nib.save(output_seg, output_seg_path)


def measure_seg(seg_path, mapping):
    '''
    Compute morphometric measurements of the spinal canal, the intervertebral discs and the neural foramen
    '''
    # Create reverse mapping:
    rev_mapping = {v:k for k,v in mapping.items()}

    # Load segmentation
    seg = Image(seg_path).change_orientation('RPI')

    # Fetch unique segmentation values
    unique_seg = np.unique(seg.data)

    # Resample image to isotropic resolution
    nx, ny, nz, nt, px, py, pz, pt = seg.dim
    pr = min([px, py, pz])
    seg = resample_nib(seg, new_size=[pr, pr, pr], new_size_type='mm', interpolation='nn')

    # Extract spinal canal from segmentation (CSF + SC)
    seg_canal = zeros_like(seg)
    seg_canal.data = np.isin(seg.data, [mapping['CSF'], mapping['SC']]).astype(int)

    # Init dictionary with metrics
    metrics = {}

    # Compute metrics onto canal segmentation
    metrics['canal'], centerline = measure_canal(seg_canal)

    # Compute metrics onto intervertebral discs
    # for struc in mapping.keys():
    #     if mapping[struc] in unique_seg and '-' in struc: # Intervertbral disc in segmentation
    #         seg_disc = zeros_like(seg)
    #         seg_disc.data = (seg.data == mapping[struc]).astype(int)
    #         metrics[struc] = measure_disc(seg_disc=seg_disc)
    
    # Compute metrics onto vertebrae foramens
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

                metrics[structure_name] = measure_foramens(seg_foramen=seg_foramen, canal_centerline=centerline, pr=pr)


def measure_disc(seg_disc):
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
    
    return

def measure_canal(seg_canal):
    '''
    Based on https://github.com/spinalcordtoolbox/spinalcordtoolbox/blob/master/spinalcordtoolbox/process_seg.py

    Extract canal metrics based on canal binary segmentation
    Expected orientation is RPI
    '''
    # List properties
    property_list = ['area',
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
    # for iz in range(min_z_index, max_z_index + 1):
    #     current_patch = seg_canal.data[:, :, iz]
    #     # Extract tangent vector to the centerline (i.e. its derivative)
    #     tangent_vect = np.array([deriv[iz][0] * px, deriv[iz][1] * py, pz])
    #     # Compute the angle about AP axis between the centerline and the normal vector to the slice
    #     angle_AP_rad = math.atan2(tangent_vect[0], tangent_vect[2])
    #     # Compute the angle about RL axis between the centerline and the normal vector to the slice
    #     angle_RL_rad = math.atan2(tangent_vect[1], tangent_vect[2])
    #     # Apply affine transformation to account for the angle between the centerline and the normal to the patch
    #     tform = transform.AffineTransform(scale=(np.cos(angle_RL_rad), np.cos(angle_AP_rad)))
    #     # Convert to float64, to avoid problems in image indexation causing issues when applying transform.warp
    #     current_patch = current_patch.astype(np.float64)
    #     current_patch_scaled = transform.warp(current_patch,
    #                                             tform.inverse,
    #                                             output_shape=current_patch.shape,
    #                                             order=1,
    #                                             )
    #     # Calculate shape metrics
    #     shape_property = _properties2d(current_patch_scaled, [px, py])

    #     if shape_property is not None:
    #         # Add custom fields
    #         shape_property['angle_AP'] = angle_AP_rad * 180.0 / math.pi
    #         shape_property['angle_RL'] = angle_RL_rad * 180.0 / math.pi
    #         shape_property['length'] = pz / (np.cos(angle_AP_rad) * np.cos(angle_RL_rad))
    #         # Loop across properties and assign values for function output
    #         for property_name in property_list:
    #             shape_properties[property_name][iz] = shape_property[property_name]
    #     else:
    #         print(f'Warning: error with slice {iz}.')

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
            import cv2
            cv2.imwrite("foramen_error.png", img*255)
            raise ValueError('Error with foramen, possibly not closed shape. See foramen_error.png')

        # Calculate foramen area
        pixel_surface = pr**2
        foramen_area = np.argwhere(foramen_mask > 0).shape[0]*pixel_surface #mm2
        foramen_areas[side] = foramen_area
    return foramen_areas


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


def _properties2d(image, dim):
    """
    Copied from https://github.com/spinalcordtoolbox/spinalcordtoolbox/blob/master/spinalcordtoolbox/process_seg.py

    Compute shape property of the input 2D image. Accounts for partial volume information.
    :param image: 2D input image in uint8 or float (weighted for partial volume) that has a single object.
    :param dim: [px, py]: Physical dimension of the image (in mm). X,Y respectively correspond to AP,RL.
    :return:
    """
    upscale = 5  # upscale factor for resampling the input image (for better precision)
    pad = 3  # padding used for cropping
    # Check if slice is empty
    if np.all(image < 1e-6):
        print('The slice is empty.')
        return None
    # Normalize between 0 and 1 (also check if slice is empty)
    image_norm = (image - image.min()) / (image.max() - image.min())
    # Convert to float64
    image_norm = image_norm.astype(np.float64)
    # Binarize image using threshold at 0. Necessary input for measure.regionprops
    image_bin = np.array(image_norm > 0.5, dtype='uint8')
    # Get all closed binary regions from the image (normally there is only one)
    regions = measure.regionprops(image_bin, intensity_image=image_norm)
    # Check number of regions
    if len(regions) > 1:
        print('There is more than one object on this slice.')
        return None
    region = regions[0]
    # Get bounding box of the object
    minx, miny, maxx, maxy = region.bbox
    # Use those bounding box coordinates to crop the image (for faster processing)
    image_crop = image_norm[np.clip(minx-pad, 0, image_bin.shape[0]): np.clip(maxx+pad, 0, image_bin.shape[0]),
                            np.clip(miny-pad, 0, image_bin.shape[1]): np.clip(maxy+pad, 0, image_bin.shape[1])]
    # Oversample image to reach sufficient precision when computing shape metrics on the binary mask
    image_crop_r = transform.pyramid_expand(image_crop, upscale=upscale, sigma=None, order=1)
    # Binarize image using threshold at 0. Necessary input for measure.regionprops
    image_crop_r_bin = np.array(image_crop_r > 0.5, dtype='uint8')
    # Get all closed binary regions from the image (normally there is only one)
    regions = measure.regionprops(image_crop_r_bin, intensity_image=image_crop_r)
    region = regions[0]
    # Compute area with weighted segmentation and adjust area with physical pixel size
    area = np.sum(image_crop_r) * dim[0] * dim[1] / upscale ** 2
    # Compute ellipse orientation, modulo pi, in deg, and between [0, 90]
    orientation = fix_orientation(region.orientation)
    # Find RL and AP diameter based on major/minor axes and cord orientation=
    [diameter_AP, diameter_RL] = \
        _find_AP_and_RL_diameter(region.major_axis_length, region.minor_axis_length, orientation,
                                 [i / upscale for i in dim])
    # TODO: compute major_axis_length/minor_axis_length by summing weighted voxels along axis
    # Deal with https://github.com/spinalcordtoolbox/spinalcordtoolbox/issues/2307
    if any(x in platform.platform() for x in ['Darwin-15', 'Darwin-16']):
        solidity = np.nan
    else:
        solidity = region.solidity
    # Fill up dictionary
    properties = {
        'area': area,
        'diameter_AP': diameter_AP,
        'diameter_RL': diameter_RL,
        'centroid': region.centroid,
        'eccentricity': region.eccentricity,
        'orientation': orientation,
        'solidity': solidity,  # convexity measure
    }

    return properties

if __name__ == '__main__':
    seg_path = '/home/GRAMES.POLYMTL.CA/p118739/data_nvme_p118739/data/datasets/measure-discs/out/step2_output/sub-016_acq-isotropic_T2w.nii.gz'
    
    # Load totalspineseg mapping
    with open('totalspineseg/resources/labels_maps/tss_map.json', 'r') as file:
        mapping = json.load(file)
    
    # Run measure_seg
    measure_seg(seg_path, mapping)