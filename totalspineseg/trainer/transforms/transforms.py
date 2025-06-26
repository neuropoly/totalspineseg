import torch
import torch.nn.functional as F

from batchgeneratorsv2.transforms.base.basic_transform import ImageOnlyTransform, BasicTransform

import torchio as tio
import scipy.ndimage as ndi
from scipy.stats import norm
from functools import partial
import gc

import random

### Contrast transform (Laplace, Scharr, Histogram Equalization, Log, Sqrt, Exp, Sin, Sig) --> removed Gamma and Inverse

class ConvTransform(ImageOnlyTransform):
    '''
    Based on https://github.com/spinalcordtoolbox/disc-labeling-playground/blob/main/src/ply/models/transform.py
    '''
    def __init__(self, kernel_type: str = 'Laplace', absolute: bool = False, retain_stats: bool = False):
        super().__init__()
        if kernel_type not in  ["Laplace","Scharr"]:
            raise NotImplementedError('Currently only "Laplace" and "Scharr" are supported.')
        else:
            self.kernel_type = kernel_type
        self.absolute = absolute
        self.retain_stats = retain_stats

    def get_parameters(self, **data_dict) -> dict:
        spatial_dims = len(data_dict['image'].shape) - 1
        if spatial_dims == 2:
            if self.kernel_type == "Laplace":
                kernel = torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=torch.float32)
            elif self.kernel_type == "Scharr":
                kernel_x = torch.tensor([[-3, 0, 3], [-10, 0, -10], [-3, 0, 3]], dtype=torch.float32)
                kernel_y = torch.tensor([[-3, -10, -3], [0, 0, 0], [3, 10, 3]], dtype=torch.float32)
                kernel = [kernel_x, kernel_y]
        elif spatial_dims == 3:
            if self.kernel_type == "Laplace":
                kernel = -1.0 * torch.ones(3, 3, 3, dtype=torch.float32)
                kernel[1, 1, 1] = 26.0
            elif self.kernel_type == "Scharr":
                kernel_x = torch.tensor([[[  9,    0,    -9],
                                          [ 30,    0,   -30],
                                          [  9,    0,    -9]],

                                          [[ 30,    0,   -30],
                                           [100,    0,  -100],
                                           [ 30,    0,   -30]],

                                          [[  9,    0,    -9],
                                           [ 30,    0,   -30],
                                           [  9,    0,    -9]]], dtype=torch.float32)
                
                kernel_y = torch.tensor([[[    9,   30,    9],
                                          [    0,    0,    0],
                                          [   -9,  -30,   -9]],

                                         [[  30,  100,   30],
                                          [   0,    0,    0],
                                          [ -30, -100,  -30]],

                                         [[   9,   30,    9],
                                          [   0,    0,    0],
                                          [  -9,  -30,   -9]]], dtype=torch.float32)
                
                kernel_z = torch.tensor([[[   9,   30,   9],
                                          [  30,  100,  30],
                                          [   9,   30,   9]],

                                         [[   0,    0,   0],
                                          [   0,    0,   0],
                                          [   0,    0,   0]],

                                         [[   -9,  -30,  -9],
                                          [  -30, -100, -30],
                                          [   -9,  -30,  -9]]], dtype=torch.float32)
                kernel = [kernel_x, kernel_y, kernel_z]
        else:
            raise ValueError(f"{self.__class__} can only handle 2D or 3D images.")

        return {
            'kernel_type': self.kernel_type,
            'kernel': kernel,
            'absolute': self.absolute,
            'retain_stats': self.retain_stats
        }
    
    def _apply_to_image(self, img: torch.Tensor, **params) -> torch.Tensor:
        '''
        We expect (C, X, Y) or (C, X, Y, Z) shaped inputs for image and seg
        '''
        for c in range(img.shape[0]):
            if params['retain_stats']:
                orig_mean = torch.mean(img[c])
                orig_std = torch.std(img[c])
            img_ = img[c].unsqueeze(0).unsqueeze(0)  # adds temp batch and channel dim
            if params['kernel_type'] == 'Laplace':
                tot_ = apply_filter(img_, params['kernel'])
            elif params['kernel_type'] == 'Scharr':
                tot_ = torch.zeros_like(img_)
                for kernel in params['kernel']:
                    if params['absolute']:
                        tot_ += torch.abs(apply_filter(img_, kernel))
                    else:
                        tot_ += apply_filter(img_, kernel)
            img[c] = tot_[0,0]
            if params['retain_stats']:
                mean = torch.mean(img[c])
                std = torch.std(img[c])
                img[c] = (img[c] - mean)/torch.clamp(std, min=1e-7)
                img[c] = img[c]*orig_std + orig_mean # return to original distribution
        return img


class HistogramEqualTransform(ImageOnlyTransform):
    def __init__(self, retain_stats: bool = False):
        super().__init__()
        self.retain_stats = retain_stats
    
    def get_parameters(self, **data_dict) -> dict:
        return {
            'retain_stats': self.retain_stats
        }
    
    def _apply_to_image(self, img: torch.Tensor, **params) -> torch.Tensor:
        for c in range(img.shape[0]):
            if params['retain_stats']:
                orig_mean = torch.mean(img[c])
                orig_std = torch.std(img[c])
            img_min, img_max = img[c].min(), img[c].max()

            # Flatten the image and compute the histogram
            img_flattened = img[c].flatten().to(torch.float32)
            hist, bins = torch.histogram(img_flattened, bins=256)

            # Compute bin edges
            bin_edges = torch.linspace(img_min, img_max, steps=257)  # 256 bins -> 257 edges

            # Compute the normalized cumulative distribution function (CDF)
            cdf = hist.cumsum(dim=0)
            cdf = (cdf - cdf.min()) / (cdf.max() - cdf.min())  # Normalize to [0,1]
            cdf = cdf * (img_max - img_min) + img_min  # Scale back to image range

            # Perform histogram equalization
            indices = torch.searchsorted(bin_edges[:-1], img_flattened)
            img_eq = torch.index_select(cdf, dim=0, index=torch.clamp(indices, 0, 255))
            img[c] = img_eq.reshape(img[c].shape)
            
            if params['retain_stats']:
                # Return to original distribution
                mean = torch.mean(img[c])
                std = torch.std(img[c])
                img[c] = (img[c] - mean)/torch.clamp(std, min=1e-7)
                img[c] = img[c]*orig_std + orig_mean
        return img


class FunctionTransform(ImageOnlyTransform):
    def __init__(self, function, retain_stats : bool = False):
        super().__init__()
        self.function = function
        self.retain_stats = retain_stats

    def get_parameters(self, **data_dict) -> dict:
        return {
            'function': self.function,
            'retain_stats': self.retain_stats
        }
    
    def _apply_to_image(self, img: torch.Tensor, **params) -> torch.Tensor:
        for c in range(img.shape[0]):
            if params['retain_stats']:
                orig_mean = torch.mean(img[c])
                orig_std = torch.std(img[c])

            # Normalize
            img[c] = (img[c] - img.min()) / (img.max() - img.min() + 0.00001)

            # Apply function
            img[c] = params['function'](img[c])

            if params['retain_stats']:
                # Return to original distribution
                mean = torch.mean(img[c])
                std = torch.std(img[c])
                img[c] = (img[c] - mean)/torch.clamp(std, min=1e-7)
                img[c] = img[c]*orig_std + orig_mean
        return img

def apply_filter(x: torch.Tensor, kernel: torch.Tensor, **kwargs) -> torch.Tensor:
    """
    Copied from https://github.com/Project-MONAI/MONAI/blob/dev/monai/networks/layers/simplelayers.py

    Filtering `x` with `kernel` independently for each batch and channel respectively.

    Args:
        x: the input image, must have shape (batch, channels, H[, W, D]).
        kernel: `kernel` must at least have the spatial shape (H_k[, W_k, D_k]).
            `kernel` shape must be broadcastable to the `batch` and `channels` dimensions of `x`.
        kwargs: keyword arguments passed to `conv*d()` functions.

    Returns:
        The filtered `x`.

    Examples:

    .. code-block:: python

        >>> import torch
        >>> from monai.networks.layers import apply_filter
        >>> img = torch.rand(2, 5, 10, 10)  # batch_size 2, channels 5, 10x10 2D images
        >>> out = apply_filter(img, torch.rand(3, 3))   # spatial kernel
        >>> out = apply_filter(img, torch.rand(5, 3, 3))  # channel-wise kernels
        >>> out = apply_filter(img, torch.rand(2, 5, 3, 3))  # batch-, channel-wise kernels

    """
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"x must be a torch.Tensor but is {type(x).__name__}.")
    batch, chns, *spatials = x.shape
    n_spatial = len(spatials)
    if n_spatial > 3:
        raise NotImplementedError(f"Only spatial dimensions up to 3 are supported but got {n_spatial}.")
    k_size = len(kernel.shape)
    if k_size < n_spatial or k_size > n_spatial + 2:
        raise ValueError(
            f"kernel must have {n_spatial} ~ {n_spatial + 2} dimensions to match the input shape {x.shape}."
        )
    kernel = kernel.to(x)
    # broadcast kernel size to (batch chns, spatial_kernel_size)
    kernel = kernel.expand(batch, chns, *kernel.shape[(k_size - n_spatial) :])
    kernel = kernel.reshape(-1, 1, *kernel.shape[2:])  # group=1
    x = x.view(1, kernel.shape[0], *spatials)
    conv = [F.conv1d, F.conv2d, F.conv3d][n_spatial - 1]
    if "padding" not in kwargs:
        kwargs["padding"] = "same"

    if "stride" not in kwargs:
        kwargs["stride"] = 1
    output = conv(x, kernel, groups=kernel.shape[0], bias=None, **kwargs)
    return output.view(batch, chns, *output.shape[2:])
    
### Image from segmentation augmentation

class ImageFromSegTransform(BasicTransform):
    def __init__(self, classes=None, leave_background=0.5, retain_stats=False):
        super().__init__()
        self.classes = classes
        self.leave_background = leave_background
        self.retain_stats = retain_stats

    def get_parameters(self, **data_dict) -> dict:
        return {
            'classes': self.classes,
            'leave_background': self.leave_background,
            'retain_stats': self.retain_stats
        }
    
    def apply(self, data_dict: dict, **params) -> dict:
        if data_dict.get('image') is not None and data_dict.get('segmentation') is not None:
            data_dict['image'], data_dict['segmentation'] = self._apply_to_image(data_dict['image'], data_dict['segmentation'], **params)
        return data_dict

    def _apply_to_image(self, img: torch.Tensor, seg: torch.Tensor, **params) -> torch.Tensor: 
        img, seg = aug_labels2image(img, seg, classes=params['classes'], leave_background=params['leave_background'], retain_stats=params['retain_stats'])
        return img, seg

def aug_labels2image(img, seg, classes=None, leave_background=0.5, retain_stats=False):
    device = img.device
    _seg = seg
    if classes:
        _seg = combine_classes(seg, classes)
    
    subject = tio.RandomLabelsToImage(label_key="seg", image_key="image")(tio.Subject(
        seg=tio.LabelMap(tensor=_seg)
    ))
    new_img = subject.image.data

    if torch.rand(1, device=device) < leave_background:
        if retain_stats:
            img_mean, img_std = img.mean(), img.std()

        img_min, img_max = img.min(), img.max()
        _img = (img - img_min) / (img_max - img_min)

        new_img_min, new_img_max = new_img.min(), new_img.max()
        new_img = (new_img - new_img_min) / (new_img_max - new_img_min)
        new_img[_seg == 0] = _img[_seg == 0]

        if retain_stats:
            # Return to original range
            mean = torch.mean(new_img)
            std = torch.std(new_img)
            new_img = (new_img - mean)/torch.clamp(std, min=1e-7)
            new_img = new_img*img_std + img_mean
    return new_img, seg

### Redistribute segmentation values
    
class RedistributeTransform(BasicTransform):
    def __init__(self, classes=None, in_seg=0.2, retain_stats=False):
        super().__init__()
        self.classes = classes
        self.in_seg = in_seg
        self.retain_stats = retain_stats

    def get_parameters(self, **data_dict) -> dict:
        return {
            'classes': self.classes,
            'in_seg': self.in_seg,
            'retain_stats': self.retain_stats
        }
    
    def apply(self, data_dict: dict, **params) -> dict:
        if data_dict.get('image') is not None and data_dict.get('segmentation') is not None:
            data_dict['image'], data_dict['segmentation'] = self._apply_to_image(data_dict['image'], data_dict['segmentation'], **params)
        return data_dict

    def _apply_to_image(self, img: torch.Tensor, seg: torch.Tensor, **params) -> torch.Tensor:
        for c in range(img.shape[0]): 
            img[c], seg[c] = aug_redistribute_seg(img[c], seg[c], classes=params['classes'], in_seg=params['in_seg'], retain_stats=params['retain_stats'])
        return img, seg

def aug_redistribute_seg(img, seg, classes=None, in_seg=0.2, retain_stats=False):
    """
    Augment the image by redistributing the values of the image within the
    regions defined by the segmentation.
    """
    device = img.device
    _seg = seg
    in_seg_bool = 1 - torch.rand(1, device=device) <= in_seg

    if classes:
        _seg = combine_classes(_seg, classes)
    
    if retain_stats:
        # Compute original mean, std and min/max values
        original_mean, original_std = img.mean(), img.std()

    # Normalize image
    img_min, img_max = img.min(), img.max()
    img = (img - img_min) / (img_max - img_min)

    # Get the unique label values (excluding 0)
    labels = torch.unique(_seg)
    labels = labels[labels != 0]

    to_add = torch.zeros_like(img, device=device)

    # Loop over each label value
    for l in labels:
        # Get the mask for the current label
        l_mask = (_seg == l)

        # Get mean and std of the current label
        l_mean, l_std = img[l_mask].mean(), img[l_mask].std()

        # Convert to NumPy for dilation operations (not supported in PyTorch)
        l_mask_np = l_mask.cpu().numpy()
        struct = ndi.iterate_structure(ndi.generate_binary_structure(3, 1), 3)
        l_mask_dilate_np = ndi.binary_dilation(l_mask_np, structure=struct)

        # Convert back to PyTorch
        l_mask_dilate = torch.tensor(l_mask_dilate_np, device=device)

        # Create mask of the dilated mask excluding the original mask
        l_mask_dilate_excl = l_mask_dilate & ~l_mask

        # Compute mean and std for the dilated region
        if l_mask_dilate_excl.any():
            l_mean_dilate = img[l_mask_dilate_excl].mean()
            l_std_dilate = img[l_mask_dilate_excl].std()
        else:
            l_mean_dilate, l_std_dilate = l_mean, l_std  # Fallback to original values
        
        redist_std = max(torch.rand(1, device=device) * 0.2 + 0.4 * abs((l_mean - l_mean_dilate) * l_std / (l_std_dilate + 1e-6)), torch.tensor([0.01], device=device))

        redist = partial(norm.pdf, loc=l_mean.cpu().numpy(), scale=redist_std.cpu().numpy())

        if in_seg_bool:
            to_add[l_mask] += torch.tensor(redist(img[l_mask].cpu().numpy()), device=device) * (2 * torch.rand(1, device=device) - 1)
        else:
            to_add += torch.tensor(redist(img.cpu().numpy()), device=device) * (2 * torch.rand(1, device=device) - 1)

    # Normalize to_add and apply it to the image
    to_add_min, to_add_max = to_add.min(), to_add.max()
    img += 2 * (to_add - to_add_min) / (to_add_max - to_add_min + 1e-6)

    if retain_stats:
        # Return to original range
        mean = torch.mean(img)
        std = torch.std(img)
        img = (img - mean)/torch.clamp(std, min=1e-7)
        img = img*original_std + original_mean

    return img, seg

def combine_classes(seg, classes):
    _seg = torch.zeros_like(seg)
    for i, c in enumerate(classes):
        _seg[torch.isin(seg, c)] = i + 1
    return _seg

### Artifacts augmentation (Motion, Ghosting, Spike, Bias Field, Blur, Noise, Swap)

class ArtifactTransform(BasicTransform):
    def __init__(self, motion=False, ghosting=False, spike=False, bias_field=False, blur=False, noise=False, swap=False, random_pick=False):
        '''
        Apply all selected artifacts (motion, ghosting, spike, bias field, blur, noise, and swap) to the image if they are enabled (set to True).  
        If `random_pick` is True, randomly select and apply ONE of the enabled artifacts.
        '''
        super().__init__()
        self.motion = motion
        self.ghosting = ghosting
        self.spike = spike
        self.bias_field = bias_field
        self.blur = blur
        self.noise = noise
        self.swap = swap
        self.random_pick = random_pick

    def get_parameters(self, **data_dict) -> dict:

        artifacts = {
            "motion": self.motion,
            "ghosting": self.ghosting,
            "spike": self.spike,
            "bias_field": self.bias_field,
            "blur": self.blur,
            "noise": self.noise,
            "swap": self.swap
        }

        enabled_artifacts = {k:v for k,v in artifacts.items() if v}

        if self.random_pick and enabled_artifacts:
            selected_artifact = random.choice(list(enabled_artifacts.keys()))
            artifacts = {k: (k == selected_artifact) for k,v in artifacts.items()}

        return artifacts
    
    def apply(self, data_dict: dict, **params) -> dict:
        if data_dict.get('image') is not None and data_dict.get('segmentation') is not None:
            data_dict['image'], data_dict['segmentation'] = self._apply_to_image(data_dict['image'], data_dict['segmentation'], **params)
        return data_dict

    def _apply_to_image(self, img: torch.Tensor, seg: torch.Tensor, **params) -> torch.Tensor:
        if params['motion']:
            img, seg = aug_motion(img, seg)
        if params['ghosting']:
            img, seg = aug_ghosting(img, seg)
        if params['spike']:
            img, seg = aug_spike(img, seg)
        if params['bias_field']:
            img, seg = aug_bias_field(img, seg)
        if params['blur']:
            img, seg = aug_blur(img, seg)
        if params['noise']:
            img, seg = aug_noise(img, seg)
        if params['swap']:
            img, seg = aug_swap(img, seg)
        return img, seg

def aug_motion(img, seg):
    subject = tio.RandomMotion()(tio.Subject(
        image=tio.ScalarImage(tensor=img),
        seg=tio.LabelMap(tensor=seg)
    ))
    img_out, seg_out = subject.image.data, subject.seg.data
    del subject
    gc.collect()  # Force garbage collection
    return img_out, seg_out

def aug_ghosting(img, seg):
    subject = tio.RandomGhosting()(tio.Subject(
        image=tio.ScalarImage(tensor=img),
        seg=tio.LabelMap(tensor=seg)
    ))
    img_out, seg_out = subject.image.data, subject.seg.data
    del subject
    gc.collect()  # Force garbage collection
    return img_out, seg_out

def aug_spike(img, seg):
    subject = tio.RandomSpike(intensity=(1, 2))(tio.Subject(
        image=tio.ScalarImage(tensor=img),
        seg=tio.LabelMap(tensor=seg)
    ))
    img_out, seg_out = subject.image.data, subject.seg.data
    del subject
    gc.collect()  # Force garbage collection
    return img_out, seg_out

def aug_bias_field(img, seg):
    subject = tio.RandomBiasField()(tio.Subject(
        image=tio.ScalarImage(tensor=img),
        seg=tio.LabelMap(tensor=seg)
    ))
    img_out, seg_out = subject.image.data, subject.seg.data
    del subject
    gc.collect()  # Force garbage collection
    return img_out, seg_out

def aug_blur(img, seg):
    subject = tio.RandomBlur()(tio.Subject(
        image=tio.ScalarImage(tensor=img),
        seg=tio.LabelMap(tensor=seg)
    ))
    img_out, seg_out = subject.image.data, subject.seg.data
    del subject
    gc.collect()  # Force garbage collection
    return img_out, seg_out

def aug_noise(img, seg):
    original_mean, original_std = img.mean(), img.std()
    img = (img - original_mean) / original_std
    subject = tio.RandomNoise()(tio.Subject(
        image=tio.ScalarImage(tensor=img),
        seg=tio.LabelMap(tensor=seg)
    ))
    img = img  * original_std + original_mean
    img_out, seg_out = subject.image.data, subject.seg.data
    del subject
    gc.collect()  # Force garbage collection
    return img_out, seg_out

def aug_swap(img, seg):
    subject = tio.RandomSwap()(tio.Subject(
        image=tio.ScalarImage(tensor=img),
        seg=tio.LabelMap(tensor=seg)
    ))
    img_out, seg_out = subject.image.data, subject.seg.data
    del subject
    gc.collect()  # Force garbage collection
    return img_out, seg_out

### Spatial augmentation (Flip, Affine, Elastic, Anisotropy) --> removed BSpline

class SpatialCustomTransform(BasicTransform):
    def __init__(self, flip=False, affine=False, elastic=False, anisotropy=False, random_pick=False):
        '''
        Apply all selected spatial transformation (flip, affine, elastic and anisotropy) to the image if they are enabled (set to True).  
        If `random_pick` is True, randomly select and apply ONE of the enabled transformation.
        '''
        super().__init__()
        self.flip = flip
        self.affine = affine
        self.elastic = elastic
        self.anisotropy = anisotropy
        self.random_pick = random_pick

    def get_parameters(self, **data_dict) -> dict:
        transfo = {
            "flip" : self.flip,
            "affine" : self.affine,
            "elastic" : self.elastic,
            "anisotropy" : self.anisotropy
        }

        enabled_transfo = {k:v for k,v in transfo.items() if v}

        if self.random_pick and enabled_transfo:
            selected_transfo = random.choice(list(enabled_transfo.keys()))
            transfo = {k: (k == selected_transfo) for k,v in transfo.items()}
        
        return transfo
    
    def apply(self, data_dict: dict, **params) -> dict:
        if data_dict.get('image') is not None and data_dict.get('segmentation') is not None:
            data_dict['image'], data_dict['segmentation'] = self._apply_to_image(data_dict['image'], data_dict['segmentation'], **params)
        return data_dict

    def _apply_to_image(self, img: torch.Tensor, seg: torch.Tensor, **params) -> torch.Tensor:
        if params['flip']:
            img, seg = aug_flip(img, seg)
        if params['affine']:
            img, seg = aug_affine(img, seg)
        if params['elastic']:
            img, seg = aug_elastic(img, seg)
        if params['anisotropy']:
            img, seg = aug_anisotropy(img, seg)
        return img, seg

def aug_flip(img, seg):
    subject = tio.RandomFlip(axes=('LR',))(tio.Subject(
        image=tio.ScalarImage(tensor=img),
        seg=tio.LabelMap(tensor=seg)
    ))
    img_out, seg_out = subject.image.data, subject.seg.data
    del subject
    gc.collect()  # Force garbage collection
    return img_out, seg_out

def aug_affine(img, seg):
    subject = tio.RandomAffine()(tio.Subject(
        image=tio.ScalarImage(tensor=img),
        seg=tio.LabelMap(tensor=seg)
    ))
    img_out, seg_out = subject.image.data, subject.seg.data
    del subject
    gc.collect()  # Force garbage collection
    return img_out, seg_out

def aug_elastic(img, seg):
    subject = tio.RandomElasticDeformation(max_displacement=40)(tio.Subject(
        image=tio.ScalarImage(tensor=img),
        seg=tio.LabelMap(tensor=seg)
    ))
    img_out, seg_out = subject.image.data, subject.seg.data
    del subject
    gc.collect()  # Force garbage collection
    return img_out, seg_out

def aug_anisotropy(img, seg, downsampling=7):
    subject = tio.RandomAnisotropy(downsampling=downsampling)(tio.Subject(
        image=tio.ScalarImage(tensor=img),
        seg=tio.LabelMap(tensor=seg, axis=0)
    ))
    img_out, seg_out = subject.image.data, subject.seg.data
    del subject
    gc.collect()  # Force garbage collection
    return img_out, seg_out


