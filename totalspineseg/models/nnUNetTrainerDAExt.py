'''
Based on https://github.com/neuropoly/AugLab/blob/main/auglab/trainers/nnUNetTrainerDAExt.py
'''

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from typing import Tuple, Union, List
import numpy as np
from batchgeneratorsv2.helpers.scalar_type import RandomScalar
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from batchgeneratorsv2.transforms.nnunet.random_binary_operator import ApplyRandomBinaryOperatorTransform
from batchgeneratorsv2.transforms.nnunet.remove_connected_components import RemoveRandomConnectedComponentFromOneHotEncodingTransform
from batchgeneratorsv2.transforms.nnunet.seg_to_onehot import MoveSegAsOneHotToDataTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from batchgeneratorsv2.transforms.utils.deep_supervision_downsampling import DownsampleSegForDSTransform
from batchgeneratorsv2.transforms.utils.nnunet_masking import MaskImageTransform
from batchgeneratorsv2.transforms.utils.random import RandomTransform
from batchgeneratorsv2.transforms.utils.remove_label import RemoveLabelTansform
from batchgeneratorsv2.transforms.utils.seg_to_regions import ConvertSegmentationToRegionsTransform
from batchgeneratorsv2.transforms.utils.pseudo2d import Convert3DTo2DTransform, Convert2DTo3DTransform
from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform

import os
import torch
import importlib
from torch import autocast
import json
import shutil
import random

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.helpers import dummy_context

import auglab.configs as configs
from auglab.transforms.gpu.transforms import AugTransformsGPU
from auglab.trainers.utils import DownsampleSegForDSTransformCustom

from totalspineseg.utils.augment import aug_flip, aug_affine, aug_elastic, aug_anisotropy


class nnUNetTrainerDAExtGPU(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)

        self.num_epochs = 1000

        # Load transform parameters from json file
        configs_path = importlib.resources.files(configs)
        json_path = os.environ.get("AUGLAB_PARAMS_GPU_JSON", str(configs_path / "transform_params_gpu.json"))
        self.transforms = AugTransformsGPU(json_path=json_path).to(self.device)
        print(f'Using AugLab GPU transforms with parameters from: {json_path}')

        # Copy json transfrom parameters to output folder
        shutil.copy(
            json_path,
            os.path.join(self.output_folder, 'transform_params_gpu_used_for_training.json')
        )

    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes = \
            super().configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        # Remove mirroring
        mirror_axes = None
        self.inference_allowed_mirroring_axes = None
        return rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes

    @staticmethod
    def get_training_transforms(
            patch_size: Union[np.ndarray, Tuple[int]],
            rotation_for_DA: RandomScalar,
            deep_supervision_scales: Union[List, Tuple, None],
            mirror_axes: Tuple[int, ...],
            do_dummy_2d_data_aug: bool,
            use_mask_for_norm: List[bool] = None,
            is_cascaded: bool = False,
            foreground_labels: Union[Tuple[int, ...], List[int]] = None,
            regions: List[Union[List[int], Tuple[int, ...], int]] = None,
            ignore_label: int = None,
            retain_stats: bool = False
    ) -> BasicTransform:
        transforms = []

        configs_path = importlib.resources.files(configs)
        json_path = os.environ.get("AUGLAB_PARAMS_GPU_JSON", str(configs_path / "transform_params_gpu.json"))
        with open(json_path, 'r') as f:
            config = json.load(f)

        ## Spatial transforms
        # Flip, Affine, Elastic, Anisotropy
        transforms.append(RandomTransform(
            SpatialCustomTransform(
                flip=False,
                affine=False,
                elastic=True,
                anisotropy=True,
                random_pick=True
            ), apply_probability=0.5
        ))

        ### Keep some nnunet transforms
        if do_dummy_2d_data_aug:
            ignore_axes = (0,)
            transforms.append(Convert3DTo2DTransform())
            patch_size_spatial = patch_size[1:]
        else:
            patch_size_spatial = patch_size
            ignore_axes = None

        if 'nnUNetSpatialTransform' in config:
            spatial_params = config['nnUNetSpatialTransform']
        else:
            spatial_params = {}

        transforms.append(
            SpatialTransform(
                patch_size_spatial, 
                patch_center_dist_from_border=spatial_params.get('patch_center_dist_from_border', 0),
                random_crop=spatial_params.get('random_crop', False),
                p_elastic_deform=spatial_params.get('p_elastic_deform', 0),
                p_rotation=spatial_params.get('p_rotation', 0),
                rotation=rotation_for_DA, 
                p_scaling=spatial_params.get('p_scaling', 0), 
                scaling=spatial_params.get('scaling', (0.7, 1.4)), 
                p_synchronize_scaling_across_axes=spatial_params.get('p_synchronize_scaling_across_axes', 1),
                bg_style_seg_sampling=False, 
                mode_seg='nearest'
            )
        )

        if do_dummy_2d_data_aug:
            transforms.append(Convert2DTo3DTransform())

        if use_mask_for_norm is not None and any(use_mask_for_norm):
            transforms.append(MaskImageTransform(
                apply_to_channels=[i for i in range(len(use_mask_for_norm)) if use_mask_for_norm[i]],
                channel_idx_in_seg=0,
                set_outside_to=0,
            ))

        transforms.append(
            RemoveLabelTansform(-1, 0)
        )

        # The following augmentations are related to special nnunet executions
        if is_cascaded:
            assert foreground_labels is not None, 'We need foreground_labels for cascade augmentations'
            transforms.append(
                MoveSegAsOneHotToDataTransform(
                    source_channel_idx=1,
                    all_labels=foreground_labels,
                    remove_channel_from_source=True
                )
            )
            transforms.append(
                RandomTransform(
                    ApplyRandomBinaryOperatorTransform(
                        channel_idx=list(range(-len(foreground_labels), 0)),
                        strel_size=(1, 8),
                        p_per_label=1
                    ), apply_probability=0.4
                )
            )
            transforms.append(
                RandomTransform(
                    RemoveRandomConnectedComponentFromOneHotEncodingTransform(
                        channel_idx=list(range(-len(foreground_labels), 0)),
                        fill_with_other_class_p=0,
                        dont_do_if_covers_more_than_x_percent=0.15,
                        p_per_label=1
                    ), apply_probability=0.2
                )
            )

        if regions is not None:
            # the ignore label must also be converted
            transforms.append(
                ConvertSegmentationToRegionsTransform(
                    regions=list(regions) + [ignore_label] if ignore_label is not None else regions,
                    channel_in_seg=0
                )
            )

        # transforms.append(ZscoreNormalization())

        # NOTE: DownsampleSegForDSTransform is now handled in train_step for GPU augmentations
        # if deep_supervision_scales is not None:
        #     transforms.append(DownsampleSegForDSTransform(ds_scales=deep_supervision_scales))

        return ComposeTransforms(transforms)

    @staticmethod
    def get_validation_transforms(
            deep_supervision_scales: Union[List, Tuple, None],
            is_cascaded: bool = False,
            foreground_labels: Union[Tuple[int, ...], List[int]] = None,
            regions: List[Union[List[int], Tuple[int, ...], int]] = None,
            ignore_label: int = None,
    ) -> BasicTransform:
        transforms = []
        transforms.append(
            RemoveLabelTansform(-1, 0)
        )

        if is_cascaded:
            transforms.append(
                MoveSegAsOneHotToDataTransform(
                    source_channel_idx=1,
                    all_labels=foreground_labels,
                    remove_channel_from_source=True
                )
            )

        if regions is not None:
            # the ignore label must also be converted
            transforms.append(
                ConvertSegmentationToRegionsTransform(
                    regions=list(regions) + [ignore_label] if ignore_label is not None else regions,
                    channel_in_seg=0
                )
            )

        # transforms.append(ZscoreNormalization())

        if deep_supervision_scales is not None:
            transforms.append(DownsampleSegForDSTransform(ds_scales=deep_supervision_scales))
        return ComposeTransforms(transforms)

    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        # Now target should be a single tensor, not a list
        target = target.to(self.device, non_blocking=True)
        # if isinstance(target, list):
        #     target = [i.to(self.device, non_blocking=True) for i in target]
        # else:
        #     target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():            
            # Apply GPU augmentations to full-resolution data/target
            data, target = self.transforms(data, target)

            # Create multi-scale targets for deep supervision after augmentation
            deep_supervision_scales = self._get_deep_supervision_scales()
            if deep_supervision_scales is not None:
                ds_transform = DownsampleSegForDSTransformCustom(ds_scales=deep_supervision_scales)
                target = ds_transform(target)

            output = self.network(data)
            # del data
            l = self.loss(output, target)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        return {'loss': l.detach().cpu().numpy()}


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

