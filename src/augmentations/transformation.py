# build-in libs
# thrid-party libs
from typing import Dict
import numpy as np
from albumentations import (
    Compose, ShiftScaleRotate, RandomBrightness, HueSaturationValue, RandomContrast, HorizontalFlip,VerticalFlip,
    Rotate, Resize, CLAHE, ColorJitter, RandomBrightnessContrast, GaussianBlur, Blur, MedianBlur,
    Downscale, ChannelShuffle, Normalize, OneOf, GaussNoise,
    RandomScale,Equalize,HistogramMatching,RandomCrop
)
from albumentations.augmentations.crops import functional as CropsF
from albumentations.core.transforms_interface import DualTransform,ImageOnlyTransform
from albumentations.pytorch import ToTensorV2
from numpy import random
from src.config import BaseConfigs
from src.enums import AugmentationTypes, DatasetTypes, DictKeys, SegmentationClassNames

import os
import cv2
import numpy as np




"""
====================
Augmentation Pipelines
====================
"""
def get_heavy_transform_pipeline(config:BaseConfigs, is_train=True):
    """
    apply heavy augmentations 

    - One of
        - ShiftScaleRotate
        - Rotate
        - RandomScale
    - HorizontalFlip
    - VerticalFlip
    - GaussNoise
    - One of
        - Blur
        - GaussianBlur
        - MedianBlur
    - One of
        - One of
            - RandomBrightness
            - RandomContrast
        - One of
            - CLAHE
            - ColorJitter
            - HueSaturationValue
            - ChannelShuffle
    - Downscale
    - Resize
    - ConvertTarget
    - Normalization
    - ToTensor

    :param config: segmentation configurations 
    :type config: SegmentationConfigs
    :param is_train: check if the pipeline specificed for training of inference, defaults to True
    :type is_train: bool, optional
    :return: composation of transformation functions
    :rtype: albumentations.Compose
    """
    class_labels_dict=config.class_labels_dict
    class_color_dict=config.class_color_dict
    label_dict={}
    for key in class_color_dict.keys():
        label_dict[class_color_dict[key]]=class_labels_dict[key]
    additional_targets=None
    height,width=config.train_img_height,config.train_img_width


    if(is_train):
        return Compose([
            Compose([
            # AdaptiveCutmix(weed_cropplets_data_dir=weed_cropplets_data_dir,n_weeds=cutmix_n,p=random.uniform(low=config.aug_bounds[0],high=config.aug_bounds[1])),
            OneOf([
                ShiftScaleRotate(rotate_limit=15, p=random.uniform(low=config.aug_bounds[0],high=config.aug_bounds[1])),
                Rotate(limit=15, p=random.uniform(low=config.aug_bounds[0],high=config.aug_bounds[1])),
                RandomScale(p=random.uniform(low=config.aug_bounds[0],high=config.aug_bounds[1])),
            ], p=random.uniform(low=config.aug_bounds[0],high=config.aug_bounds[1])),
            HorizontalFlip(p=random.uniform(low=config.aug_bounds[0],high=config.aug_bounds[1])),
            VerticalFlip(p=random.uniform(low=config.aug_bounds[0],high=config.aug_bounds[1])),
            GaussNoise(p=random.uniform(low=config.aug_bounds[0],high=config.aug_bounds[1])),
            OneOf([
                Blur(p=random.uniform(low=config.aug_bounds[0],high=config.aug_bounds[1])),
                GaussianBlur(p=random.uniform(low=config.aug_bounds[0],high=config.aug_bounds[1])),
                MedianBlur(p=random.uniform(low=config.aug_bounds[0],high=config.aug_bounds[1])),
            ], p=random.uniform(low=config.aug_bounds[0],high=config.aug_bounds[1])),
            OneOf([
                OneOf([
                    RandomBrightness(p=random.uniform(low=config.aug_bounds[0],high=config.aug_bounds[1])),
                    RandomContrast(p=random.uniform(low=config.aug_bounds[0],high=config.aug_bounds[1])),
                ], p=random.uniform(low=config.aug_bounds[0],high=config.aug_bounds[1])),#p=0.75),
                OneOf([
                    CLAHE(p=random.uniform(low=config.aug_bounds[0],high=config.aug_bounds[1])),
                    ColorJitter(p=random.uniform(low=config.aug_bounds[0],high=config.aug_bounds[1])),
                    HueSaturationValue(p=random.uniform(low=config.aug_bounds[0],high=config.aug_bounds[1])),
                    ChannelShuffle(p=random.uniform(low=config.aug_bounds[0],high=config.aug_bounds[1])),
                ], p=random.uniform(low=config.aug_bounds[0],high=config.aug_bounds[1])),#p=0.75),
            ],p=random.uniform(low=config.aug_bounds[0],high=config.aug_bounds[1])),#0.5),
            Downscale(scale_min=0.25, scale_max=0.25, p=random.uniform(low=config.aug_bounds[0],high=config.aug_bounds[1])),
            
                
            
        ],additional_targets=additional_targets),
        Resize(height=height, width=width),
            # Normalize(),
        Compose([
            ConvertTarget(label_dict=label_dict,always_apply=True),
            Normalization(always_apply=True),
            ToTensorV2()
        ],additional_targets=additional_targets)
        
        ])
    else:
        return Compose([
            Resize(height=height, width=width),
            # Normalize(),

            Compose([
                ConvertTarget(label_dict=label_dict,always_apply=True),
                Normalization(always_apply=True),
                ToTensorV2()
            ],additional_targets=additional_targets)
        

        ])

def get_light_transform_pipeline(config:BaseConfigs, is_train=True):
    """
    apply light augmentations 

    - Rotate
    - HorizontalFlip
    - Downscale
    - Resize
    - ConvertTarget
    - Normalization
    - ToTensor

    :param config: segmentation configurations 
    :type config: SegmentationConfigs
    :param is_train: check if the pipeline specificed for training of inference, defaults to True
    :type is_train: bool, optional
    :return: composation of transformation functions
    :rtype: albumentations.Compose
    """
    class_labels_dict=config.class_labels_dict
    class_color_dict=config.class_color_dict
    label_dict={}
    for key in class_color_dict.keys():
        label_dict[class_color_dict[key]]=class_labels_dict[key]
    additional_targets=None
    height,width=config.train_img_height,config.train_img_width
    if(is_train):
        return Compose([
            # AdaptiveCutmix(weed_cropplets_data_dir=weed_cropplets_data_dir,n_weeds=cutmix_n,p=random.uniform(low=config.aug_bounds[0],high=config.aug_bounds[1])),
            Compose([
                Rotate(limit=15, p=random.uniform(low=config.aug_bounds[0],high=config.aug_bounds[1])),
                HorizontalFlip(p=random.uniform(low=config.aug_bounds[0],high=config.aug_bounds[1])),
                Downscale(),

            ],additional_targets=additional_targets),
            Resize(height=height, width=width),
            # Normalize(),
            Compose([
                ConvertTarget(label_dict=label_dict,always_apply=True),
                Normalization(always_apply=True),
                ToTensorV2()

            ],additional_targets=additional_targets)
        ])
    else:
        return Compose([
            Resize(height=height, width=width),
            # Normalize(),

            Compose([
                ConvertTarget(label_dict=label_dict,always_apply=True),
                Normalization(always_apply=True),
                ToTensorV2()
            ],additional_targets=additional_targets)
        

        ])

def get_transform_pipeline(config:BaseConfigs, is_train=True):
    """
    apply no augmentations 

    - Resize
    - ConvertTarget
    - Normalization
    - ToTensor

    :param config: segmentation configurations 
    :type config: SegmentationConfigs
    :param is_train: check if the pipeline specificed for training of inference, defaults to True
    :type is_train: bool, optional
    :return: composation of transformation functions
    :rtype: albumentations.Compose
    """
    class_labels_dict=config.class_labels_dict
    class_color_dict=config.class_color_dict
    label_dict={}
    for key in class_color_dict.keys():
        try:
            label_dict[tuple(class_color_dict[key])]=class_labels_dict[key]
        except:
            label_dict[class_color_dict[key]]=class_labels_dict[key]
    additional_targets=None
    height,width=config.train_img_height,config.train_img_width

    return Compose([
        Resize(height=height,width=width),
        Compose([
            ConvertTarget(label_dict=label_dict,always_apply=True),
            Normalization(always_apply=True),
            ToTensorV2()
        ],additional_targets=additional_targets)
        
    ])


"""
====================
Augmentation Implementations
====================
"""

# It takes a dictionary of RGB values and their corresponding label values and converts the mask to a
# label mask
class ConvertTarget(DualTransform):
    def __init__(self,label_dict:dict, always_apply: bool = True, p: float = 1):
        """
        This function takes in a dictionary of labels and their corresponding values and returns a
        dictionary of labels and their corresponding values
        
        :param label_dict: A dictionary of the form {'label_name': 'label_id'}
        :type label_dict: dict
        :param always_apply: If set to True, the transform will be applied to all the images in the
        dataset. If set to False, the transform will only be applied to images that have a matching key
        in the label_dict, defaults to True
        :type always_apply: bool (optional)
        :param p: probability of applying the transform. Default: 1.0, defaults to 1
        :type p: float (optional)
        """
        super().__init__(always_apply, p)
        self.label_dict=label_dict
    
    def apply(self, img, **params) :
        return img
    def apply_to_mask(self, mask, **params):
        new_mask=np.zeros((mask.shape[0],mask.shape[1]),dtype=float)
        for pixel_value,label_value in self.label_dict.items():
            if(isinstance(pixel_value,tuple)):
                new_mask[np.where(np.all(mask == pixel_value[::-1], axis=-1))] = label_value
            elif(isinstance(pixel_value,int)):
                new_mask[mask==pixel_value] = label_value
        return new_mask

class Normalization(ImageOnlyTransform):
    """
    this class normalize the input image by dividing image values by 255

    """
    def __init__(self, always_apply: bool = True, p: float = 1):
        super().__init__(always_apply, p)
    
    def apply(self, img, **params) :
        return img/255.0


class SpeckleNoise(ImageOnlyTransform):
    def __init__(self,std=0.4,scale=0.3, always_apply: bool = False, p: float = 0.5):
        super().__init__(always_apply, p)
        self.std=std
        self.scale=scale
        
    def apply(self, img, **params) :

        row, col, ch = img.shape
        gauss = np.random.randn(row, col, ch)
        gauss = gauss.reshape(row, col, ch)
        noisy_img = img + (np.random.normal(self.std,self.scale) * img * gauss)
        noisy_img_clipped = np.clip(noisy_img, 0, 255)
        noisy_img_clipped = noisy_img_clipped.astype(np.uint8)
        return noisy_img_clipped
class PoisonNoise(ImageOnlyTransform):
    def __init__(self, always_apply: bool = False, p: float = 0.5):
        super().__init__(always_apply, p)
    
    def apply(self, img, **params) :

        vals = len(np.unique(img))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy_img = np.random.poisson(img * vals) / float(vals)
        noisy_img_clipped = np.clip(noisy_img, 0, 255)
        noisy_img_clipped = noisy_img_clipped.astype(np.uint8)
        return noisy_img_clipped

class SaltPepperNoise(ImageOnlyTransform):
    def __init__(self,s_vs_p=0.9,amount=0.05, always_apply: bool = False, p: float = 0.5):
        super().__init__(always_apply, p)
        self.s_vs_p = s_vs_p
        self.amount=amount
    def apply(self, img, **params) :

        row, col, ch = img.shape
        
        out = np.copy(img)
        # Salt mode
        num_salt = np.ceil(self.amount * img.size * self.s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in img.shape]
        out[tuple(coords)] = 1

        # Pepper mode
        num_pepper = np.ceil(self.amount * img.size * (1. - self.s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in img.shape]
        out[tuple(coords)] = 0
        # print(np.unique(out))
        return out

