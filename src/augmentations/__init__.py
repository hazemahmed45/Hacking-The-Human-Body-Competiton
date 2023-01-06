from typing import Dict
from src.config import  BaseConfigs
from src.enums import AugmentationTypes, DictKeys
from src.augmentations.pipeline import BaseAugmentationPipeline, \
                                                    NoAugmentationPipeline,\
                                                    LightAugmentationPipeline,\
                                                    HeavyAugmentationPipeline
                                                    


def get_augmentations(config:BaseConfigs)->Dict[str,BaseAugmentationPipeline]:
    """
    augmentation pipeline builder function, that get the correct augmentation pipeline specificed in the config class
    :param config: object of the config class
    :type config: BaseConfigs
    :return: return dictionary of split name to augmentation of the split
    :rtype: Dict[str,BaseAugmentationPipeline]
    """
    if(config.augmentation_type==AugmentationTypes.ALBUMENTATION_NONE):
        return {
            DictKeys.TRAIN.value:NoAugmentationPipeline(config,True),
            DictKeys.VALID.value:NoAugmentationPipeline(config,False),
            DictKeys.TEST.value:NoAugmentationPipeline(config,False)
            }
    elif(config.augmentation_type==AugmentationTypes.ALBUMENTATION_LIGHT):
        return {
            DictKeys.TRAIN.value:LightAugmentationPipeline(config,True),
            DictKeys.VALID.value:LightAugmentationPipeline(config,False),
            DictKeys.TEST.value:LightAugmentationPipeline(config,False)
            }
    elif(config.augmentation_type==AugmentationTypes.ALBUMENTATION_HEAVY):
        return {
            DictKeys.TRAIN.value:HeavyAugmentationPipeline(config,True),
            DictKeys.VALID.value:HeavyAugmentationPipeline(config,False),
            DictKeys.TEST.value:HeavyAugmentationPipeline(config,False)
            }
    
    else:
        raise NotImplementedError(str.format('{} augmentation type is not yet implemented in pipelines',config.augmentation_type.value))
    return 