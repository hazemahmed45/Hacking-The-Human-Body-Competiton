from src.augmentations.base_pipeline import BaseAugmentationPipeline
from src.augmentations.transformation import get_heavy_transform_pipeline, \
                                            get_light_transform_pipeline, \
                                            get_transform_pipeline
                                          
from src.config import BaseConfigs


class LightAugmentationPipeline(BaseAugmentationPipeline):
    """
    Augmentation pipeline for that call light transformation pipeline
    """
    def set_pipeline(self, config: BaseConfigs, is_train: bool):
        """
        set the transformation pipeline to the light transformation

        :param config: base configurations 
        :type config: BaseConfigs
        :param is_train: check if the pipeline specificed for training of inference, defaults to True
        :type is_train: bool
        :return: light transformation pipeline 
        :rtype: func
        """
        return get_light_transform_pipeline(config,is_train)
        
class HeavyAugmentationPipeline(BaseAugmentationPipeline):
    """
    Augmentation pipeline for that call heavy transformation pipeline
    """
    def set_pipeline(self, config: BaseConfigs, is_train: bool):
        """
        set the transformation pipeline to the heavy transformation

        :param config: base configurations 
        :type config: BaseConfigs
        :param is_train: check if the pipeline specificed for training of inference, defaults to True
        :type is_train: bool
        :return: heavy transformation pipeline 
        :rtype: func
        """
        return get_heavy_transform_pipeline(config,is_train)

class NoAugmentationPipeline(BaseAugmentationPipeline):
    """
    Augmentation pipeline for that call no transformation pipeline
    """
    def set_pipeline(self, config: BaseConfigs, is_train: bool):
        """
        set the transformation pipeline to the no transformation

        :param config: base configurations 
        :type config: BaseConfigs
        :param is_train: check if the pipeline specificed for training of inference, defaults to True
        :type is_train: bool
        :return: no transformation pipeline 
        :rtype: func
        """
        return get_transform_pipeline(config,is_train)

