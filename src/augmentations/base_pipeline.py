from abc import abstractmethod

from src.config import BaseConfigs


class BaseAugmentationPipeline():
    """
    Base Augmentation Pipeline , this base class is the building block for any augmentation or transformation pipeline

    """
    def __init__(self,config:BaseConfigs,is_train=True) :
        """
        base augmentation pipeline constructor

        :param config: base configurations 
        :type config: BaseConfigs
        :param is_train: check if the pipeline specificed for training of inference, defaults to True
        :type is_train: bool, optional
        """
        self.config=config
        self.is_train=is_train
        self.pipeline=self.set_pipeline(config,is_train)
    def __call__(self,**kwargs):
        return self.pipeline(**kwargs)
    @abstractmethod
    def set_pipeline(self,config:BaseConfigs,is_train:bool):
        """
        this abstract function is for setting the pipeline function that will be called in the call function

        :param config: base configurations 
        :type config: BaseConfigs
        :param is_train: check if the pipeline specificed for training of inference, defaults to True
        :type is_train: bool
        """
        pass