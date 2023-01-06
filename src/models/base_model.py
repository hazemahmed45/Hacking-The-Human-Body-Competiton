from collections import OrderedDict
from torch.nn import Module
from abc import abstractmethod
import torch
from src.enums import DictKeys

class BaseModel(Module):
    """
    base model architecture
    """
    def get_state_dict(self):
        """
        return model state dictionary

        :return: model state dictionary
        :rtype: dict
        """
        state_dict=self.state_dict()
        # print(state_dict.keys())
        return state_dict
    def load_model(self,ckpt_path,device='cuda'):
        """
        load previously trained model checkpoint in the model architecture

        :param ckpt_path: path of the model checkpoint
        :type ckpt_path: str
        :param device: the device the tensor will be put on, defaults to 'cuda'
        :type device: str, optional
        """
        state_dict=torch.load(ckpt_path,map_location=device)
        if(DictKeys.MODEL_STATE_DICT.value in state_dict.keys()):
            state_dict=state_dict[DictKeys.MODEL_STATE_DICT.value]
        self.load_state_dict(state_dict,strict=True)
        return 
    @abstractmethod
    def unpack_kwargs(self,**kwargs):
        """
        unpacks the key word dictionary passed to the model to take the model required input
        """
        pass
    @abstractmethod
    def load_backbone(self,backbone_path,device='cuda'):
        """
        load model backbone, which loads the encoder of another model in the current encoder

        :param backbone_path: checkpoint path of the model to take the encoder from
        :type backbone_path: str
        :param device: the device the tensor will be put on, defaults to 'cuda'
        :type device: str, optional
        :raises NotImplementedError: if the model is not designed to load backbones
        """
        raise NotImplementedError('Does not support backbone loading')
    @abstractmethod
    def forward_step(self,**kwargs):
        """
        custom forward function to unpacks inputs and pack outputs
        """
        pass