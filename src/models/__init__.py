from src.models.models import *
from src.models.base_model import BaseModel
from src.config import BaseConfigs
from src.enums import ModelTypes,ConfigTypes
from warnings import warn


"""
model builder function, that get the correct model specificed in the config class
:param config: object of the config class
:type config: BaseConfigs
:return: src.models.base_model.BaseModel
"""
def get_model(config:BaseConfigs) -> BaseModel:
    model=None
    if(config.model_type==ModelTypes.UNET):
        model= UNet(n_classes=config.out_channels,batchnorm=config.batch_norm)
    if(config.model_type==ModelTypes.UNET_LIGHT):
        model= UNetLight(n_classes=config.out_channels,batchnorm=config.batch_norm)
    elif(config.model_type==ModelTypes.CONDITIONAL_GUIDED_UNET_OUT_MASK_OUT_CLASS):
        model=ConditionalGuidedUNetOutMaskOutClass(n_classes=config.out_channels,n_organs=len(config.organ_labels_dict.keys()),batchnorm=config.batch_norm)
    elif(config.model_type == ModelTypes.CONDITIONAL_GUIDED_UNET):
        model=ConditionalGuidedUNetOutMaskInClass(n_classes=config.out_channels,n_organs=len(config.organ_labels_dict.keys()),batchnorm=config.batch_norm)
    else:
        raise NotImplementedError
    if(not((config.checkpoint_path is None) or (config.checkpoint_path == ''))):
        model.load_model(ckpt_path=config.checkpoint_path,device=config.device)
        print("CHECKPOINT LOADED...")
        print()

    if(config.config_type==ConfigTypes.EVALUTAION or config.config_type==ConfigTypes.INFERENCE):
        model.eval()
    model.to(config.device)
    return model
