from src.config import BaseConfigs
from src.enums import OptimizerTypes,SchedularTypes
from src.models.base_model import BaseModel
from torch import optim
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import LambdaLR,\
                                    CosineAnnealingLR,\
                                    ExponentialLR,\
                                    StepLR,\
                                    ChainedScheduler,\
                                    MultiStepLR,\
                                    CosineAnnealingWarmRestarts,\
                                    ConstantLR,\
                                    CyclicLR,\
                                    LinearLR,\
                                    ReduceLROnPlateau,\
                                    SequentialLR


def get_optimizer(config:BaseConfigs,model:BaseModel) -> optim.Optimizer:
    """
    optimizer builder function, that get the correct optimizer specificed in the config class
    :param config: object of the config class
    :type config: BaseConfigs
    :return: torch.optim.Optimizer
    """
    optimizer=None
    if(config.optim_type==OptimizerTypes.ADAM):
        optimizer= optim.Adam(params=model.parameters(),lr=config.lr)
    elif(config.optim_type == OptimizerTypes.SGD):
        optimizer= optim.SGD(params=model.parameters(),lr=config.lr)
    else:
        raise NotImplementedError(str.format('{} optimizer type is not implemented in optimizer',config.optim_type.value))
    return optimizer

# TODO set in config file schedular parameters

def get_lr_schedular(config:BaseConfigs,optimizer:optim.Optimizer) -> _LRScheduler:
    """
    schedular builder function, that get the correct schedular specificed in the config class
    :param config: object of the config class
    :type config: BaseConfigs
    :param optimizer: optimizer responsible for model weights updates
    :type optimizer: optim.Optimizer
    :return: torch.optim.lr_scheduler._LRScheduler
    """
    if(config.schedular_type==SchedularTypes.STEP):
        lr_schedular=StepLR(optimizer,1)
    elif(config.schedular_type==SchedularTypes.CONSINE_ANNEALING):
        lr_schedular=CosineAnnealingLR(optimizer)
    elif(config.schedular_type==SchedularTypes.COSINE_ANNEALING_WARM_START):
        lr_schedular=CosineAnnealingWarmRestarts(optimizer)
    elif(config.schedular_type==SchedularTypes.LAMBDA):
        lr_schedular=LambdaLR(optimizer)
    elif(config.schedular_type==SchedularTypes.CHAINED):
        lr_schedular=ChainedScheduler(optimizer)
    elif(config.schedular_type==SchedularTypes.CYCLIC):
        lr_schedular=CyclicLR(optimizer)
    elif(config.schedular_type==SchedularTypes.Linear):
        lr_schedular=LinearLR(optimizer)
    elif(config.schedular_type==SchedularTypes.SEQUENTIAL):
        lr_schedular=SequentialLR(optimizer)
    elif(config.schedular_type==SchedularTypes.MULTISTEP):
        lr_schedular=MultiStepLR(optimizer)
    elif(config.schedular_type==SchedularTypes.REDUCE_ON_PLATEAU):
        lr_schedular=ReduceLROnPlateau(optimizer)
    elif(config.schedular_type==SchedularTypes.EXPONENTIAL):
        lr_schedular=ExponentialLR(optimizer)
    else:
        lr_schedular=ConstantLR(optimizer,1)
    
    return lr_schedular