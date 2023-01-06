import os
from typing import List
from src.config import BaseConfigs
from src.enums import CallbackTypes
from pytorch_lightning.callbacks import Callback,\
                                        ModelCheckpoint,\
                                        EarlyStopping,\
                                        GradientAccumulationScheduler,\
                                        TQDMProgressBar,\
                                        DeviceStatsMonitor,\
                                        GPUStatsMonitor,\
                                        LearningRateMonitor,\
                                        ModelSummary,\
                                        RichModelSummary,\
                                        Timer
# TODO complete callback parameters, and add more callbacks

def get_callbacks(config:BaseConfigs) -> List[Callback]:
    """
    callbacks builder function, that get the correct list of callbacks specificed in the config class
    :param config: object of the config class
    :type config: BaseConfigs
    :return: list of callbacks
    :rtype: List[Callback]
    """
    callbacks=[]
    
    for callback_type in config.callbacks:
        if(callback_type==CallbackTypes.EARLY_STOPPING):
            callbacks.append(EarlyStopping(
                monitor=config.early_stop_monitor,
                min_delta=config.early_stop_min_delta,
                patience=config.early_stop_patience,
                verbose=config.early_stop_verbose,
                mode=config.early_stop_monitor_mode.value,
                check_finite=True
                ))
        elif(callback_type==CallbackTypes.GRADIENT_ACCUMULATION_SCHEDULAR):
            callbacks.append(GradientAccumulationScheduler(scheduling=config.gradient_accumlation_schedular_factor))
        elif(callback_type==CallbackTypes.MODEL_CHECKPOINT):
            callbacks.append(ModelCheckpoint(
                dirpath=os.path.join(config.get_exp_dir(),'ckpt'),
                filename=str.format('[{}]-[{}]-best_weights',config.get_exp_name(),config.model_type.value),
                monitor=config.ckpt_monitor,
                mode=config.ckpt_monitor_mode.value,
                verbose=config.ckpt_verbose,
                save_last=config.save_last,
                save_top_k=config.save_top_k,
                save_weights_only=config.save_weight_only,
                save_on_train_epoch_end=True
                ))
        elif(callback_type==CallbackTypes.TQDM_PROGRESS_BAR):
            callbacks.append(TQDMProgressBar())
        elif(callback_type == CallbackTypes.DEVICE_STATS_MONITOR):
            callbacks.append(DeviceStatsMonitor())
        elif(callback_type==CallbackTypes.GPUS_STATS_MONITOR):
            callbacks.append(GPUStatsMonitor())
        elif(callback_type==CallbackTypes.LEARNING_RATE_MONITOR):
            callbacks.append(LearningRateMonitor())
        elif(callback_type==CallbackTypes.MODEL_SUMMARY):
            callbacks.append(ModelSummary())
        elif(callback_type==CallbackTypes.RICH_MODEL_SUMMARY):
            callbacks.append(RichModelSummary())
        elif(callback_type==CallbackTypes.TIMER):
            callbacks.append(Timer())
        else: 
            raise NotImplementedError(str.format('{} is not implemented',callback_type.value))
    return callbacks