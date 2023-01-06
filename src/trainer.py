from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Sequence
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.optim import Optimizer
from src.enums import DictKeys, StrategyTypes
from src.config import BaseConfigs
from src.loss import get_criterion
from src.models import get_model
from src.scores.base_metric import MetricCompose
from src.scores import get_metrics
from src.dataloaders import get_dataloader
from src.optimizer import get_optimizer,get_lr_schedular
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
import torch


class BaseSegmentationTrainer(LightningModule):
    """
    this controller class, is responsible for the training pipeline, from fetching the model and using the dataloaders with calculating 
    loss and doing backwards and logging metrics calculated
    """
    trainer: Trainer
    def __init__(self,config:BaseConfigs):
        """
        constructor of the base segmentation trainer

        attributes:-

        - config : configurations of segmentation trial
        - model : model architecture specified in the config class
        - criterion : criterion specified in the config class
        - history : dictionary that holds all metric logs
        :param config: configurations of segmentation trial
        :type config: BaseConfigs
        """
        super().__init__()
        self.config=config
        
        self.model=get_model(config=self.config)
        self.criterion=get_criterion(self.config)
        self.save_hyperparameters(self.config.get_config_dict())
        self.history={}
        return
    def on_fit_start(self) -> None:
        
        self.history.clear()
        if(self.on_gpu):
            self.train_metrics={i:get_metrics(config=self.config) for i in range(self.trainer.num_devices)}
            self.val_metrics={i:get_metrics(config=self.config) for i in range(self.trainer.num_devices)}
        else:
            print(self.trainer.num_devices)
            print(self.trainer.num_processes)
            print(self.device)
            if(self.config.strategy_type==StrategyTypes.CPU):
                self.train_metrics={self.device:get_metrics(config=self.config)}
                self.val_metrics={self.device:get_metrics(config=self.config)}
            else:
                raise NotImplementedError(f'this strategy {self.config.strategy_type.value} is not handled with cpu run')

                # self.train_metrics={i:get_metrics(config=self.config) for i in range(self.trainer.num_devices)}
                # self.val_metrics={i:get_metrics(config=self.config) for i in range(self.trainer.num_devices)}
        return 
    def on_validation_start(self) -> None:
        self.history.clear()
        if(self.on_gpu):
            self.val_metrics={i:get_metrics(config=self.config) for i in range(self.trainer.num_devices)}
        else:
            if(self.config.strategy_type==StrategyTypes.CPU):
                self.val_metrics={self.device:get_metrics(config=self.config)}
            else:
                raise NotImplementedError(f'this strategy {self.config.strategy_type.value} is not handled with cpu run')
                # self.val_metrics={i:get_metrics(config=self.config) for i in range(self.trainer.num_devices)}
        return 
    def on_test_start(self) -> None:
        self.history.clear()
        if(self.on_gpu):
            self.test_metrics={i:get_metrics(config=self.config) for i in range(self.trainer.num_devices)}
        else:
            if(self.config.strategy_type==StrategyTypes.CPU):
                self.test_metrics={self.device:get_metrics(config=self.config)}
            else:
                raise NotImplementedError(f'this strategy {self.config.strategy_type.value} is not handled with cpu run')
                # self.test_metrics={i:get_metrics(config=self.config) for i in range(self.trainer.num_devices)}
        return 
        
    def training_step(self,kwargs,batch_idx) -> Union[int, Dict[str, Union[Tensor, Dict[str, Tensor]]]]:
        output_dict=self.model.forward_step(**kwargs)
        loss=self.criterion(**{**output_dict,**kwargs})
        if(self.on_gpu):
            if(self.config.strategy_type==StrategyTypes.DATA_PARALLEL):
                self.train_metrics[self.device.index].update(**{**output_dict,**kwargs,**{DictKeys.BATCH_LOSS.value:loss}})
            else:
                self.train_metrics[self.local_rank].update(**{**output_dict,**kwargs,**{DictKeys.BATCH_LOSS.value:loss}})
        else:
            if(self.config.strategy_type == StrategyTypes.CPU):
                self.train_metrics[self.device].update(**{**output_dict,**kwargs,**{DictKeys.BATCH_LOSS.value:loss}})
            else:
                raise NotImplementedError(f'this strategy {self.config.strategy_type.value} is not handled with cpu run')
                # self.train_metrics[self.local_rank].update(**{**output_dict,**kwargs,**{DictKeys.BATCH_LOSS.value:loss}})
        return loss

    def training_epoch_end(self, batchs_output_dict) -> Dict[str, Dict[str, Tensor]]:
        merged_train_metrics=MetricCompose.merge_metrics(self.train_metrics.values())
        metrics_values=merged_train_metrics.get_metrics_dict()
        metrics_values=self.all_gather(metrics_values)

        metric_results={}
        for metric_values_key,metric_values_value in metrics_values.items():
            metric_results[f'{DictKeys.TRAIN.value}/{metric_values_key}']=metric_values_value.mean()     
        if(self.local_rank==0):
            self.log_dict(metric_results,rank_zero_only=True)
            for metric_key, metric_value in metric_results.items():
                if(metric_key not in self.history.keys()):
                    self.history[metric_key]=[]
                self.history[metric_key].append(metric_value.item())

        for device in self.train_metrics.keys():
            if(self.on_gpu):
                if(self.config.strategy_type==StrategyTypes.DATA_PARALLEL):
                    self.train_metrics[device].reset()
                else:
                    self.train_metrics[self.local_rank].reset()
            else:
                if(self.config.strategy_type==StrategyTypes.CPU):
                    self.train_metrics[device].reset()
                else:
                    raise NotImplementedError(f'this strategy {self.config.strategy_type.value} is not handled with cpu run')

        return 
    def validation_step(self,kwargs,batch_idx) -> Union[int, Dict[str, Union[Tensor, Dict[str, Tensor]]]]:
        output_dict=self.model.forward_step(**kwargs)
        loss=self.criterion(**{**output_dict,**kwargs})
        if(self.on_gpu):
            if(self.on_gpu):
                if(self.config.strategy_type==StrategyTypes.DATA_PARALLEL):
                    self.val_metrics[self.device.index].update(**{**output_dict,**kwargs,**{DictKeys.BATCH_LOSS.value:loss}})
                else:
                    self.val_metrics[self.local_rank].update(**{**output_dict,**kwargs,**{DictKeys.BATCH_LOSS.value:loss}})
        else:
            if(self.config.strategy_type == StrategyTypes.CPU):
                self.val_metrics[self.device].update(**{**output_dict,**kwargs,**{DictKeys.BATCH_LOSS.value:loss}})
            else:
                raise NotImplementedError(f'this strategy {self.config.strategy_type.value} is not handled with cpu run')

                # self.val_metrics[self.local_rank].update(**{**output_dict,**kwargs,**{DictKeys.BATCH_LOSS.value:loss}})
        return 

    def validation_epoch_end(self, batchs_output_dict) -> Dict[str, Dict[str, Tensor]]:
        merged_val_metrics=MetricCompose.merge_metrics(self.val_metrics.values())
        metrics_values=merged_val_metrics.get_metrics_dict()
        metrics_values=self.all_gather(metrics_values)
        metric_results={}
        for metric_values_key,metric_values_value in metrics_values.items():
            metric_results[f'{DictKeys.VALID.value}/{metric_values_key}']=metric_values_value.mean()

        if(self.local_rank == 0):
            self.log_dict(metric_results,rank_zero_only=True)
            for metric_key, metric_value in metric_results.items():
                if(metric_key not in self.history.keys()):
                    self.history[metric_key]=[]
                self.history[metric_key].append(metric_value.item())
        for device in self.val_metrics.keys():
            if(self.on_gpu):
                if(self.config.strategy_type==StrategyTypes.DATA_PARALLEL):
                    self.val_metrics[device].reset()
                else:
                    self.val_metrics[self.local_rank].reset()
            else:
                if(self.config.strategy_type==StrategyTypes.CPU):

                    self.val_metrics[device].reset()
                else:
                    raise NotImplementedError(f'this strategy {self.config.strategy_type.value} is not handled with cpu run')

        
        return 

    def test_step(self,kwargs,batch_idx) -> Union[int, Dict[str, Union[Tensor, Dict[str, Tensor]]]]:
        output_dict=self.model.forward_step(**kwargs)
        loss=self.criterion(**{**output_dict,**kwargs})
        if(self.on_gpu):
            if(self.on_gpu):
                if(self.config.strategy_type==StrategyTypes.DATA_PARALLEL):
                    self.test_metrics[self.device.index].update(**{**output_dict,**kwargs,**{DictKeys.BATCH_LOSS.value:loss}})
                else:
                    self.test_metrics[self.local_rank].update(**{**output_dict,**kwargs,**{DictKeys.BATCH_LOSS.value:loss}})
        else:
            if(self.config.strategy_type == StrategyTypes.CPU):
                self.test_metrics[self.device].update(**{**output_dict,**kwargs,**{DictKeys.BATCH_LOSS.value:loss}})
            else:
                raise NotImplementedError(f'this strategy {self.config.strategy_type.value} is not handled with cpu run')
                # self.test_metrics[self.local_rank].update(**{**output_dict,**kwargs,**{DictKeys.BATCH_LOSS.value:loss}})
        return  
    
    def test_epoch_end(self, batchs_output_dict) -> Dict[str, Dict[str, Tensor]]:
        merged_test_metrics=MetricCompose.merge_metrics(self.test_metrics.values())
        metrics_values=merged_test_metrics.get_metrics_dict()
        metrics_values=self.all_gather(metrics_values)

        metric_results={}
        for metric_values_key,metric_values_value in metrics_values.items():
            metric_results[f'{DictKeys.TEST.value}/{metric_values_key}']=metric_values_value.mean()

        if(self.local_rank == 0):
            self.log_dict(metric_results,rank_zero_only=True)
            for metric_key, metric_value in metric_results.items():
                if(metric_key not in self.history.keys()):
                    self.history[metric_key]=[]
                self.history[metric_key].append(metric_value.item())
        for device in self.test_metrics.keys():
            if(self.on_gpu):
                if(self.config.strategy_type==StrategyTypes.DATA_PARALLEL):
                    self.test_metrics[device].reset()
                else:
                    self.test_metrics[self.local_rank].reset()
            else:
                if(self.config.strategy_type==StrategyTypes.CPU):
                    self.test_metrics[device].reset()
                else:
                    raise NotImplementedError(f'this strategy {self.config.strategy_type.value} is not handled with cpu run')

        return 
    
    def predict_step(self,**kwargs):
        output_dict=self.model.forward_step(**kwargs)
        return
    
    def configure_optimizers(self) -> Optional[Union[Optimizer, Sequence[Optimizer], Dict, Sequence[Dict], Tuple[List, List]]]:
        optimizer=get_optimizer(self.config,self.model)
        lr_schedular=get_lr_schedular(self.config,optimizer)
        return [optimizer],[lr_schedular]
    
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        model_state_dict=self.model.get_state_dict()
        if(not isinstance(self.model,torch.nn.Module)):
            model_state_dict=model_state_dict.module
        checkpoint[DictKeys.MODEL_STATE_DICT.value]=model_state_dict
        checkpoint[DictKeys.BEST_SCORE.value]=self.trainer.checkpoint_callback.best_model_score
        return 

    def forward(self, **kwargs):
        return self.model.forward(self.model.unpack_kwargs(**kwargs))



class WithDataloaderSegmentationTrainer(BaseSegmentationTrainer):
    trainer: Trainer
    def __init__(self,config:BaseConfigs):
        super().__init__(config)
        return

    def train_dataloader(self) -> DataLoader:
        train_dataloader=get_dataloader(self.config)[DictKeys.TRAIN.value]
        return train_dataloader
    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        val_dataloader=get_dataloader(self.config)[DictKeys.VALID.value]
        return val_dataloader
    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        test_dataloader=get_dataloader(self.config)[DictKeys.TEST.value]
        return test_dataloader
    def predict_dataloader(self)-> Union[DataLoader, List[DataLoader]]:
        infer_dataloader=get_dataloader(self.config)[DictKeys.INFERENCE.value]
        return infer_dataloader