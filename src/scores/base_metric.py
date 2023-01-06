from abc import abstractmethod
from typing import List
from src.config import BaseConfigs, MetricTypes
import torch

from src.enums import DictKeys


class BaseMetric():
    """
    base metric class
    """
    def __init__(self,config:BaseConfigs,name='metric') -> None:
        """
        metric constructor

        attributes:-

        - config : configurations of the trial
        - name : name of the metric

        :param config: configurations of the trial
        :type config: BaseConfigs
        :param name: name of the metric, defaults to 'metric'
        :type name: str, optional
        """
        self.name=name
        self.config=config
    @abstractmethod
    def update(self,**kwargs):
        """
        function that is called when updating the metrics values
        """
        return 
    @abstractmethod
    def get_metric_value(self):
        """
        function that returns the metric values calculated
        """
        return 
    @abstractmethod
    def reset(self):
        """
        function that resets the metric values
        """
        return 

class RunningLoss(BaseMetric):
    """
    this class is responsible for calculating the average running loss for each epoch
    """
    def __init__(self,config:BaseConfigs,name='RunningLoss'):
        super().__init__(config,name)
        self.reset()
    def update(self,**kwargs):

        step_loss=kwargs[DictKeys.BATCH_LOSS.value]
        self.val=self.val.to(step_loss)
        self.val+=step_loss
        self.n+=1.0
        return 
    def get_metric_value(self):
        return {self.name:self.val/self.n if self.n>0 else 0.0}
    def reset(self):
        self.val=torch.tensor([0],dtype=torch.float,device='cpu')
        self.n=0.0
        return 
    def __add__(self,other_metric):
        self.val=self.val.to(other_metric.val.device)
        self.val+=other_metric.val
        self.n+=other_metric.n
        return self
class Loss(BaseMetric):
    """
    this class is responsible for calculating the loss for each batch
    """
    def __init__(self,config:BaseConfigs,name='loss'):
        super().__init__(config,name)
        self.val=0
    def update(self,**kwargs):  
        self.val=kwargs[DictKeys.BATCH_LOSS.value]
        return 
    def get_metric_value(self):
        return {self.name:self.val}
    def reset(self):
        self.val=torch.tensor([0],dtype=torch.float,device='cpu')
        return  
    def __add__(self,other_loss):
        if(isinstance(other_loss.val,torch.Tensor)):
            if(not isinstance(self.val,torch.Tensor)):
                self.val=torch.tensor([0],dtype=torch.float,device='cpu')
            self.val=self.val.to(other_loss.val.device)
            other_loss.val=torch.reshape(other_loss.val,shape=self.val.shape)

        self.val+=other_loss.val
        return self

class MetricCompose():
    """
    this class is responsible for creating a pipeline of metrics 
    """
    def __init__(self,metrics:List[BaseMetric]) -> None:
        """
        constructor for creating composition of metrics calculations

        :param metrics: list of metrics that inherits from BaseMetric
        :type metrics: List[BaseMetric]
        """
        self.metrics=metrics
    def update(self, **kwargs) :
        """
        update function that call all the updates of the list of metrics
        """
        for metric in self.metrics:
            metric.update(**kwargs)

        return 
    def get_metrics_dict(self):
        """
        get the metrics of all the list of the metrics inside the metric composer

        :return: dictionary contains all the concatinations of metrics names and metric values
        :rtype: dict
        """
        all_metric_dict={}
        for metric in self.metrics:
            all_metric_dict={**all_metric_dict,**metric.get_metric_value()}
        return all_metric_dict
    def reset(self):
        """
        reset all metrics
        """
        for metric in self.metrics:
            metric.reset()
        return 
    @staticmethod
    def merge_metrics(metric_composers:List):
        """
        merge two MetricCompose with in one in form of merging the metrics values of each metric composer
        with the metric values of the other metric composer

        :param metric_composers: metric composers that needs to be merged
        :type metric_composers: List
        :return: merged metric composer
        :rtype: MetricCompose
        """
        assert len(metric_composers)>0
        merged_metric_composer=None
        for ii, metric_composer in enumerate(metric_composers):
            if(ii==0):
                merged_metric_composer=MetricCompose(metric_composer.metrics)
                # merged_metric_composer.reset()
            else:
                for i in range(len(metric_composer.metrics)):
                    
                    merged_metric_composer.metrics[i]+=metric_composer.metrics[i]
                        
        return merged_metric_composer