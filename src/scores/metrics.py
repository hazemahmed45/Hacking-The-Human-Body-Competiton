from src.enums import DictKeys
from src.scores.base_metric import BaseMetric
from src.config import BaseConfigs
from torchmetrics.functional.classification.jaccard import jaccard_index
from torchmetrics.functional.classification.stat_scores import stat_scores 

import torch


class Precision(BaseMetric):
    """
    calculate the precision of model prediction against the ground truth for each class
    """
    def __init__(self, config: BaseConfigs, name='precision') -> None:
        super().__init__(config=config, name=name)
        self.classes_labels_dict = config.class_labels_dict
        self.classes_names = list(self.classes_labels_dict.keys())
        self.classes_labels = list(self.classes_labels_dict.values())
        
        self.reset()
        self.epsilon = 1e-8
    def update(self, **kwargs):
        
        prediction=kwargs[DictKeys.Y_PRED.value]
        ground_truth=kwargs[DictKeys.Y_TRUE.value]
        prediction=torch.argmax(prediction,dim=1)
        stats_values=stat_scores(preds=prediction,target=ground_truth,reduce='macro',num_classes=len(self.classes_names),mdmc_reduce='global')
        
        stats_values[torch.isnan(stats_values)]=0
        self.val=self.val.to(stats_values.device)
        self.val+=stats_values#.detach().cpu()
        self.n+=1
        return 
    
    def reset(self):
        self.val=torch.zeros(size=(len(self.classes_names),5),dtype=torch.float64,device='cpu')
        self.n=0
        return 
    def get_metric_value(self):
        metric_values={}
        prec_scores=(self.val[:,0]/(self.val[:,0]+self.val[:,1]+self.epsilon)).float()

        for class_name,class_label in self.classes_labels_dict.items():
            metric_values[self.name+'-'+class_name]=prec_scores[class_label]
        return metric_values
    def __add__(self, other_metric):
        self.val=self.val.to(other_metric.val.device)
        self.val+=other_metric.val
        return self

class Recall(BaseMetric):
    """
    calculate the recall of model prediction against the ground truth for each class
    """
    def __init__(self, config: BaseConfigs, name='recall') -> None:
        super().__init__(config=config, name=name)
        self.classes_labels_dict = config.class_labels_dict
        self.classes_names = list(self.classes_labels_dict.keys())
        self.classes_labels = list(self.classes_labels_dict.values())
        
        self.reset()
        self.epsilon = 1e-8
    def update(self, **kwargs):

        prediction=kwargs[DictKeys.Y_PRED.value]
        ground_truth=kwargs[DictKeys.Y_TRUE.value]
        prediction=torch.argmax(prediction,dim=1)
 
        stats_values=stat_scores(preds=prediction,target=ground_truth,reduce='macro',num_classes=len(self.classes_names),mdmc_reduce='global')
        stats_values[torch.isnan(stats_values)]=0
        # if(self.val is None):
        #     self.val=torch.zeros_like(stats_values)
        self.val=self.val.to(stats_values.device)
        self.val+=stats_values#.detach().cpu()
        self.n+=1
        return 
    
    def reset(self):

        self.val=torch.zeros(size=(len(self.classes_names),5),dtype=torch.float64,device='cpu')
        self.n=0
        return 
    def get_metric_value(self):
        metric_values={}
        recall_scores=(self.val[:,0]/(self.val[:,0]+self.val[:,3]+self.epsilon)).float()
        for class_name,class_label in self.classes_labels_dict.items():
            metric_values[self.name+'-'+class_name]=recall_scores[class_label]
        
        return metric_values
    def __add__(self, other_metric):
        self.val=self.val.to(other_metric.val.device)
        self.val+=other_metric.val
        return self

class F1Score(BaseMetric):
    """
    calculate the f1 score of model prediction against the ground truth for each class
    """
    def __init__(self, config: BaseConfigs, name='f1score') -> None:
        super().__init__(config, name)

        self.classes_labels_dict = config.class_labels_dict
        self.classes_names = list(self.classes_labels_dict.keys())
        self.classes_labels = list(self.classes_labels_dict.values())
        
        self.reset()
        self.epsilon = 1e-8
    def update(self, **kwargs):

        prediction=kwargs[DictKeys.Y_PRED.value]
        ground_truth=kwargs[DictKeys.Y_TRUE.value]
        prediction=torch.argmax(prediction,dim=1)
        stats_values=stat_scores(preds=prediction,target=ground_truth,reduce='macro',num_classes=len(self.classes_names),mdmc_reduce='global')
        stats_values[torch.isnan(stats_values)]=0
        self.val=self.val.to(stats_values.device)
        self.val+=stats_values#.detach().cpu()
        self.n+=1
        return 
    def reset(self):

        self.val=torch.zeros(size=(len(self.classes_names),5),dtype=torch.float64,device='cpu')
        self.n=0
        return 
    
    def get_metric_value(self):
        metric_values={}
        f1_scores=(self.val[:,0]/(self.val[:,0]+0.5*(self.val[:,1]+self.val[:,3]))+self.epsilon).float()

        # f1_scores=self.get_f1_score()
        for class_name,class_label in self.classes_labels_dict.items():
            # metric_values[self.name+'-'+class_name]=(2 * self.val[class_label,0]) / (
            #         (2 * self.val[class_label,0]) + self.val[class_label,1] + self.val[class_label,3] + self.epsilon)
            metric_values[self.name+'-'+class_name]=f1_scores[class_label]#self.val[class_label]/self.n if self.n>0 else torch.tensor(0)
        return metric_values
    def __add__(self, other_metric):
        self.val=self.val.to(other_metric.val.device)
        self.val+=other_metric.val
        return self

class StatsMeter(BaseMetric):
    """
    calculate the precision, recall and f1 score of model prediction against the ground truth for each class
    """
    def __init__(self, config: BaseConfigs, name='agro') -> None:
        super().__init__(config, name)
        self.classes_labels_dict = config.class_labels_dict
        self.classes_names = list(self.classes_labels_dict.keys())
        self.classes_labels = list(self.classes_labels_dict.values())

        self.reset()

        self.epsilon = 1e-8
    def update(self, **kwargs):
        prediction=kwargs[DictKeys.Y_PRED.value]
        ground_truth=kwargs[DictKeys.Y_TRUE.value]
        prediction=torch.argmax(prediction,dim=1)
        stats_values=stat_scores(preds=prediction,target=ground_truth,reduce='macro',num_classes=len(self.classes_names),mdmc_reduce='global')
        stats_values[torch.isnan(stats_values)]=0.0
        self.val=self.val.to(stats_values.device)
        self.val+=stats_values#.detach().cpu()
        return 
    def reset(self):
        # print("###########HERE############")
        self.val=torch.zeros(size=(len(self.classes_names),5),dtype=torch.float32,device='cpu')
        return 
    def get_metric_value(self):
        metric_values={}
        precisions=(self.val[:,0]/(self.val[:,0]+self.val[:,1]+self.epsilon)).float()
        recalls=(self.val[:,0]/(self.val[:,0]+self.val[:,3]+self.epsilon)).float()
        f1_scores=(self.val[:,0]/(self.val[:,0]+0.5*(self.val[:,1]+self.val[:,3])+self.epsilon)).float()
        for class_name,class_label in self.classes_labels_dict.items():
            metric_values[str.format("{}-precision-{}",self.name,class_name)]=precisions[class_label]
            metric_values[str.format("{}-recall-{}",self.name,class_name)]=recalls[class_label]
            metric_values[str.format("{}-f1score-{}",self.name,class_name)]=f1_scores[class_label]
        return metric_values
    
    def true_positives(self,prediction, ground_truth, value):
        return torch.sum(torch.logical_and(ground_truth == value, prediction == value)).to(torch.float32).item()

    def false_positives(self,prediction, ground_truth, value):
        return torch.sum(torch.logical_and(ground_truth != value, prediction == value)).to(torch.float32).item()

    def false_negatives(self,prediction, ground_truth, value):
        return torch.sum(torch.logical_and(ground_truth == value, prediction != value)).to(torch.float32).item()

    def __add__(self, other_metrics):
        # self.TP=other_metrics.copy()
        self.val=self.val.to(other_metrics.val.device)
        self.val+=other_metrics.val
        return self

class StatsClassificationMeter(BaseMetric):
    """
    calculate the precision, recall and f1 score of model prediction against the ground truth for each class
    """
    def __init__(self, config: BaseConfigs, name='stats') -> None:
        super().__init__(config, name)
        self.classes_labels_dict = config.organ_labels_dict
        self.classes_names = list(self.classes_labels_dict.keys())
        self.classes_labels = list(self.classes_labels_dict.values())

        self.reset()

        self.epsilon = 1e-8
    def update(self, **kwargs):
        prediction=kwargs[DictKeys.Y_PRED.value]
        ground_truth=kwargs[DictKeys.Y_TRUE.value]
        prediction=torch.argmax(prediction,dim=1)
        # print(ground_truth,prediction)
        stats_values=stat_scores(preds=prediction,target=ground_truth,reduce='macro',num_classes=len(self.classes_names),mdmc_reduce='global')
        stats_values[torch.isnan(stats_values)]=0.0
        self.val=self.val.to(stats_values.device)
        self.val+=stats_values#.detach().cpu()
        return 
    def reset(self):
        # print("###########HERE############")
        self.val=torch.zeros(size=(len(self.classes_names),5),dtype=torch.float32,device='cpu')
        return 
    def get_metric_value(self):
        metric_values={}
        precisions=(self.val[:,0]/(self.val[:,0]+self.val[:,1]+self.epsilon)).float()
        recalls=(self.val[:,0]/(self.val[:,0]+self.val[:,3]+self.epsilon)).float()
        f1_scores=(self.val[:,0]/(self.val[:,0]+0.5*(self.val[:,1]+self.val[:,3])+self.epsilon)).float()
        for class_name,class_label in self.classes_labels_dict.items():
            metric_values[str.format("{}-precision-{}",self.name,class_name)]=precisions[class_label]
            metric_values[str.format("{}-recall-{}",self.name,class_name)]=recalls[class_label]
            metric_values[str.format("{}-f1score-{}",self.name,class_name)]=f1_scores[class_label]
        return metric_values
    
    def true_positives(self,prediction, ground_truth, value):
        return torch.sum(torch.logical_and(ground_truth == value, prediction == value)).to(torch.float32).item()

    def false_positives(self,prediction, ground_truth, value):
        return torch.sum(torch.logical_and(ground_truth != value, prediction == value)).to(torch.float32).item()

    def false_negatives(self,prediction, ground_truth, value):
        return torch.sum(torch.logical_and(ground_truth == value, prediction != value)).to(torch.float32).item()

    def __add__(self, other_metrics):
        # self.TP=other_metrics.copy()
        self.val=self.val.to(other_metrics.val.device)
        self.val+=other_metrics.val
        return self
# TODO : solve iou metrics
class Iou(BaseMetric):
    """
    calculate the iou of model prediction against the ground truth for each class
    """
    def __init__(self, config: BaseConfigs, name='iou') -> None:
        super().__init__(config, name)
        
        self.classes_labels_dict = config.class_labels_dict
        self.labels_classes_dict={label:class_name for class_name,label in self.classes_labels_dict.items()}
        self.classes_names = list(self.classes_labels_dict.keys())
        self.classes_labels = list(self.classes_labels_dict.values())
        self.reset()

    def update(self, **kwargs):
        prediction=kwargs[DictKeys.Y_PRED.value]
        target=kwargs[DictKeys.Y_TRUE.value]
        prediction=torch.argmax(prediction,dim=1)
        classes_ious=jaccard_index(preds=prediction,target=target,num_classes=self.config.out_channels,reduction='none',absent_score=0.0)
        classes_ious[torch.isnan(classes_ious)]=0
        self.val=self.val.to(classes_ious.device)

        self.val+=classes_ious#.detach().cpu()
        self.n+=1
        return 

    def reset(self):
        self.val=torch.zeros(size=(len(self.classes_names),),dtype=torch.float64,device='cpu')
        self.n=0.0
        return 
    def get_metric_value(self):
        metric_values={}
        for class_name,class_label in self.classes_labels_dict.items():
            metric_values[self.name+'-'+class_name]=(self.val[class_label]/self.n) if self.n>0 else 0.0
        metric_values[self.name+'-miou']=(self.val.mean()/self.n) if self.n>0 else 0.0
        return metric_values
    def __add__(self, other_metric):
        self.val=self.val.to(other_metric.val.device)
        self.val+=other_metric.val
        self.n+=other_metric.n
        return self

class DiceCoeff(BaseMetric):
    """
    calculate the dice coefficient of model prediction against the ground truth for each class
    """
    def __init__(self, config: BaseConfigs, name='dice') -> None:
        super().__init__(config, name)
        self.classes_labels_dict = config.class_labels_dict
        self.labels_classes_dict={label:class_name for class_name,label in self.classes_labels_dict.items()}
        self.classes_names = list(self.classes_labels_dict.keys())
        self.classes_labels = list(self.classes_labels_dict.values())
        self.epsilon=1e-8
        self.reset()
    def update(self, **kwargs):
        prediction=kwargs[DictKeys.Y_PRED.value]
        target=kwargs[DictKeys.Y_TRUE.value]
        prediction=torch.argmax(prediction,dim=1)
        
        stats_values=stat_scores(preds=prediction,target=target,reduce='macro',num_classes=len(self.classes_names),mdmc_reduce='global')
        stats_values[torch.isnan(stats_values)]=0
        # stats_values=stats_values.detach().cpu()
        # if(self.val is None):
        #     self.val=torch.zeros_like(stats_values)
        self.val=self.val.to(stats_values.device)
        self.val+=stats_values#.detach().cpu()
        self.n+=1
        return 
    def reset(self):
        self.val=torch.zeros(size=(len(self.classes_names),5),dtype=torch.float64,device='cpu')
        self.n=0
        return 
    def get_metric_value(self):
        metric_values={}
        mean_dice=0
        dice_coffs=(self.val[:,0]*2/(2*self.val[:,0]+self.val[:,1]+self.val[:,2]+self.epsilon))
        for class_name,class_label in self.classes_labels_dict.items():

            metric_values[self.name+'-'+class_name]=dice_coffs[class_label]#(self.val[class_label,0]*2/(2*self.val[class_label,0]+self.val[class_label,1]+self.val[class_label,2]))
            mean_dice+=metric_values[self.name+'-'+class_name]
        metric_values[self.name+'-mdice']=dice_coffs.mean()
        return metric_values
    def __add__(self, other_metric):
        self.val=self.val.to(other_metric.val.device)
        self.val+=other_metric.val
        return self


class MaskWithOrganStatsMeter(BaseMetric):
    def __init__(self, config: BaseConfigs, name='agronometer') -> None:
        super().__init__(config, name)
        self.mask_agro=StatsMeter(config,name=str.format('Mask_{}',self.name))
        self.organ_agro=StatsClassificationMeter(config,name=str.format('Organ_{}',self.name))
    def update(self, **kwargs):
        mask_pred=kwargs[DictKeys.Y_PRED.value]['mask']
        organ_pred=kwargs[DictKeys.Y_PRED.value]['organ']
        mask_target=kwargs[DictKeys.Y_TRUE.value]['mask']
        organ_target=kwargs[DictKeys.Y_TRUE.value]['organ']
        self.mask_agro.update(**{DictKeys.Y_PRED.value:mask_pred,DictKeys.Y_TRUE.value:mask_target})
        self.organ_agro.update(**{DictKeys.Y_PRED.value:organ_pred,DictKeys.Y_TRUE.value:organ_target})
        return 
    def reset(self):
        self.mask_agro.reset()
        self.organ_agro.reset()
        return 
    def get_metric_value(self):
        mask_metric_dict=self.mask_agro.get_metric_value()
        organ_metric_dict=self.organ_agro.get_metric_value()

        return {**organ_metric_dict,**mask_metric_dict}

    def __add__(self, other_metric):
        self.mask_agro+=other_metric.mask_agro
        self.organ_agro+=other_metric.organ_agro

        return self
class MaskWithOrganIou(BaseMetric):
    def __init__(self, config: BaseConfigs, name='iou') -> None:
        super().__init__(config, name)
        self.mask_iou=Iou(config,name=str.format('Mask_{}',self.name))
    def update(self, **kwargs):
        mask_pred=kwargs[DictKeys.Y_PRED.value]['mask']
        mask_target=kwargs[DictKeys.Y_TRUE.value]['mask']
        self.mask_iou.update(**{DictKeys.Y_PRED.value:mask_pred,DictKeys.Y_TRUE.value:mask_target})
        return 
    def reset(self):
        self.mask_iou.reset()
        return 
    def get_metric_value(self):
        mask_metric_dict=self.mask_iou.get_metric_value()

        return {**mask_metric_dict}

    def __add__(self, other_metric):
        self.mask_iou+=other_metric.mask_iou

        return self

class MaskWithOrganDice(BaseMetric):
    def __init__(self, config: BaseConfigs, name='dice') -> None:
        super().__init__(config, name)
        self.mask_dice=DiceCoeff(config,name=str.format('Mask_{}',self.name))
    def update(self, **kwargs):
        mask_pred=kwargs[DictKeys.Y_PRED.value]['mask']
        mask_target=kwargs[DictKeys.Y_TRUE.value]['mask']
        self.mask_dice.update(**{DictKeys.Y_PRED.value:mask_pred,DictKeys.Y_TRUE.value:mask_target})
        return 
    def reset(self):
        self.mask_dice.reset()
        return 
    def get_metric_value(self):
        mask_metric_dict=self.mask_dice.get_metric_value()

        return {**mask_metric_dict}

    def __add__(self, other_metric):
        self.mask_dice+=other_metric.mask_dice

        return self