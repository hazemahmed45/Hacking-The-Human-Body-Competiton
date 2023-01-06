from src.config import BaseConfigs
from src.enums import MetricTypes
from src.scores.base_metric import MetricCompose,RunningLoss,Loss
from src.scores.metrics import *


def get_metrics(config:BaseConfigs)-> MetricCompose:
    """
    metrics builder function, that get the correct list of metrics specificed in the config class
    :param config: object of the config class
    :type config: BaseConfigs
    :return: src.scores.base_metric.MetricCompose
    """
    metrics=[]
    for metric_type in config.metrics:
        if(metric_type==MetricTypes.F1SCORE):
            metrics.append(F1Score(config))
        elif(metric_type==MetricTypes.LOSS):
            metrics.append(RunningLoss(config))
        elif(metric_type==MetricTypes.PRECISION):
            metrics.append(Precision(config))
        elif(metric_type==MetricTypes.RECALL):
            metrics.append(Recall(config))
        elif(metric_type==MetricTypes.IOU):
            metrics.append(Iou(config))
        elif(metric_type==MetricTypes.DICE):
            metrics.append(DiceCoeff(config))
        elif(metric_type==MetricTypes.MASK_ORGAN_STATS):
            metrics.append(MaskWithOrganStatsMeter(config))
        elif(metric_type==MetricTypes.MASK_ORGAN_IOU):
            metrics.append(MaskWithOrganIou(config))
        elif(metric_type==MetricTypes.MASK_ORGAN_DICE):
            metrics.append(MaskWithOrganDice(config))
        else:
            raise NotImplementedError(str.format('{} is not implemented',metric_type.value))
    metrics.append(Loss(config))
    return MetricCompose(metrics=metrics)