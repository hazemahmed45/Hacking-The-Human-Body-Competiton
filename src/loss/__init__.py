from src.config import BaseConfigs
from src.loss.base_loss import BaseCriterion
from src.enums import CriterionTypes
from src.loss.loss import *


def get_criterion(config:BaseConfigs) -> BaseCriterion:
    """
    criterion builder function, that get the correct criterion specificed in the config class
    :param config: object of the config class
    :type config: SegmentationConfigs
    :return: src.loss.base_loss.BaseCriterion
    """
    if(config.criterion_type == CriterionTypes.CROSS_ENTROPY):
        return CrossEntropyCriterion()
    elif(config.criterion_type == CriterionTypes.FOCAL):
        return FocalLossCriterion(gamma=config.focal_gamma,alpha=config.focal_alpha)
    elif(config.criterion_type == CriterionTypes.CROSS_ENTROPY_WITH_CLASS):
        return CrossEntropyWithClassCriterion()

    else:
        raise NotImplementedError()
    return 