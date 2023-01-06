import torch
import torch.nn.functional as F
from torch.autograd import Variable
from src.enums import DictKeys
from src.loss.base_loss import BaseCriterion

# focal loss criterion
class FocalLossCriterion(BaseCriterion):
    """
    Focal loss loss designed for vanilla unet
    """
    def __init__(self, gamma=0, alpha=None, size_average=True):
        """
        implementation of focal loss for unbalanced classification

        :param gamma: loss function hyperparameter, defaults to 0
        :type gamma: int, optional
        :param alpha: loss function hyperparameter, defaults to None
        :type alpha: float, optional
        :param size_average: loss function hyperparameter, defaults to True
        :type size_average: bool, optional
        """
        super(FocalLossCriterion, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, **kwargs):
        """
        forward pass

        :param kwargs: key-word dictionary that contains y_pred and y_true
        :return: loss
        :rtype: tensor
        """
        prediction=kwargs[DictKeys.Y_PRED.value]
        target=kwargs[DictKeys.Y_TRUE.value]
        if prediction.dim()>2:
            prediction = prediction.view(prediction.size(0),prediction.size(1),-1)  # N,C,H,W => N,C,H*W
            prediction = prediction.transpose(1,2)    # N,C,H*W => N,H*W,C
            prediction = prediction.contiguous().view(-1,prediction.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(prediction)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=prediction.data.type():
                self.alpha = self.alpha.type_as(prediction.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()

class CrossEntropyCriterion(BaseCriterion):
    """
    Crossentropy loss designed for vanilla unet
    """
    def __init__(self) -> None:
        """
        implementation of cross entropy designed for repo architecture

        - ce : native cross entropy loss

        """
        super().__init__()
        self.ce=torch.nn.CrossEntropyLoss()
        return 

    def forward(self, **kwargs):
        """
        forward pass

        :param kwargs: key-word dictionary that contains y_pred and y_true
        :return: loss
        :rtype: tensor
        """
        prediction=kwargs[DictKeys.Y_PRED.value]
        target=kwargs[DictKeys.Y_TRUE.value]
        return self.ce(prediction,target)
class CrossEntropyWithClassCriterion(BaseCriterion):
    """
    Crossentropy loss designed for vanilla unet
    """
    def __init__(self) -> None:
        """
        implementation of cross entropy designed for repo architecture

        - ce : native cross entropy loss

        """
        super().__init__()
        self.ce=torch.nn.CrossEntropyLoss()
        return 

    def forward(self, **kwargs):
        """
        forward pass

        :param kwargs: key-word dictionary that contains y_pred and y_true
        :return: loss
        :rtype: tensor
        """
        mask_pred=kwargs[DictKeys.Y_PRED.value]['mask']
        organ_pred=kwargs[DictKeys.Y_PRED.value]['organ']
        mask_target=kwargs[DictKeys.Y_TRUE.value]['mask']
        organ_target=kwargs[DictKeys.Y_TRUE.value]['organ']
        # print(organ_pred.shape,organ_target.shape)
        organ_loss=self.ce(organ_pred,organ_target)
        mask_loss=self.ce(mask_pred,mask_target)
        return mask_loss+organ_loss

