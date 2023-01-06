from torch.nn import Module

class BaseCriterion(Module):
    """
        base class for calculating loss functions
    """
    def __init__(self) -> None:
        super().__init__()
    def forward(self,**kwargs):
        """
        base forward pass function
        """
        return 