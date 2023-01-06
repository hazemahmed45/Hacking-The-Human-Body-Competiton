from typing import List
from torch.utils.data import Dataset
from abc import abstractmethod
import numpy as np

class BaseDataset(Dataset):
    """
    base dataset class
    """
    def __init__(self) -> None:
        super().__init__()
        """
        dataset constructor
        """
    @abstractmethod
    def get_data_list(self) -> List:
        """
        return a list of items that will be read during the training

        :return: dataset with any related data
        :rtype: List
        """
        return
    @abstractmethod
    def collate_fn(self,batch):
        """
        abstract function of the collate function that is responsible for reshaping the return 
        of the dataloader to match the trainer structure

        :param batch: tuple of dataloader returns
        :type batch: Tuple
        :return: dictionary of dataloader returns that contain 'filenames', 'input' and 'y_true' as a must in keys
        :rtype: dict 
        """
        pass

