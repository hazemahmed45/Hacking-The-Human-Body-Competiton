from typing import Dict
from torch.utils.data import DataLoader
from src.config import BaseConfigs
from src.enums import ConfigTypes,DatasetTypes, DictKeys,SamplerTypes
from src.augmentations import get_augmentations
from src.dataloaders.dataloaders import *
import os
from src.dataloaders.base_dataset import BaseDataset
from torch.utils.data.sampler import RandomSampler,SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import ConcatDataset


def get_dataloader(config:BaseConfigs)-> Dict[str,DataLoader]:
    """
    dataloader builder function, that get the correct dataloader splits (train-valid-test) specificed in the config class
    :param config: object of the config class
    :type config: BaseConfigs
    :return: return dictionary of split name to dataloader of the split
    :rtype: Dict[str,DataLoader]
    """
    augmentation_dict=get_augmentations(config)
    data_loaders={}
    if(config.config_type == ConfigTypes.EVALUTAION or config.config_type==ConfigTypes.TRAIN):
        train_collate_fn=None
        val_collate_fn=None
        test_collate_fn=None
        if(config.dataset_type==DatasetTypes.SEGMENTATION):

            train_dataset,valid_dataset,test_dataset,train_collate_fn,val_collate_fn,test_collate_fn=create_train_val_test_datasets(
                config=config,
                dataset_class=SegDataset,
                train_kwargs={"split":DictKeys.TRAIN.value,"transforms":augmentation_dict[DictKeys.TRAIN.value]},
                valid_kwargs={"split":DictKeys.VALID.value,"transforms":augmentation_dict[DictKeys.VALID.value]},
                test_kwargs={"split":DictKeys.TEST.value,"transforms":augmentation_dict[DictKeys.TEST.value]}
                )
        elif(config.dataset_type==DatasetTypes.SEGMENTATION_BINARY):

            train_dataset,valid_dataset,test_dataset,train_collate_fn,val_collate_fn,test_collate_fn=create_train_val_test_datasets(
                config=config,
                dataset_class=SegBinaryDataset,
                train_kwargs={"split":DictKeys.TRAIN.value,"transforms":augmentation_dict[DictKeys.TRAIN.value]},
                valid_kwargs={"split":DictKeys.VALID.value,"transforms":augmentation_dict[DictKeys.VALID.value]},
                test_kwargs={"split":DictKeys.TEST.value,"transforms":augmentation_dict[DictKeys.TEST.value]}
                )
        elif(config.dataset_type==DatasetTypes.SEGMENTATION_BINARY_WITH_CLASS):

            train_dataset,valid_dataset,test_dataset,train_collate_fn,val_collate_fn,test_collate_fn=create_train_val_test_datasets(
                config=config,
                dataset_class=SegBinaryWithClassDataset,
                train_kwargs={"split":DictKeys.TRAIN.value,"transforms":augmentation_dict[DictKeys.TRAIN.value],'organ_label_dict':config.organ_labels_dict},
                valid_kwargs={"split":DictKeys.VALID.value,"transforms":augmentation_dict[DictKeys.VALID.value],'organ_label_dict':config.organ_labels_dict},
                test_kwargs={"split":DictKeys.TEST.value,"transforms":augmentation_dict[DictKeys.TEST.value],'organ_label_dict':config.organ_labels_dict}
                )
        elif(config.dataset_type==DatasetTypes.SEGMENTATION_BINARY_IMAGE_ORGAN):

            train_dataset,valid_dataset,test_dataset,train_collate_fn,val_collate_fn,test_collate_fn=create_train_val_test_datasets(
                config=config,
                dataset_class=SegBinaryGuidedOrganDataset,
                train_kwargs={"split":DictKeys.TRAIN.value,"transforms":augmentation_dict[DictKeys.TRAIN.value],'organ_label_dict':config.organ_labels_dict},
                valid_kwargs={"split":DictKeys.VALID.value,"transforms":augmentation_dict[DictKeys.VALID.value],'organ_label_dict':config.organ_labels_dict},
                test_kwargs={"split":DictKeys.TEST.value,"transforms":augmentation_dict[DictKeys.TEST.value],'organ_label_dict':config.organ_labels_dict}
                )
        train_sampler,valid_sampler,test_sampler=get_data_samplers(
            config=config,
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            test_dataset=test_dataset
            )
        
        data_loaders[DictKeys.TRAIN.value]=DataLoader(dataset=train_dataset,batch_size=config.effective_batch_size,sampler=train_sampler,num_workers=config.num_workers,pin_memory=config.pin_memory,collate_fn=train_collate_fn)
        data_loaders[DictKeys.VALID.value]=DataLoader(dataset=valid_dataset,batch_size=config.effective_batch_size,sampler=valid_sampler,num_workers=config.num_workers,pin_memory=config.pin_memory,collate_fn=val_collate_fn)
        data_loaders[DictKeys.TEST.value]=DataLoader(dataset=test_dataset,batch_size=config.effective_batch_size,sampler=test_sampler,num_workers=config.num_workers,pin_memory=config.pin_memory,collate_fn=test_collate_fn)

    
    return data_loaders

def create_train_val_test_datasets(config:BaseConfigs,dataset_class:BaseDataset,train_kwargs:Dict,valid_kwargs:Dict,test_kwargs:Dict):
    dataset_paths=[]
    if(isinstance(config.dataset_path,list)):
        dataset_paths=config.dataset_path
    elif(isinstance(config.dataset_path,str)):
        dataset_paths=[config.dataset_path]

    train_datasets=[]
    valid_datasets=[]
    test_datasets=[]
    for data_path in dataset_paths:
        if(os.path.exists(os.path.join(data_path,DictKeys.TRAIN.value))):
            train_datasets.append(dataset_class(**{**{"dir_path":data_path},**train_kwargs}))
        if(os.path.exists(os.path.join(data_path,DictKeys.VALID.value))):
            valid_datasets.append(dataset_class(**{**{"dir_path":data_path},**valid_kwargs}))
        if(os.path.exists(os.path.join(data_path,DictKeys.TEST.value))):
            test_datasets.append(dataset_class(**{**{"dir_path":data_path},**test_kwargs}))
    train_collate_fn=train_datasets[-1].collate_fn
    val_collate_fn=valid_datasets[-1].collate_fn
    test_collate_fn=test_datasets[-1].collate_fn
    train_dataset=ConcatDataset(train_datasets)
    valid_dataset=ConcatDataset(valid_datasets)
    test_dataset=ConcatDataset(test_datasets)
    return train_dataset,valid_dataset,test_dataset,train_collate_fn,val_collate_fn,test_collate_fn
def get_data_samplers(config:BaseConfigs,train_dataset,valid_dataset,test_dataset):
    train_sampler=None
    valid_sampler=None
    test_sampler=None
    if(config.sampler_type == SamplerTypes.NONE_SAMPLER):
        pass
    elif(config.sampler_type == SamplerTypes.RANDOM_SAMPLER):
        train_sampler=RandomSampler(train_dataset)
        valid_sampler=SequentialSampler(valid_dataset)
        test_sampler=SequentialSampler(test_dataset)
    elif(config.sampler_type == SamplerTypes.SEQUNETIAL_SAMPLER):
        train_sampler=SequentialSampler(train_dataset)
        valid_sampler=SequentialSampler(valid_dataset)
        test_sampler=SequentialSampler(test_dataset)
    elif(config.sampler_type == SamplerTypes.WEIGHTED_RANDOM_SAMPLER):
        raise NotImplementedError(str.format('{} sampler mode is not implemented for samplers',config.sampler_type.value)) 
    elif(config.sampler_type == SamplerTypes.SUBSET_RANDOM_SAMPLER):
        raise NotImplementedError(str.format('{} sampler mode is not implemented for samplers',config.sampler_type.value))
    elif(config.sampler_type == SamplerTypes.DISTRIBUTED_SAMPLER):
        train_sampler=DistributedSampler(train_dataset,drop_last=True)
        valid_sampler=DistributedSampler(valid_dataset,drop_last=True)
        test_sampler=SequentialSampler(test_dataset)
    return train_sampler,valid_sampler,test_sampler