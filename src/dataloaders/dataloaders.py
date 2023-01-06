# build-in libs
import json
import os
from typing import List
# thrid-party libs
import torch
from tqdm import tqdm
from src.enums import DictKeys
# from src.data_loaders.utils.loaders import listImagesFilesInDir
from src.dataloaders.base_dataset import BaseDataset
import cv2

class SegDataset(BaseDataset):  
    """
    dataloader that read dataset in the directory tree form only
    ---dataset
            |
            |----train
            |      |
            |      |---img
            |      |---lbl
            |
            |----val
            |      |
            |      |---img
            |      |---lbl
            |
            |----test
                   |
                   |---img
                   |---lbl

    """
    def __init__(self, dir_path,split, transforms=None):
        super(SegDataset, self).__init__()
        """
        dataset constructor

        attributes:-

        - dir_path : split path inside the dataset path
        - transforms : augmentation or transformation needs to be done on the image
        - data : list of paths that will be loaded during training item by item

        :param dir_path: dataset path
        :type dir_path: str
        :param split: split name
        :type split: str
        :param transform: transformations to be applied on each image
        :type transform: albumentations.Compose
        """
        self.dir_path = os.path.join(dir_path,split)
        self.transforms=transforms
        self.data = self.get_data_list()
    def get_data_list(self) -> List:

        input_and_labels = []
        images_dir = os.path.join(self.dir_path, DictKeys.IMG.value)
        labels_dir = os.path.join(self.dir_path, DictKeys.LBL.value)
        assert os.path.exists(images_dir), f'image directory @ {images_dir} does not exist'
        assert os.path.exists(labels_dir), f'image directory @ {labels_dir} does not exist'
        images_names = listImagesFilesInDir(images_dir) #list of dataset images names
        print(f"[INFO] loading files names from {images_dir, labels_dir}")
        for file_name in tqdm(images_names):
            img_path = os.path.join(images_dir, file_name)
            lbl_path = os.path.join(labels_dir, file_name)
            input_and_labels.append([file_name, img_path, lbl_path])
        return input_and_labels


    def __getitem__(self, index):
        """_summary_

        :param index: index of the item to be retrieved
        :type index: int
        :return: filename of the image read, image in float tensor, label in float tensor
        :rtype: str,torch.FloatTensor,torch.FloatTensor
        """
        file_name, img_path, lbl_path = self.data[index]
        image = cv2.imread(img_path)
        label = cv2.imread(lbl_path)

        if(self.transforms is not None):
            aug_out = self.transforms(image=image,mask=label)
            image=aug_out[DictKeys.IMAGE.value]
            label=aug_out[DictKeys.MASK.value]

        return file_name, image, label
    

    def __len__(self):
        return len(self.data) 
         
    def collate_fn(self, batch):
        file_names=[item[0] for item in batch]
        imgs=[item[1] for item in batch]
        lbls=[item[2] for item in batch]
        return {
            DictKeys.FILENAMES.value:file_names,
            DictKeys.INPUT.value:torch.stack(imgs).float(),
            DictKeys.Y_TRUE.value:torch.stack(lbls).long()
            }
class SegBinaryDataset(SegDataset):
    def __getitem__(self, index):
        """_summary_

        :param index: index of the item to be retrieved
        :type index: int
        :return: filename of the image read, image in float tensor, label in float tensor
        :rtype: str,torch.FloatTensor,torch.FloatTensor
        """
        file_name, img_path, lbl_path = self.data[index]
        image = cv2.imread(img_path)
        label = cv2.threshold(cv2.imread(lbl_path,cv2.IMREAD_GRAYSCALE),0,1,cv2.THRESH_BINARY)[1]*255
        # print(image.shape,label.shape)

        if(self.transforms is not None):
            aug_out = self.transforms(image=image,mask=label)
            image=aug_out[DictKeys.IMAGE.value]
            label=aug_out[DictKeys.MASK.value]

        return file_name, image, label


class SegDatasetInfer(SegDataset):
    """
    dataloader that read dataset in the directory tree form only
    ---dataset
            |
            |----img

    """
    def get_data_list(self) -> List:
        input_and_labels = []
        images_dir = os.path.join(self.dir_path, DictKeys.IMG.value)
        
        assert os.path.exists(images_dir), f'image directory @ {images_dir} does not exist'
        images_names = listImagesFilesInDir(images_dir) #list of dataset images names
        
        print(f"[INFO] loading files names from {images_dir}")
        for file_name in tqdm(images_names):
            img_path = os.path.join(images_dir, file_name)
            input_and_labels.append([file_name, img_path])

        return input_and_labels
    def __getitem__(self, index):
        """_summary_

        :param index: index of the item to be retrieved
        :type index: int
        :return: filename of the image read, image in float tensor
        :rtype: str,torch.FloatTensor
        """
        file_name, img_path = self.data[index]
        image = cv2.imread(img_path)#Image.open(img_path).convert('RGB')

        if(self.transforms is not None):
            aug_out = self.transforms(image=image)
            image=aug_out[DictKeys.IMAGE.value]

        return file_name, image
    def collate_fn(self, batch):
        file_names=[file_name for file_name,_ in batch]
        imgs=[img for _,img in batch]
        return {
            DictKeys.FILENAMES.value:file_names,
            DictKeys.INPUT.value:torch.stack(imgs).float()
            }

      
class SegBinaryWithClassDataset(SegDataset):
    def __init__(self, dir_path, split,organ_label_dict:dict, transforms=None):
        super().__init__(dir_path, split, transforms)
        self.organ_label_dict=organ_label_dict
    def get_data_list(self) -> List:

        input_and_labels = []
        images_dir = os.path.join(self.dir_path, DictKeys.IMG.value)
        labels_dir = os.path.join(self.dir_path, DictKeys.LBL.value)
        meta_dir = os.path.join(self.dir_path, 'meta')
        assert os.path.exists(images_dir), f'image directory @ {images_dir} does not exist'
        assert os.path.exists(labels_dir), f'image directory @ {labels_dir} does not exist'
        images_names = listImagesFilesInDir(images_dir) #list of dataset images names
        print(f"[INFO] loading files names from {images_dir, labels_dir}")
        for file_name in tqdm(images_names):
            img_path = os.path.join(images_dir, file_name)
            lbl_path = os.path.join(labels_dir, file_name)
            meta_path = os.path.join(meta_dir, file_name.replace('.png','.json'))
            input_and_labels.append([file_name, img_path, lbl_path,meta_path])
        return input_and_labels
    def __getitem__(self, index):
        """_summary_

        :param index: index of the item to be retrieved
        :type index: int
        :return: filename of the image read, image in float tensor, label in float tensor
        :rtype: str,torch.FloatTensor,torch.FloatTensor
        """
        file_name, img_path, lbl_path,meta_path = self.data[index]
        image = cv2.imread(img_path)
        label = cv2.threshold(cv2.imread(lbl_path,cv2.IMREAD_GRAYSCALE),0,1,cv2.THRESH_BINARY)[1]*255
        with open(meta_path,'r')as f:
            
            meta=json.load(f)
        organ_label=self.organ_label_dict[meta['organ']]
        # print(image.shape,label.shape)

        if(self.transforms is not None):
            aug_out = self.transforms(image=image,mask=label)
            image=aug_out[DictKeys.IMAGE.value]
            label=aug_out[DictKeys.MASK.value]

        return file_name, image, label,organ_label
    def collate_fn(self, batch):
        file_names=[item[0] for item in batch]
        imgs=[item[1] for item in batch]
        lbls=[item[2] for item in batch]
        organ_labels=[item[3] for item in batch]
        return {
            DictKeys.FILENAMES.value:file_names,
            DictKeys.INPUT.value:torch.stack(imgs).float(),
            DictKeys.Y_TRUE.value:{
                    "mask":torch.stack(lbls).long(),
                    'organ':torch.tensor(organ_labels).long()
                }
            }
class SegBinaryGuidedOrganDataset(SegBinaryWithClassDataset):
    def __getitem__(self, index):
        """_summary_

        :param index: index of the item to be retrieved
        :type index: int
        :return: filename of the image read, image in float tensor, label in float tensor
        :rtype: str,torch.FloatTensor,torch.FloatTensor
        """
        file_name, img_path, lbl_path,meta_path = self.data[index]
        image = cv2.imread(img_path)
        label = cv2.threshold(cv2.imread(lbl_path,cv2.IMREAD_GRAYSCALE),0,1,cv2.THRESH_BINARY)[1]*255
        # print(image.shape,label.shape)
        with open(meta_path,'r')as f:
            
            meta=json.load(f)
        organ_label=self.organ_label_dict[meta['organ']]
        if(self.transforms is not None):
            aug_out = self.transforms(image=image,mask=label)
            image=aug_out[DictKeys.IMAGE.value]
            label=aug_out[DictKeys.MASK.value]
        organ_tensor=torch.zeros(size=(len(self.organ_label_dict.keys()),image.shape[1],image.shape[2]))
        # print(organ_tensor.shape,image.shape,organ_label)
        organ_tensor[organ_label,:,:]=1
        image=torch.concat(tensors=(image,organ_tensor),dim=0)
        return file_name, image, label
    def collate_fn(self, batch):
        file_names=[item[0] for item in batch]
        imgs=[item[1] for item in batch]
        lbls=[item[2] for item in batch]
        return {
            DictKeys.FILENAMES.value:file_names,
            DictKeys.INPUT.value:torch.stack(imgs).float(),
            DictKeys.Y_TRUE.value:torch.stack(lbls).long()
            }
        


def checkPathExist(dir_path:str)->None:
    '''
    check if a given path is exist
    Parameters:
        dir_path : (str) path to be checked
    Returns:
        None
    '''
    if os.path.exists(dir_path):
        return True
    else:
        raise Exception("[ERROR] THIS PATH", dir_path, " , iS NOT VALID!...")



def listImagesFilesInDir(dir_path:str)->list:

    #check dir_path exists
    if checkPathExist(dir_path):
        files_list = [filename for filename in os.listdir(dir_path) if
                      (str.lower(filename[-4:]) in ['.jpg', '.png']) or (str.lower(filename[-5:]) in ['.jpeg'])]
        # safty check to ensure there are images/masks to work with.
        if files_list == None or len(files_list) <= 0:
            raise Exception("[ERROR] No Images/Masks found in `", dir_path, " , terminating Now...")
    return files_list

