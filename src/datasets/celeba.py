import torch
import torchvision
import numpy as np
import os
import pandas
from PIL import Image
from src.datasets.utils import get_loader, get_subset_data

class CelebA(torch.utils.data.Dataset):
    def __init__(
        self, 
        transform=None, 
        cls: list = list(range(40)), 
        download=False,
        data_path="../datasets", 
    ):
        if download:
            raise ValueError("Torch vision CelebA is broken... Download dataset manually from https://www.kaggle.com/datasets/jessicali9530/celeba-dataset")
        self.root = os.path.join(data_path, "celeba")
        self.transform = transform
        self.image_path = os.path.join(self.root, "img_align_celeba/img_align_celeba")
        self.image_file_names = os.listdir(self.image_path)
        self.target_path = os.path.join(self.root, "list_attr_celeba.csv")
        dataframe = pandas.read_csv(self.target_path)
        self.target_dataframe = dataframe.loc[:, dataframe.columns != 'image_id']
        self.ids = dataframe.loc[:, 'image_id'].to_numpy()
        print(f"Possible targets are: {self.target_dataframe.columns}, --- {len(self.target_dataframe.keys())}")
        print(f"There are {len(self.image_file_names)} images and {len(self.ids)} targets")

        self.mean = np.ones((3, 218, 178))
        self.mean[0] *= 129.14
        self.mean[1] *= 108.67
        self.mean[2] *= 97.83
        #self.std = np.ones((3, 218, 178))
        #self.std[0] *= 0.3113
        #self.std[1] *= 0.2914
        #self.std[2] *= 0.2909

        for index in range(len(self)):
        #for index in range(20):
            try:
                image_id = self.ids[index]
                image = Image.open(
                    os.path.join(self.image_path, image_id)
                ).convert('RGB')
            except:
                print(f"An exception occurred loading image -> {self.ids[index]}")
            if str(index+1) not in self.ids[index]:
                print(f"Ahia -> {index} {self.ids[index]}")
        
    def __len__(self):
        return 12661
        #return len(self.image_file_names) #I can't find all 200k images :(
    
    def __getitem__(self, index):
        image_id = self.ids[index]
        image = Image.open(
            os.path.join(self.image_path, image_id)
        ).convert('RGB')
        image = np.array(image)
        image = np.moveaxis(image, 2, 0)
        image = (image - self.mean) #/ self.std
        target = self.target_dataframe.iloc[index].to_numpy()
        if self.transform is not None:
            # torch wants channel dimension before height and width
            image = torch.from_numpy(image)
            image = self.transform(image)
            image = image.permute(1, 2, 0).numpy()
        else:
            image = np.moveaxis(image, 0, 2)
        return image, target
    

def get_celeba(
        batch_size = 128,
        shuffle = False,
        seed = 0,
        download: bool = False,
        data_path="../datasets",
    ):
    dataset = CelebA(
        download=download, 
        data_path=data_path, 
    )
    train_loader, valid_loader = get_loader(
        dataset,
        split_train_val_ratio = 0.9,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
        seed=seed
    )
    return train_loader, valid_loader, None


def get_celeba_augmented(
        batch_size = 128,
        shuffle = False,
        seed = 0,
        download: bool = False,
        data_path="../datasets",
    ):
    train_transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(),
            #torchvision.transforms.RandomVerticalFlip(),
            torchvision.transforms.RandomResizedCrop((218,178),scale=(0.8,1.0),ratio=(0.9,1.1), antialias=True),
            ])
    dataset = CelebA(
        download=download, 
        transform=train_transform,
        data_path=data_path, 
    )
    train_loader, valid_loader = get_loader(
        dataset,
        split_train_val_ratio = 0.9,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
        seed=seed
    )
    return train_loader, valid_loader, None
