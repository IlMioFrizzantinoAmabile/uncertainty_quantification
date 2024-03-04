import torch
import torchvision
import numpy as np
import os
import pandas
from PIL import Image
from src.datasets.utils import get_loader, get_subset_data

easy_targets = [
    'Arched_Eyebrows', 'Bushy_Eyebrows', 
    'Bags_Under_Eyes', 'Eyeglasses', 'Narrow_Eyes', 'Big_Lips', 'Big_Nose', 
    'Bald', 'Receding_Hairline', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Hat', 
    'Double_Chin', 'Mouth_Slightly_Open', 'Mustache', 'No_Beard',
    'Pale_Skin', 'Rosy_Cheeks', 'Wearing_Earrings', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie'
]

#targets with occurrence between 40% and 60%
balanced_targets = [
    'Attractive', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open', 'Smiling', 'Wearing_Lipstick'
]

class CelebA(torch.utils.data.Dataset):
    def __init__(
        self, 
        transform=None, 
        cls: list = list(range(40)), 
        download=False,
        data_path="../datasets", 
    ):
        if download:
            raise ValueError("Torch vision CelebA is broken... \
                             Download dataset manually from https://www.kaggle.com/datasets/jessicali9530/celeba-dataset \
                             Then unzip it with 'unzip archive.zip -d {data_path}/celeba")
        self.root = os.path.join(data_path, "celeba")
        self.transform = transform
        self.image_path = os.path.join(self.root, "img_align_celeba/img_align_celeba")
        self.image_file_names = os.listdir(self.image_path)
        self.target_path = os.path.join(self.root, "list_attr_celeba.csv")
        dataframe = pandas.read_csv(self.target_path)
        self.target_dataframe = dataframe.loc[:, dataframe.columns != 'image_id']
        self.target_dataframe = self.target_dataframe.loc[:, balanced_targets]
        for column in self.target_dataframe.columns:
            occurrences = self.target_dataframe[column].value_counts()
            #print(f" target {column}: {occurrences[1]} yes,  {occurrences[-1]} no")
            print(f" {100*occurrences[1]/(occurrences[1]+occurrences[-1]):.3f}% of datapoints has target {column}")
        self.ids = dataframe.loc[:, 'image_id'].to_numpy()
        print(f"Possible targets are: {self.target_dataframe.columns}, --- {len(self.target_dataframe.keys())}")
        #print(f"There are {len(self.image_file_names)} images and {len(self.ids)} targets")

        self.mean = np.ones((3, 218, 178))
        self.mean[0] *= 127.85 #129.14
        self.mean[1] *= 107.33 #108.67
        self.mean[2] *= 96.61  #97.83
        self.std = np.ones((3, 218, 178))
        self.std[0] *= 78.94
        self.std[1] *= 73.79
        self.std[2] *= 73.61

        if False:
            for index in range(len(self)):
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
        return len(self.image_file_names)
    
    def __getitem__(self, index):
        image_id = self.ids[index]
        image = Image.open(
            os.path.join(self.image_path, image_id)
        ).convert('RGB')
        image = np.array(image)
        image = np.moveaxis(image, 2, 0)
        image = (image - self.mean) / self.std
        if self.transform is not None:
            # torch wants channel dimension before height and width
            image = torch.from_numpy(image)
            image = self.transform(image)
            image = image.permute(1, 2, 0).numpy()
        else:
            image = np.moveaxis(image, 0, 2)
        target = self.target_dataframe.iloc[index].to_numpy()
        target = (target * 0.5) +  1 #move targets from {-1,1} to {0,1} 
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
    test_size = 1000
    trainvalid_size = len(dataset) - test_size
    dataset_trainvalid, dataset_test = torch.utils.data.random_split(
        dataset, (trainvalid_size, test_size), generator=torch.Generator().manual_seed(0)
    )
    train_loader, valid_loader = get_loader(
        dataset_trainvalid,
        split_train_val_ratio = 0.9,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
        seed=seed
    )
    test_loader = get_loader(
        dataset_test,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
        seed=seed
    )
    return train_loader, valid_loader, test_loader


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
            torchvision.transforms.RandomResizedCrop((218,178),scale=(0.7,1.0),ratio=(1.0,1.0), antialias=True),
            ])
    dataset = CelebA(
        download=download, 
        transform=train_transform,
        data_path=data_path, 
    )
    test_size = 1000
    trainvalid_size = len(dataset) - test_size
    dataset_trainvalid, dataset_test = torch.utils.data.random_split(
        dataset, (trainvalid_size, test_size), generator=torch.Generator().manual_seed(0)
    )
    train_loader, valid_loader = get_loader(
        dataset_trainvalid,
        split_train_val_ratio = 0.9,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
        seed=seed
    )
    test_loader = get_loader(
        dataset_test,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
        seed=seed
    )
    return train_loader, valid_loader, test_loader
