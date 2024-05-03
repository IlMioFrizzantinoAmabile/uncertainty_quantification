import torch
import torchvision
import numpy as np
import os
import pandas
from PIL import Image
from src.datasets.utils import get_loader, get_subset_data

all_targets = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes',
       'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair',
       'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
       'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
       'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
       'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline',
       'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair',
       'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick',
       'Wearing_Necklace', 'Wearing_Necktie', 'Young'
       ]
all_targets = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes',
       'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair',
       'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
       'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
       'Male', 'Mouth_Slightly_Open', 'Narrow_Eyes', 'No_Beard',
       'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline',
       'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair',
       'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick',
       'Wearing_Necklace', 'Wearing_Necktie', 'Young'
       ]


#targets with occurrence between 10% and 90%  (18 classes)
decent_targets = [
    '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 
    'Bushy_Eyebrows', 'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pointy_Nose', 'Smiling', 
    'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Lipstick', 'Wearing_Necklace', 'Young'
]

#targets with occurrence between 20% and 80%  (17 classes)
easy_targets = [
    'Arched_Eyebrows', 'Attractive', 'Big_Lips', 'Black_Hair', 'Brown_Hair', 
    'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open', 'Oval_Face', 'Pointy_Nose', 'Smiling', 
    'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Lipstick', 'Young'
]


#targets with occurrence between 40% and 60%  (6 classes)
balanced_targets = [
    'Attractive', 'Heavy_Makeup', 'High_Cheekbones', 'Mouth_Slightly_Open', 'Smiling', 'Wearing_Lipstick'
]

class CelebA(torch.utils.data.Dataset):
    def __init__(
        self, 
        transform=None, 
        only_with: list = [], 
        only_without: list = ['Bald', 'Mustache', 'Eyeglasses'], 
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
        #self.image_file_names = os.listdir(self.image_path)
        self.target_path = os.path.join(self.root, "list_attr_celeba.csv")
        dataframe = pandas.read_csv(self.target_path)
        self.target_dataframe = dataframe.loc[:, dataframe.columns != 'image_id']
        #self.target_dataframe = self.target_dataframe.loc[self.target_dataframe['Mustache'] == 1]
        for class_with in only_with:
            self.target_dataframe = self.target_dataframe.loc[self.target_dataframe[class_with] == 1]
        for class_without in only_without:
            self.target_dataframe = self.target_dataframe.loc[self.target_dataframe[class_without] == -1]
        indexes = self.target_dataframe.index

        self.ids = dataframe.loc[indexes, 'image_id'].to_numpy()

        #self.target_dataframe = self.target_dataframe.loc[:, all_targets]
        #self.target_dataframe = self.target_dataframe.loc[:, decent_targets]
        self.target_dataframe = self.target_dataframe.loc[:, easy_targets]
        #self.target_dataframe = self.target_dataframe.loc[:, balanced_targets]
        if False:
            for column in self.target_dataframe.columns:
                occurrences = self.target_dataframe[column].value_counts()
                for i in [-1,1]:
                    if i not in occurrences.keys():
                        occurrences[i] = 0
                #print(f" target {column}: {occurrences[1]} yes,  {occurrences[-1]} no")
                print(f" {100*occurrences[1]/(occurrences[1]+occurrences[-1]):.3f}% of datapoints has target {column}")
        #print(f"Possible targets are: {self.target_dataframe.columns}, --- {len(self.target_dataframe.keys())}")
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
        #return len(self.image_file_names)
        return len(self.ids)
    
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


def get_celeba_ood(
        batch_size = 128,
        shuffle = False,
        only_with: str = 'Bald',
        seed = 0,
        download: bool = False,
        data_path="../datasets",
    ):
    dataset = CelebA(
        only_with = [only_with], 
        only_without = [], 
        download=download, 
        data_path=data_path, 
    )
    test_loader = get_loader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
        seed=seed
    )
    return None, None, test_loader