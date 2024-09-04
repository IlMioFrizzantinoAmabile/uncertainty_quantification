from pathlib import Path
import torch
import torchvision
import numpy as np
from src.datasets.utils import get_loader, get_subset_data

    
resize = torchvision.transforms.Resize((256, 256))
def my_transform(image):
    image = image.convert('RGB')
    image = resize(image)
    #image = np.asarray(image)
    #print(image.shape)
    return image
def my_collate(batch):
    data = [resize(item[0]) for item in batch]
    #data = [np.asarray(resize(item[0])) for item in batch]
    #print([d.shape for d in data])
    target = [item[1] for item in batch]
    return torch.Tensor(np.asarray(data, dtype=np.float32)), torch.Tensor(target)
    #return np.asarray(data, dtype=np.float32), np.asarray(target)

def get_imagenet(
        batch_size = 128,
        shuffle = False,
        seed = 0,
        download: bool = True,
        data_path="../../../imagenet/ILSVRC/Data/CLS-LOC/train",
    ):
    n_classes = 1000
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    def normalize(img):
        img = np.array(img, dtype=np.float32)
        img = (img / 255. - mean) / std
        return img
    train_transform = torchvision.transforms.Compose(
            #[torchvision.transforms.RandomResizedCrop(224), torchvision.transforms.RandomHorizontalFlip(), normalize]
            #[torchvision.transforms.Resize((256, 256)), normalize]
            [torchvision.transforms.Resize((224, 224)), normalize]
        )
    def target_transform(y):
        return torch.nn.functional.one_hot(torch.tensor(y), n_classes).numpy()
    
    def numpy_collate_fn(batch):
        data, target = zip(*batch)
        data = np.stack(data)
        target = np.stack(target)
        data = torch.from_numpy(data)
        target = torch.from_numpy(target)
        return data, target
    

    #dataset = torchvision.datasets.ImageFolder(data_path, transform=my_transform)
    dataset = torchvision.datasets.ImageFolder(
        data_path, transform=train_transform, target_transform=target_transform
    )
    #idx = [i for i in range(len(dataset)) if dataset.imgs[i][1] in [dataset.class_to_idx['n02110185'], dataset.class_to_idx['n02123394']] ]  #n01440764  n01698640  n01860187
    # build the appropriate subset
    #dataset = torch.utils.data.Subset(dataset, idx)

    train_loader, valid_loader = get_loader(
        dataset,
        split_train_val_ratio = 0.9,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
        seed=seed,
        collate_fn=numpy_collate_fn
    )
    return train_loader, valid_loader, None


### list of folder-name to class ---> https://gist.github.com/aaronpolhamus/964a4411c0906315deb9f4a3723aac57
# n07753275 - pineapple
# n09472597 - volcano
# n02007558 - flamingo
# n01704323 - triceratops
# n03841143 - odometer
# n03666591 - lighter
# n01944390 - snail
# n04005630 - prison
# n02980441 - castle
# n03160309 - dam
# n07718747 - artichoke
# n07565083 - menu
# n03857828 - oscilloscope
# n07880968 - burrito
# n03888257 - parachute
# n07831146 - carbonara
# n02791124 - barber chair

def get_imagenet_subclasses(
        batch_size = 128,
        shuffle = False,
        seed = 0,
        download: bool = True,
        data_path="../../../imagenet/ILSVRC/Data/CLS-LOC/train",
        only_with = None,
        only_without = None
    ):
    n_classes = 1000
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    dataset = torchvision.datasets.ImageFolder(data_path)
    assert only_with is None or only_without is None # not together
    if only_with is not None:
        classes = [dataset.class_to_idx[c] for c in only_with]
    else:
        classes = list(range(n_classes))
    if only_without is not None:
        only_without_idx = [dataset.class_to_idx[c] for c in only_without]
        for c in only_without:
            classes.remove(dataset.class_to_idx[c])
    old_idx_to_new_idx = {}
    for new_idx, old_idx in enumerate(classes):
        old_idx_to_new_idx[old_idx] = new_idx
    n_classes = len(classes)
    def target_transform(y):
        return torch.nn.functional.one_hot(torch.tensor(old_idx_to_new_idx[y]), n_classes).numpy()
    
    def normalize(img):
        img = np.array(img, dtype=np.float32)
        img = (img / 255. - mean) / std
        return img
    train_transform = torchvision.transforms.Compose(
            #[torchvision.transforms.RandomResizedCrop(224), torchvision.transforms.RandomHorizontalFlip(), normalize]
            #[torchvision.transforms.Resize((256, 256)), normalize]
            [torchvision.transforms.Resize((224, 224)), normalize]
        )
    
    dataset = torchvision.datasets.ImageFolder(
        data_path, transform=train_transform, target_transform=target_transform
    )
    if only_with is not None:
        idx = [i for i in range(len(dataset)) if dataset.imgs[i][1] in classes] 
    if only_without is not None:
        idx = [i for i in range(len(dataset)) if dataset.imgs[i][1] not in only_without_idx]
    #idx = [i for i in range(len(dataset)) if dataset.imgs[i][1] in [dataset.class_to_idx['n02110185'], dataset.class_to_idx['n02123394']] ]  #n01440764  n01698640  n01860187
    dataset = torch.utils.data.Subset(dataset, idx)


    def numpy_collate_fn(batch):
        data, target = zip(*batch)
        data = np.stack(data)
        target = np.stack(target)
        data = torch.from_numpy(data)
        target = torch.from_numpy(target)
        return data, target

    train_loader, valid_loader = get_loader(
        dataset,
        split_train_val_ratio = 0.9,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
        seed=seed,
        collate_fn=numpy_collate_fn
    )
    return train_loader, valid_loader, None


def get_imagenet_id(
        batch_size = 128,
        shuffle = False,
        seed = 0,
        download: bool = True,
        data_path="../../../imagenet/ILSVRC/Data/CLS-LOC/train",
    ):
    return get_imagenet_subclasses(
        batch_size = batch_size,
        shuffle = shuffle,
        seed = seed,
        download = download,
        data_path = data_path,
        only_with = None, 
        only_without = ["n07753275", "n07831146", "n07565083", "n09472597", "n02007558", "n01704323", "n03841143", "n03666591", "n02980441", "n03888257"]
        # these classes folder names correspond to ["pineapple", "carbonara", "menu", "volcano", "flamingo", "triceratops", "odometer", "lighter", "castle", "parachute"]
    )

def get_imagenet_ood(
        batch_size = 128,
        shuffle = False,
        seed = 0,
        download: bool = True,
        data_path="../../../imagenet/ILSVRC/Data/CLS-LOC/train",
        ood_class = "pineapple"
    ):
    # 10 nice classes
    keys = ["pineapple", "carbonara", "menu", "volcano", "flamingo", "triceratops", "odometer", "lighter", "castle", "parachute"]
    values = ["n07753275", "n07831146", "n07565083", "n09472597", "n02007558", "n01704323", "n03841143", "n03666591", "n02980441", "n03888257"]
    dictionary = dict(zip(keys, values))

    return get_imagenet_subclasses(
        batch_size = batch_size,
        shuffle = shuffle,
        seed = seed,
        download = download,
        data_path = data_path,
        only_with = [dictionary[ood_class]], 
        only_without = None
    )
