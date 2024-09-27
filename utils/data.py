from torchvision.transforms import v2 as torch_transforms
from torch.utils.data import DataLoader
import torch

#Data Preprocessing
def get_transforms(type='train', augments={}):

    transform_dict={
        "horizontal_flip": torch_transforms.RandomHorizontalFlip
    }

    #Define base transforms
    transforms = [
        torch_transforms.ToDtype(torch.float32, scale=True),
        torch_transforms.ToTensor(),
    ]

    #Return label transforms
    if type == "label":
        return torch_transforms.Compose([torch_transforms.ToTensor()])
    
    elif type == "train":
        for aug in augments.keys():
            transforms.extend(transform_dict(aug)(**augments[aug]))
        return torch_transforms.Compose(transforms)
    
    elif type == "test" or type == "valid":
        return torch_transforms.Compose(transforms)

#Dataloaders
def get_dataset(cfg, split, transforms=None, label_transforms=None):
    raise NotImplemented

def get_dataloader(cfg, dataset):
    raise NotImplemented

