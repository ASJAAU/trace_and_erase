from torchvision.transforms import v2 as torch_transforms
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode
import torch
import pandas as pd
import os
import numpy as np

#Data Preprocessing
def get_transforms(type='train', augments={}):

    transform_dict={
        "RandomHorizontalFlip": torch_transforms.RandomHorizontalFlip
    }
    #Potential label formatting (for throughput optimization)
    if type == "label":
        return torch_transforms.Compose([
            torch_transforms.ToDtype(torch.float32, scale=True),
        ])
    
    #Input transforms for training
    elif type == "train":
        transforms = [
            torch_transforms.ToImage(),
            torch_transforms.ToDtype(torch.float32, scale=True),
            ]
        for aug in augments.keys():
            transforms.append(transform_dict[aug](**augments[aug][0]))
        return torch_transforms.Compose(transforms)
    
    #Input transforms for evaluation
    elif type == "test" or type == "valid":
        transforms = [
            torch_transforms.ToImage(),
            torch_transforms.ToDtype(torch.float32, scale=True),
            ]
        return torch_transforms.Compose(transforms)
class HarborfrontDataset(Dataset):
    CLASS_LIST = {
        0: "human",
        1: "bicycle",
        2: "motorcycle",
        3: "vehicle"
    }
    def __init__(self, data_split, root, 
                 transform=None, 
                 target_transform=None,
                 in_channels = 3, 
                 classes=CLASS_LIST.values(), 
                 binary_labels=False,
                 classwise=True, 
                 device='cpu', 
                 verbose=False) -> None:
        if verbose:
            print(f'Loading "{data_split}"')
            print(f'Target Classes {classes}')

        #Transform objects
        self.transform = transform
        self.target_transform = target_transform

        #Set read format
        if in_channels == 1:
            self.read_mode = ImageReadMode.GRAY
        elif in_channels == 2:
            self.read_mode = ImageReadMode.GRAY_ALPHA
        elif in_channels == 3:
            self.read_mode = ImageReadMode.RGB
        elif in_channels == 4:
            self.read_mode = ImageReadMode.RGB_ALPHA
        else:
            raise Exception(f"Unsupported input_channel_depth '{in_channels}' ")

        #Use target device for storage
        self.device = device

        #Load dataset file
        self.root = root
        data = pd.read_csv(data_split, sep=";")

        #Isolate desired classes
        if "all" in classes:
            print(f"Class: 'all' listed in config. Classes: ]{','.join(self.CLASS_LIST.values())}] will be used.")
            self.classes = list(self.CLASS_LIST.values())
        else:
            for c in classes:
                assert c in self.CLASS_LIST.values(), f"{c} is not a known class. \n Known classes:{','.join(self.CLASS_LIST.values())}" 
            self.classes = list(classes)

        #Create dataset of relevant info
        dataset = {"file_name": list(data['file_name'])}
        for cls in self.classes:
            dataset[f'{cls}'] = data[f'{cls}']

        #Reconstruct Dataframe with only training data
        self.dataset = pd.DataFrame(dataset)

        #Join paths with root
        self.images = self.dataset.apply(lambda x: os.path.join(root, x["file_name"]), axis=1)
        

        #Grab labels
        self.dataset["labels"] = self.dataset.apply(lambda x: np.asarray([int(x[g]) for g in self.classes], dtype=np.int8), axis=1)

        #Format labels
        if binary_labels:
            self.dataset["labels"] = self.dataset.apply(lambda x: np.asarray([1 if g>=1 else 0 for g in x["labels"]], dtype=np.int8), axis=1)
        if not classwise:
            self.dataset["labels"] = self.dataset.apply(lambda x: [np.sum(x["labels"], dtype=np.int8)], axis=1)
        
        if verbose:
            print(f'Successfully loaded "{data_split}" as {self.__repr__()}')
            print("")

    def __len__(self):
        return len(self.dataset.index)

    def __getitem__(self, idx):
        image = read_image(self.images.iloc[idx], self.read_mode)
        label = torch.Tensor(self.dataset["labels"].iloc[idx])

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image.to(self.device), label.to(self.device)
    
    def __repr__(self):
        return self.dataset.__str__()

    def __str__(self):
        sample=self.__getitem__(0)
        return f'Harborfront Dataset (Pytorch)' + f"\nExample input: {sample[0].shape} \n{sample[0]}" + f"\nExample label: {sample[1].shape} \n{sample[1]}"

if __name__ == '__main__':
    from misc import get_config
    from torch.utils.data import DataLoader
    cfg = get_config("configs/base.yaml")
    subset = "test"

    #Set transforms
    transforms = get_transforms(subset)
    label_transforms = get_transforms("label")

    #Load dataset
    dataset = HarborfrontDataset(
        data_split=cfg["data"][subset], 
        root=cfg["data"]["root"], 
        transform=transforms, 
        target_transform=transforms, 
        classes=cfg["data"]["classes"], 
        binary_labels=cfg["data"]["binary_cls"],
        classwise=cfg["data"]["classwise"], 
        device="cpu", 
        verbose=True)

    #Load dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=cfg["training"]["batch_size"],
    )

    #Print dummy sample
    dummy_sample = next(iter(dataloader))
    print(f"Input Tensor = {dummy_sample[0].shape}")
    print(f"Label Tensor = {dummy_sample[1].shape}")

