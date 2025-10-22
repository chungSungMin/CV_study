import torch 
from torchvision import datasets, transforms
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from utils.augmentation import get_train_transforms, get_val_transforms, apply_alb_transforms
import os 

def get_dataloader(train_dataset, val_dataset, test_dataset):
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False
    )
    
    return train_dataloader, val_dataloader, test_dataloader


def get_dataset(path):
    
    train_pipline = get_train_transforms()
    val_pipline = get_val_transforms()

    transform_train = lambda pil_img : apply_alb_transforms(pil_img, train_pipline)
    transform_val = lambda pil_img : apply_alb_transforms(pil_img, val_pipline)

    full_dataset_train = datasets.ImageFolder(
        root=path,
        transform=transform_train
    )

    full_dataset_val = datasets.ImageFolder(
        root=path,
        transform=transform_val
    )

    total_dataset = len(full_dataset_train)
    train_size = int(total_dataset * 0.7)
    val_size = int(total_dataset * 0.2)
    test_size = total_dataset - train_size - val_size

    print(f"Spliting into Train : {train_size} | Val : {val_size} | Test : {test_size}")

    generator = torch.Generator().manual_seed(42)

    train_dataset, _, _ = random_split(
        full_dataset_train,
        [train_size, val_size, test_size],
        generator=generator
    )

    generator.manual_seed(42)
    _, val_dataset, test_dataset = random_split(
        full_dataset_val,
        [train_size, val_size, test_size],
        generator=generator
    )

    return train_dataset, val_dataset, test_dataset



