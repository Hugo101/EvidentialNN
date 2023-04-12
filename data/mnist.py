# to standardize the datasets used in the experiments
# datasets are TinyImageNet, and CIFAR100 (later)
# use create_val_folder() function to convert original Tiny ImageNet structure to structure PyTorch expects

import os
import torch
from torchvision import datasets, transforms, utils
from torch.utils.data import sampler, random_split, DataLoader
from torch.utils.data import TensorDataset, Dataset
from torch.utils.data import Subset, ConcatDataset
from PIL import Image
import random

import sys
data_path = "/home/cxl173430/data/uncertainty_Related/HENN_Git_VScode/HyperEvidentialNN/"
sys.path.insert(1, data_path)
from helper_functions import CustomDataset, AddLabelDataset
import csv
import math  
import time
from sklearn.model_selection import train_test_split
from collections import defaultdict


def train_valid_split(data, valid_perc=0.1, seed=42):
    # generate indices: instead of the actual data we pass in integers instead
    train_indices, valid_indices, _, _ = train_test_split(
        range(len(data)),
        data.targets,
        stratify=data.targets,
        test_size=valid_perc,
        random_state=seed
    )
    # generate subset based on indices
    train_split = Subset(data, train_indices)
    valid_split = Subset(data, valid_indices)
    return train_split, valid_split


class originalMNIST():
    def __init__(
        self, 
        data_dir, 
        num_comp=0, 
        batch_size=128, 
        ratio_train=0.9, 
        pretrain=False, #if using pretrained model, resize img to 224
        num_workers=4, #tune this on different servers
        seed=42):
        self.name = "mnist"
        print('Loading mnist...')
        start_time = time.time()
        self.batch_size = batch_size
        self.pretrain = pretrain
        self.num_workers = num_workers
        self.num_classes = 10 #K  
        self.num_comp = num_comp
        self.kappa = self.num_classes + self.num_comp

        # original MNIST
        train_ds_original = datasets.MNIST(data_dir, train=True, download=True)
        test_ds_original = datasets.MNIST(data_dir, train=False, download=True)

        train_split, valid_split = train_valid_split(train_ds_original, valid_perc=1-ratio_train, seed=seed)
        train_ds_original_n = AddLabelDataset(train_split) #add an aditional label
        valid_ds_original_n = AddLabelDataset(valid_split)
        test_ds_original_n = AddLabelDataset(test_ds_original)
        
        if self.pretrain:
            norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            pre_norm_train = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(), 
                norm])
            pre_norm_test = transforms.Compose([
                transforms.Resize(256),     # Resize images to 256 x 256
                transforms.CenterCrop(224), # Center crop image
                transforms.ToTensor(),
                norm])
        else:
            norm = transforms.Normalize(mean=(0.1307,), std=(0.3081,))
            pre_norm_train = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(), 
                norm])
            pre_norm_test = transforms.Compose([
                transforms.ToTensor(),
                norm])
        train_ds = train_ds_original_n
        valid_ds = valid_ds_original_n
        test_ds = test_ds_original_n
        train_ds = CustomDataset(train_ds, transform=pre_norm_train)
        valid_ds = CustomDataset(valid_ds, transform=pre_norm_test)
        test_ds = CustomDataset(test_ds, transform=pre_norm_test)

        print(f'Data splitted. Train, Valid, Test size: {len(train_ds), len(valid_ds), len(test_ds)}')
        self.train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        self.valid_loader = DataLoader(valid_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
        self.test_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
        
        time_load = time.time() - start_time
        print(f"Loading data finished. Time: {time_load//60:.0f}m {time_load%60:.0f}s")


if __name__ == '__main__':
    data_dir = '/home/cxl173430/data/DATASETS/'
    batch_size = 64
    dataset = originalMNIST(
            data_dir,
            batch_size=batch_size)
    
