from turtle import down
import torch
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import ssl
# import random

class CIFAR10_Dataset:
    def __init__(self, root:str) -> None:
        ssl._create_default_https_context = ssl._create_unverified_context
        transform = transforms.ToTensor()

        # self.train_set = CIFAR10(root=root, train=True, download=True, transform=transform)
        # self.test_set = CIFAR10(root=root, train=False, download=True, transform=transform)
        self.train_set = myCIFAR10(root=root, train=True, download=True, transform=transform)
        self.test_set = myCIFAR10(root=root, train=False, download=True, transform=transform)


    def loaders(self, batch_size: int, shuffle_train=True, shuffle_test=False) -> (DataLoader, DataLoader):
        trainloader = DataLoader(self.train_set, batch_size=batch_size, shuffle=shuffle_train)
        testloader = DataLoader(self.test_set, batch_size=batch_size, shuffle=shuffle_test)
        return trainloader, testloader

class myCIFAR10(CIFAR10):
    def __init__(self, *args, **kwargs):
        super(myCIFAR10, self).__init__(*args, **kwargs)
        self.transformed_data = []
        for i in range(len(self.data)):
            # print(self.data[i].shape)
            img = Image.fromarray(self.data[i])
            self.transformed_data.append(self.transform(img))
        
    def __getitem__(self, index: int):
        return self.data[index], self.targets[index]

