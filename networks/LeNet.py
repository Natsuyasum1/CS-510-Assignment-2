import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet_Cifar(nn.Module):
    """
    This is supposed to be a description, but im too lazy to write one.
    """
    def __init__(self, activation, kernel_size: int, n_classes=10):
        super().__init__()
        self.n_classes = n_classes
        self.kernel_size = kernel_size

        # specify the activation func
        if activation == 'sigmoid':
            # self.activation = lambda x: F.sigmoid(x)
            self.activation = nn.Sigmoid()
        elif activation == 'tanh':
            # self.activation = lambda x: F.tanh(x)
            self.activation = nn.Tanh()
        elif activation == 'relu':
            # self.activation = lambda x: F.relu(x)
            self.activation = nn.ReLU()

        self.pool = nn.AvgPool2d(2, 2)
        self.conv1 = nn.Conv2d(3, 6, self.kernel_size, stride=1, padding=0)  # check the output dim here
        self.conv2 = nn.Conv2d(6, 16, self.kernel_size, stride=1, padding=0)
        self.conv3 = nn.Conv2d(16, 120, self.kernel_size, stride=1, padding=0)
        if kernel_size == 3:
            self.fc1 = nn.Linear(120*4*4, 128)
            self.fc2 = nn.Linear(128, self.n_classes)    # output

        elif kernel_size == 5:
            self.fc1 = nn.Linear(120, 84)
            self.fc2 = nn.Linear(84, self.n_classes)    # output

    
    def forward(self, x):
        x = x.view(-1, 3, 32, 32)

        x = self.conv1(x)
        x = self.pool(self.activation(x))
        x = self.conv2(x)
        x = self.pool(self.activation(x))
        x = self.conv3(x)
        # x = self.pool(self.activation(x))
        x = self.activation(x)
        x = x.view(int(x.size(0)), -1)  # flatten
        x = self.fc1(x)
        x = self.fc2(self.activation(x))
        x = F.softmax(x, dim=1)         # apply Softmax

        return x