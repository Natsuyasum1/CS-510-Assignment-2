import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_Cifar(nn.Module):
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
            # self.activation = nn.ELU()

        self.pool = nn.AvgPool2d(2, 1)
        self.conv1 = nn.Conv2d(3, 8, self.kernel_size, stride=1, padding=1)  # check the output dim here
        nn.init.xavier_uniform_(self.conv1.weight)
        self.conv2 = nn.Conv2d(8, 16, self.kernel_size, stride=1, padding=1)
        nn.init.xavier_uniform_(self.conv2.weight)
        self.conv3 = nn.Conv2d(16, 32, self.kernel_size, stride=1, padding=1)
        nn.init.xavier_uniform_(self.conv3.weight)
        self.conv4 = nn.Conv2d(32, 32, self.kernel_size, stride=1, padding=1)
        nn.init.xavier_uniform_(self.conv4.weight)
        self.conv5 = nn.Conv2d(32, 64, self.kernel_size, stride=1, padding=1)
        nn.init.xavier_uniform_(self.conv5.weight)

        self.fc1 = nn.Linear(64*28*28, 128)
        nn.init.xavier_uniform_(self.fc1.weight)
        # self.fc2 = nn.Linear(4096, 1024)    # output
        # self.fc3 = nn.Linear(1024, self.n_classes)    # output

        self.fc2 = nn.Linear(128, self.n_classes)    # output
        nn.init.xavier_uniform_(self.fc2.weight)

    
    def forward(self, x):
        x = x.view(-1, 3, 32, 32)

        x = self.conv1(x)
        x = self.pool(self.activation(x))
        x = self.conv2(x)
        x = self.pool(self.activation(x))
        x = self.conv3(x)
        x = self.pool(self.activation(x))
        x = self.conv4(x)
        x = self.pool(self.activation(x))
        x = self.conv5(x)
        x = self.activation(x)
        # print(x.shape)
        x = x.view(int(x.size(0)), -1)  # flatten
        x = self.fc1(x)
        x = self.fc2(self.activation(x))
        # x = self.fc3(self.activation(x))
        x = F.softmax(x, dim=1)         # apply Softmax

        return x