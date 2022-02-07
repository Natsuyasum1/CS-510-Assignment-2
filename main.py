from sklearn import datasets
import torch
import numpy as np

from networks.LeNet import LeNet_Cifar
from networks.CNN import CNN_Cifar
from datasets.cifar10 import CIFAR10_Dataset
from trainer.trainer import Trainer
import pandas as pd

import matplotlib.pyplot as plt

# parameters
# lr = 0.001
# loss = 'MSE'
# loss = 'CrossEntropy'
# activation = 'sigmoid'
# activation = 'tanh'
# activation = 'relu'

# parameters don't change
epochs = 400
batch_size = 128
device = torch.device("cuda")

# --------------- question 1.a ------------------ #
# lrs = [0.1, 0.01, 0.001]
# activations = ['sigmoid', 'tanh']
# losses = ['MSE', 'CrossEntropy']

# train_accs, test_accs, titles = [], [], []

# dataset = CIFAR10_Dataset('./datasets')
# trainloader, testloader = dataset.loaders(batch_size=batch_size)

# for lr in lrs:
#     for activation in activations:
#         for loss in losses:
#             model = LeNet_Cifar(activation, 5).to(device)
#             trainer = Trainer(model, device, loss)
#             l, train_a, test_a = trainer.train(trainloader, lr, epochs, testloader)
#             train_accs.append(train_a)
#             test_accs.append(test_a)
#             # title = 'lr = ' + str(lr) + ', activation = ' + activation + ', loss = ' + loss 
#             title = str(lr) + ' ' + activation + ' ' + loss
#             titles.append(title)

# fig, axs = plt.subplots(3, 4)
# count = 0
# for i in range(3):
#     for j in range(4):
#         axs[i, j].plot(train_accs[count], label='train_acc', c='blue')
#         axs[i, j].plot(test_accs[count], label='test_acc', c='orange')
#         axs[i, j].set_title(titles[count], y=1.0, pad=-14, loc='left')
#         count += 1
# # for ax in axs.flat:
# #     ax.set(xlabel='epochs', ylabel='acc')
# # for ax in axs.flat:
#     # ax.label_outer()
# plt.legend()
# plt.show()
        
# --------------- question 1.b ------------------ #
# lr = 0.001
# loss = 'CrossEntropy'
# activation = 'tanh'

# dataset = CIFAR10_Dataset('./datasets')
# trainloader, testloader = dataset.loaders(batch_size=batch_size)
# model = LeNet_Cifar(activation, 5).to(device)
# trainer = Trainer(model, device, loss)
# _, _, _ = trainer.train(trainloader, lr, epochs, testloader)
# features, targets = trainer.examine_features(trainloader)

# features = features.reshape(10, 120)

# count = 0
# fig, axs = plt.subplots(10, 1)
# for i in range(10):
#     # for j in range(5):
#     axs[i].imshow(features[count].reshape(1, 120), cmap='gray')
#     axs[i].set_title(str(targets[count]), fontdict={'fontsize': 10}, x=0, y=1)
#     axs[i].set_xticks([])
#     axs[i].set_yticks([])
#     count += 1
# plt.show()

# ---------------- question 2 ------------------ #
# lr = 0.001
# loss = 'CrossEntropy'
# activation = 'relu'

# dataset = CIFAR10_Dataset('./datasets')
# trainloader, testloader = dataset.loaders(batch_size=batch_size)
# model = LeNet_Cifar(activation, 3).to(device)
# trainer = Trainer(model, device, loss)
# l, train_a, test_a = trainer.train(trainloader, lr, epochs, testloader)

# plt.plot(train_a, label='train_acc', c='blue')
# plt.plot(test_a, label='test_acc', c='orange')
# plt.legend()
# plt.show()


# ---------------- question 3 ------------------ #
lr = 0.001
loss = 'CrossEntropy'
# loss = 'MSE'
activation = 'tanh'

dataset = CIFAR10_Dataset('./datasets')
trainloader, testloader = dataset.loaders(batch_size=batch_size)
model = CNN_Cifar(activation, 3).to(device)
trainer = Trainer(model, device, loss)
l, train_a, test_a = trainer.train(trainloader, lr, epochs, testloader)

plt.plot(train_a, label='train_acc', c='blue')
plt.plot(test_a, label='test_acc', c='orange')
plt.legend()
plt.show()