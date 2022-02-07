import numpy as np

# a = lambda x: np.abs(x)

# print(a(-3))
import matplotlib.pyplot as plt
from torch import arange

a = []

for i in range(12):
    a.append(arange(i, i+10))

fig, axs = plt.subplots(2, 2)

count = 0

for i in range(2):
    for j in range(2):
        axs[i, j].plot(a[count], label='a')
        axs[i, j].plot(a[count+1], label='b')
        axs[i, j].plot(a[count+2], label='c')
        count += 3


plt.legend()
plt.show()
