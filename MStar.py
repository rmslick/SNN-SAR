import SpykeTorch.utils as utils
import SpykeTorch.functional as sf
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import random
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler
import SpykeTorch.snn as snn
import torch
import torchvision




kernels = [	utils.GaborKernel(window_size = 3, orientation = 45+22.5),
            utils.GaborKernel(3, 90+22.5),
            utils.GaborKernel(3, 135+22.5),
            utils.GaborKernel(3, 180+22.5)]
filter = utils.Filter(kernels, use_abs = True)

def time_dim(input):
    return input.unsqueeze(0)


transform = transforms.Compose(
    [transforms.Grayscale(),
    transforms.ToTensor(),
    time_dim,
    filter,
    sf.pointwise_inhibition,
    utils.Intensity2Latency(number_of_spike_bins = 15, to_spike = True)])

mnist_train_dataset = torchvision.datasets.ImageFolder('mstardataset64/train', transform=transform)
mnist_test_dataset = torchvision.datasets.ImageFolder('mstardataset64/test', transform=transform)

mnist_train_loader = torch.utils.data.DataLoader(mnist_train_dataset)
mnist_test_loader = torch.utils.data.DataLoader(mnist_test_dataset)
import numpy as np

plt.style.use('seaborn-white')
plt_idx = 0
sample_idx = random.randint(0, len(mnist_train_dataset) - 1)

# splitting training and testing sets



pool = snn.Pooling(kernel_size = 3, stride = 2)
conv = snn.Convolution(in_channels=4, out_channels=20, kernel_size=30)
stdp = snn.STDP(conv_layer = conv, learning_rate = (0.05, -0.015))
conv.reset_weight()
anti_stdp = snn.STDP(conv_layer = conv, learning_rate = (-0.05, 0.0005))

feature2class = [0] * 10 + [1] * 10
print(feature2class)

print("Starting Reinforcement Learning ...")
for iter in range(20):
    print('\rIteration:', iter, end="")
    for data,targets in mnist_train_loader:
        for x,t in zip(data, targets):
            x = pool(x)
            #print(x.size())
            p = conv(x)
            o, p = sf.fire(p, 20, return_thresholded_potentials=True)
            winners = sf.get_k_winners(p, kwta=1, inhibition_radius=0, spikes=o)
            if len(winners) != 0:
                if feature2class[winners[0][0]] == t:
                    stdp(x, p, o, winners)
                else:
                    anti_stdp(x, p, o, winners)
print()
print("Reinforcement Learning is Done.")


error = 0
silent = 0
total = 0
for data,targets in mnist_test_loader:
    for x,t in zip(data, targets):
        total += 1
        x = pool(x)
        p = conv(x)
        o, p = sf.fire(p, 20, return_thresholded_potentials=True)
        winners = sf.get_k_winners(p, kwta=1, inhibition_radius=0, spikes=o)
        if len(winners) != 0:
            if feature2class[winners[0][0]] != t:
                error += 1
            else:
                #print('Correct')
                pass
        else:
            silent += 1
print('Total',total)
print("         Error:", error/total)
print("Silent Samples:", silent/total)