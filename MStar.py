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
from scipy.ndimage.filters import uniform_filter
from scipy.ndimage.measurements import variance
#https://stackoverflow.com/questions/39785970/speckle-lee-filter-in-python
def lee_filter(img, size):
    img_mean = uniform_filter(img, (size, size))
    img_sqr_mean = uniform_filter(img**2, (size, size))
    img_variance = img_sqr_mean - img_mean**2

    overall_variance = variance(img)

    img_weights = img_variance / (img_variance + overall_variance)
    img_output = img_mean + img_weights * (img - img_mean)
    return img_output



#https://www.tutorialguruji.com/python/speckle-lee-filter-in-python/#:~:text=I%E2%80%99m%20not%20familiar%20with%20SAR%20so%20I%20don%E2%80%99t%20know%20if%20Lee%20filter%20has%20some%20features%20that%20make%20it%20particularly%20good%20for%20speckle%20in%20SAR%2C%20but%20you%20may%20want%20to%20look%20into%20modern%20edge-aware%20denoisers%2C%20like%20guided%20filter%20or%20bilateral%20filter.
kernels = [	utils.GaborKernel(window_size = 5, orientation = 45+22.5),
            utils.GaborKernel(5, 90+22.5),
            utils.GaborKernel(5, 135+22.5),
            utils.GaborKernel(5, 180+22.5)]
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
conv = snn.Convolution(in_channels=4, out_channels=30, kernel_size=29)
stdp = snn.STDP(conv_layer = conv, learning_rate = (0.001, -0.015))
conv.reset_weight()
anti_stdp = snn.STDP(conv_layer = conv, learning_rate = (-0.05, 0.0005))

feature2class = [0] * 15 + [1] * 15
print(feature2class)

print("Starting Reinforcement Learning ...")
for iter in range(25):
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