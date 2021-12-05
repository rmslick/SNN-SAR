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

mstar_train_dataset = ImageFolder("mstardataset/train", transform)
mstar_test_dataset = ImageFolder("mstardataset/test",transform)
mstar_train_dataset = utils.CacheDataset(mstar_train_dataset)
mstar_test_dataset = utils.CacheDataset(mstar_test_dataset)
mstar_train_loader = DataLoader(mstar_train_dataset)
mstar_test_loader = DataLoader(mstar_test_dataset)

# splitting training and testing sets



pool = snn.Pooling(kernel_size = 3, stride = 2)
conv = snn.Convolution(in_channels=4, out_channels=20, kernel_size=24)
stdp = snn.STDP(conv_layer = conv, learning_rate = (0.05, -0.015))
conv.reset_weight()
anti_stdp = snn.STDP(conv_layer = conv, learning_rate = (-0.05, 0.0005))

feature2class = [0] * 10 + [1] * 10
print(feature2class)

print("Starting Reinforcement Learning ...")
for iter in range(20):
    print('\rIteration:', iter, end="")
    for data,targets in mstar_train_loader:
        for x,t in zip(data, targets):
            print('\nImage size ',x.size())
            x = pool(x)
            print('Pool size ',x.size())
            p = conv(x)
            print('Post conv',p.size())
            o, p = sf.fire(p, 20, return_thresholded_potentials=True)
            winners = sf.get_k_winners(p, kwta=1, inhibition_radius=0, spikes=o)
            if len(winners) != 0:
                print(p)
                if feature2class[winners[0][0]] == t:
                    stdp(x, p, o, winners)
                else:
                    anti_stdp(x, p, o, winners)
            break
        break
    break
print()
print("Reinforcement Learning is Done.")
'''

error = 0
silent = 0
total = 0
for data,targets in mstar_test_loader:
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
'''