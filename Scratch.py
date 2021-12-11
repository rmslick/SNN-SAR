from pylab import *
import numpy as np

from scipy.ndimage.filters import uniform_filter
from scipy.ndimage.measurements import variance

def lee_filter(img, size):
    img_mean = uniform_filter(img, (size, size))
    img_sqr_mean = uniform_filter(img**2, (size, size))
    img_variance = img_sqr_mean - img_mean**2

    overall_variance = variance(img)

    img_weights = img_variance / (img_variance + overall_variance)
    img_output = img_mean + img_weights * (img - img_mean)
    return img_output


img = np.random.normal(0.5, 0.1, (100,100))
img[:,:50] += 0.25
print(img.shape)
imshow(img, vmin=0, vmax=1, cmap='gray')
print(lee_filter(img, 20).shape)