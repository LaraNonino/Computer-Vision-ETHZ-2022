import time
import os
import random
import math
import torch
import numpy as np

# run `pip install scikit-image` to install skimage, if you haven't done so
from skimage import io, color
from skimage.transform import rescale

BATCHSIZE = 4096

def distance(x, X):
    dist = torch.cdist(x.unsqueeze(0), X)  # x.shape = [3]  --> x.unsqueeze(0).shape = [1, 3]
    return dist

def distance_batch(x, X):
    dist = torch.cdist(x, X)  # x.shape = [BATCHSIZE, 3]
    return dist

def gaussian(dist, bandwidth):
    weight = torch.exp(-dist ** 2 / (2 * (bandwidth ** 2)))
    return weight

def update_point(weight, X):
    # weight.shape = [1, 3675]
    # X.shape = [3675, 3]
    num = weight.mm(X)  # num.shape = [1, 3]
    den = weight.sum(axis=1)  # den.shape = [1]
    w_mean = num / den  # w_mean.shape = [1, 3]
    return w_mean

def update_point_batch(weight, X):
    # weight.shape = [BATCHSIZE, 3675]
    # X.shape = [3675, 3]
    num = torch.transpose(weight.mm(X), 0, 1)  # num.shape = [3, BATCHSIZE]
    den = weight.sum(axis=1)  # den.shape = [BATCHSIZE]
    w_mean = torch.transpose(num / den, 0, 1)  # w_mean.shape = [BATCHSIZE, 3]
    return w_mean

def meanshift_step(X, bandwidth=2.5):
    X_ = X.clone()
    for i, x in enumerate(X):
        dist = distance(x, X)
        weight = gaussian(dist, bandwidth)
        X_[i] = update_point(weight, X)
    return X_

def meanshift_step_batch(X, bandwidth=2.5):
    X_ = torch.empty((0, 3))
    data_loader = torch.utils.data.DataLoader(X, batch_size=BATCHSIZE)

    for x in data_loader:
        # x.shape = [BATCHSIZE, 3]
        dist = distance_batch(x, X)  # dist.shape = [BATCHSIZE, 3675]
        weight = gaussian(dist, bandwidth)  # weight.shape = [BATCHSIZE, 3675]
        X_ = torch.cat((X_, update_point_batch(weight, X)), 0)
    return X_

def meanshift(X):
    X = X.clone()
    for _ in range(20):
        # X = meanshift_step(X)   # slow implementation
        X = meanshift_step_batch(X)   # fast implementation
    return X

scale = 0.25    # downscale the image to run faster

# Load image and convert it to CIELAB space
image = rescale(io.imread('cow.jpg'), scale, channel_axis=-1)
image_lab = color.rgb2lab(image)
shape = image_lab.shape # record image shape
image_lab = image_lab.reshape([-1, 3])  # flatten the image

# Run your mean-shift algorithm
t = time.time()
X = meanshift(torch.from_numpy(image_lab)).detach().cpu().numpy()
# X = meanshift(torch.from_numpy(data).cuda()).detach().cpu().numpy()  # you can use GPU if you have one
t = time.time() - t
print ('Elapsed time for mean-shift: {}'.format(t))

# Load label colors and draw labels as an image
colors = np.load('colors.npz')['colors']
colors[colors > 1.0] = 1
colors[colors < 0.0] = 0

centroids, labels = np.unique((X / 4).round(), return_inverse=True, axis=0)

result_image = colors[labels].reshape(shape)
result_image = rescale(result_image, 1 / scale, order=0, channel_axis=-1)     # resize result image to original resolution
result_image = (result_image * 255).astype(np.uint8)
io.imsave('result.png', result_image)
