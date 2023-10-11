import random

import numpy as np
import scipy

from scipy import signal

import cv2

# Harris corner detector
def extract_harris(img, sigma = 1.0, k = 0.05, thresh = 1e-5):
    '''
    Inputs:
    - img:      (h, w) gray-scaled image
    - sigma:    smoothing Gaussian sigma. suggested values: 0.5, 1.0, 2.0
    - k:        Harris response function constant. suggest interval: (0.04 - 0.06)
    - thresh:   scalar value to threshold corner strength. suggested interval: (1e-6 - 1e-4)
    Returns:
    - corners:  (q, 2) numpy array storing the keypoint positions [x, y]
    - C:     (h, w) numpy array storing the corner strength
    '''
    # Convert to float
    img = img.astype(float) / 255.0

    # Compute image gradients
    Ix = signal.convolve2d(img, np.array([-0.5, 0, 0.5]).reshape(1, 3), 'same')
    Iy = signal.convolve2d(img, np.array([-0.5, 0, 0.5]).reshape(3, 1), 'same')

    # You may refer to scipy.signal.convolve2d for the convolution.
    # Do not forget to use the mode "same" to keep the image size unchanged.

    # Compute local auto-correlation matrix
    Ixx = Ix * Ix
    Iyy = Iy * Iy
    Ixy = Ix * Iy

    #blurring the image and computing the coefficients of the auto-correlation matrix
    Mxx = cv2.GaussianBlur(Ixx, ksize=(3, 3), sigmaX= sigma, borderType=cv2.BORDER_REPLICATE)
    Myy = cv2.GaussianBlur(Iyy, ksize=(3, 3), sigmaX= sigma, borderType=cv2.BORDER_REPLICATE)
    Mxy = cv2.GaussianBlur(Ixy, ksize=(3, 3), sigmaX= sigma, borderType=cv2.BORDER_REPLICATE)

    # You may refer to cv2.GaussianBlur for the gaussian filtering (border_type=cv2.BORDER_REPLICATE)

    # Compute Harris response function
    C = Mxx * Myy - (Mxy ** 2) - k * ((Mxx + Myy) ** 2)

    # Detection with threshold
    case1 = C > thresh
    case2 = C == scipy.ndimage.maximum_filter(C, size=(3, 3))

    #finding the corners that respect the previous two conditions
    corners = np.argwhere(case1 & case2)
    corners[:,[0,1]] = corners[:,[1,0]]

    # For the local maximum check, you may refer to scipy.ndimage.maximum_filter to check a 3x3 neighborhood.
    return corners, C

