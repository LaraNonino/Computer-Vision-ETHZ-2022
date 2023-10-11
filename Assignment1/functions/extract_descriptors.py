import numpy as np
import cv2

def filter_keypoints(img, keypoints, patch_size = 9):
    
    Xs = np.array(keypoints[:, 1])
    Ys = np.array(keypoints[:, 0])
    offset = int(np.floor(patch_size / 2.0))
    
    #masking put the poins too close to the borders
    mask = (Xs > offset) & (Xs < img.shape[0] - offset) & \
            (Ys > offset) & (Ys < img.shape[1] - offset)
    mask = np.column_stack((mask, mask))
    #returing the points that weren't masked out
    return keypoints[mask].reshape(-1,2)

# The implementation of the patch extraction is already provided here
def extract_patches(img, keypoints, patch_size = 9):
    '''
    Extract local patches for each keypoint
    Inputs:
    - img:          (h, w) gray-scaled images
    - keypoints:    (q, 2) numpy array of keypoint locations [x, y]
    - patch_size:   size of each patch (with each keypoint as its center)
    Returns:
    - desc:         (q, patch_size * patch_size) numpy array. patch descriptors for each keypoint
    '''
    h, w = img.shape[0], img.shape[1]
    img = img.astype(float) / 255.0
    offset = int(np.floor(patch_size / 2.0))
    ranges = np.arange(-offset, offset + 1)
    desc = np.take(img, ranges[:,None] * w + ranges + (keypoints[:, 1] * w + keypoints[:, 0])[:, None, None]) # (q, patch_size, patch_size)
    desc = desc.reshape(keypoints.shape[0], -1) # (q, patch_size * patch_size)
    return desc

