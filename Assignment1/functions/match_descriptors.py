from cv2 import threshold
import numpy as np
from scipy.spatial.distance import cdist

def ssd(desc1, desc2):
    '''
    Sum of squared differences
    Inputs:
    - desc1:        - (q1, feature_dim) descriptor for the first image
    - desc2:        - (q2, feature_dim) descriptor for the first image
    Returns:
    - distances:    - (q1, q2) numpy array storing the squared distance
    '''
    assert desc1.shape[1] == desc2.shape[1]
    
    #cidst computes the distance between each pair of the two collections \
    # of inputs. To get the SSD, it need to be squared.
    distances = cdist(desc1, desc2) ** 2
    return distances

def match_descriptors(desc1, desc2, method = "one_way", ratio_thresh=0.5):
    '''
    Match descriptors
    Inputs:
    - desc1:        - (q1, feature_dim) descriptor for the first image
    - desc2:        - (q2, feature_dim) descriptor for the first image
    Returns:
    - matches:      - (m x 2) numpy array storing the indices of the matches
    '''
    assert desc1.shape[1] == desc2.shape[1]
    distances = ssd(desc1, desc2)
    q1, q2 = desc1.shape[0], desc2.shape[0]
    matches = None
    if method == "one_way": # Query the nearest neighbor for each keypoint in image 1

        #1D array - the ith element of the array contains the minimum gradient \
        # distance between the ith keypoint of the first image and the closest one \
        # in the second image
        min = np.amin(distances, axis=1)

        #xind contains the index of all the first image corners ([0, 1, 2, ..., n])
        #yind is an array where the ith element is the index of the keypoint of the \
        # second image whose gradient is closer to the ith keypoint of the first image
        xind, yind = np.where(distances == min[:, None]) 
        matches = np.column_stack((xind, yind))

    elif method == "mutual":

        #same implementation as ow, but taking the intersection of the matches at the end

        min_ow = np.amin(distances, axis=1)
        xind, yind = np.where(distances == min_ow[:, None])
        matches_ow= np.column_stack((xind, yind))

        min_mut = np.amin(distances, axis=0)
        xind, yind = np.where(distances == min_mut[None, :])
        matches_mut = np.column_stack((xind, yind))

        matches = np.unique(matches_ow[np.where(cdist(matches_ow, matches_mut) == 0)[0]], axis=0)

    elif method == "ratio":

        #same 2D distances array sorted such as the first two elements of each \
        # row are the smallest ones (the first one is the minimum).
        ordered_distances = np.partition(distances, 1, axis = 1) 
        
        min = ordered_distances[:,0]

        #ratio between the first and the second smallest values of each \
        # row in distances which are the first and the second nearest neighbor. \
        ratios = ordered_distances[:, 0] / ordered_distances[:, 1] 
        
        xind, yind = np.where(distances == min[:, None])

        matches = np.column_stack((xind, yind))
        matches = matches[ratios < ratio_thresh]

    else:
        raise NotImplementedError
    return matches

