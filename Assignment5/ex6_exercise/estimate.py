import numpy as np


def estimate(particles, particles_w):

    mean_state = np.sum(particles*particles_w, 0)/np.sum(particles_w)

    return mean_state
