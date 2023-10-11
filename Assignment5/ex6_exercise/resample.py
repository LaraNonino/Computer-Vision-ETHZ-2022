import numpy as np


def resample(particles, particles_w):

    update_indices = np.random.choice(len(particles),
                                      size=len(particles),
                                      replace=True,
                                      p=particles_w.flatten())

    particles_update = particles[update_indices]
    particles_w_update = particles_w[update_indices] / sum(particles_w[update_indices])

    return particles_update, particles_w_update

