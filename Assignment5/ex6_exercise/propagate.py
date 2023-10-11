import numpy as np


def propagate(particles, frame_height, frame_width, params):

    if params["model"] == 0:    # no motion
        A = np.array([[1, 0], [0, 1]])
        noise = np.array([params["sigma_position"], params["sigma_position"]])

    if params["model"] == 1:    # constant velocity motion
        A = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]])
        noise = np.array([params["sigma_position"], params["sigma_position"],
                          params["sigma_velocity"], params["sigma_velocity"]])

    deterministic = np.matmul(A, particles.T).T
    stochastic = noise * np.random.randn(particles.shape[0], particles.shape[1])

    particles_update = deterministic + stochastic
    particles_update[:, 0] = np.clip(particles_update[:, 0], 0, frame_width-1)
    particles_update[:, 1] = np.clip(particles_update[:, 1], 0, frame_height - 1)

    return particles_update
