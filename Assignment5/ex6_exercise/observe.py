import numpy as np
from color_histogram import color_histogram
from chi2_cost import chi2_cost


def observe(particles, frame, bbox_height, bbox_width, hist_bin, hist, sigma_observe):

    xs_min = particles[:, 0] - bbox_width / 2
    xs_max = particles[:, 0] + bbox_width / 2
    ys_min = particles[:, 1] - bbox_height / 2
    ys_max = particles[:, 1] + bbox_height / 2

    particles_w = np.zeros((len(particles), 1))

    for i in range(len(particles)):
        hist_i = color_histogram(xs_min[i], ys_min[i], xs_max[i], ys_max[i], frame, hist_bin)
        chi_i = chi2_cost(hist_i, hist)
        particles_w[i] = (1 / (np.sqrt(2 * np.pi) * sigma_observe)) * \
                         np.exp(- (chi_i ** 2) / ((sigma_observe ** 2) * 2))

    particles_w = particles_w/sum(particles_w)

    return particles_w
