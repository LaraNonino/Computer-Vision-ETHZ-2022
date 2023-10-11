import numpy as np


def color_histogram(xmin, ymin, xmax, ymax, frame, hist_bin):

    hist = np.zeros((3, hist_bin))  # hist.shape = [3, hist_bin]

    x_min = max(0, round(xmin))
    y_min = max(0, round(ymin))
    x_max = min(frame.shape[1] - 1, round(xmax))
    y_max = min(frame.shape[0] - 1, round(ymax))

    color_0 = frame[y_min:y_max, x_min:x_max, 0]
    color_1 = frame[y_min:y_max, x_min:x_max, 1]
    color_2 = frame[y_min:y_max, x_min:x_max, 2]

    hist[0] = np.histogram(color_0, bins=hist_bin)[0]
    hist[1] = np.histogram(color_1, bins=hist_bin)[0]
    hist[2] = np.histogram(color_2, bins=hist_bin)[0]

    hist = hist/np.sum(hist)    # normalized histogram
    hist = hist.flatten()

    return hist




