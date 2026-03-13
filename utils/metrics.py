import numpy as np

def moving_average(data, window=100):
    if len(data) < window:
        return np.mean(data)
    return np.convolve(data, np.ones(window)/window, mode='valid')