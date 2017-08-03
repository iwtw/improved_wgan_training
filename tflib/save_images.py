"""
Image grid saver, based on color_grid_vis from github.com/Newmu
"""

import numpy as np
import scipy.misc
from scipy.misc import imsave

def save_images( X , nh ,  save_path):
    # [0, 1] -> [0,255]
    if isinstance(X.flatten()[0], np.floating):
        X = (255.99*X).astype('uint8')

    n_samples = X.shape[0]/3

    if X.ndim == 2:
        X = np.reshape(X, (X.shape[0], int(np.sqrt(X.shape[1])), int(np.sqrt(X.shape[1]))))

    if X.ndim == 4:
        h, w = X[0].shape[:2]
        img = np.zeros((h*nh, w*3, 3))
    elif X.ndim == 3:
        h, w = X[0].shape[:2]
        img = np.zeros((h*nh, w*3))

    for idx in range(nh*3):
        i = idx / 3
        j = idx % 3 
        img[i*h:i*h+h, j*w:j*w+w] = X[j*n_samples +i]

    imsave(save_path, img)
