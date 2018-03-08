import os
import gdal
import imageio
from time import time
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def clip_and_scale_image(img, img_type='naip', clip_min=0, clip_max=10000):
    """
    Clips and scales bands to between [0, 1] for NAIP, RGB, and Landsat
    satellite images. Clipping applies for Landsat only.
    """
    if img_type in ['naip', 'rgb']:
        return img / 255
    elif img_type == 'landsat':
        return np.clip(img, clip_min, clip_max) / (clip_max - clip_min)

