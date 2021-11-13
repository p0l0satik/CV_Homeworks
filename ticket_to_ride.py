from typing import Union

from collections import defaultdict
from itertools import combinations

import numpy as np
import cv2
from skimage.transform import rescale
from skimage.measure import label, find_contours
from skimage.filters import gaussian
from scipy.spatial.distance import cdist
import scipy.stats as st

from skimage.feature import match_template
from scipy.ndimage import convolve
from skimage.measure import label
import matplotlib.pyplot as plt
import imageio

COLORS = ('blue', 'green', 'black', 'yellow', 'red')
TRAINS2SCORE = {1: 1, 2: 2, 3: 4, 4: 7, 6: 15, 8: 21}


def predict_image(img: np.ndarray) -> (Union[np.ndarray, list], dict, dict):
    # raise NotImplementedError
    city_centers = city_centers(img)
    n_trains = {'blue': 20, 'green': 30, 'black': 0, 'yellow': 30, 'red': 0}
    scores = {'blue': 60, 'green': 90, 'black': 0, 'yellow': 45, 'red': 0}
    return city_centers, n_trains, scores

def get_local_centers(corr, th):
    lbl, n = label(corr >= th, connectivity=2, return_num=True)
    print(lbl)
    return np.int16([np.round(np.mean(np.argwhere(lbl == i), axis=0)) for i in range(1, n + 1)])


def plot_rectangles(img, points, bbox_shape):
    points = np.int16(points)[::, ::-1]
    res_img = np.int16(img.copy())
    for pt in points:
        cv2.rectangle(res_img, (pt[0] - bbox_shape[0] // 2, pt[1] - bbox_shape[1] // 2),
                      (pt[0] + bbox_shape[0] // 2, pt[1] + bbox_shape[1] // 2), (255, 0, 0), 3)
    return res_img
# def predict():

def plot_img(img, cmap='gray'):
    plt.figure(figsize=(12,6))
    plt.imshow(img, cmap=cmap)
    plt.axis('off')
    plt.show()

def city_centers(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_gray = cv2.bitwise_not(img_gray)
    template = cv2.imread("teplate_c.jpg") 

    temp_gray = cv2.cvtColor(template, cv2.COLOR_RGB2GRAY)
    temp_gray = cv2.bitwise_not(temp_gray)


    corr_skimage = match_template(img_gray, temp_gray, pad_input=True)

    return  np.int64(get_local_centers(corr_skimage, 0.5))

