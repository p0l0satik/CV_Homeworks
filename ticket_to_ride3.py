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




def get_local_centers(corr, th):
    lbl, n = label(corr >= th, connectivity=2, return_num=True)
    return np.int16([np.round(np.mean(np.argwhere(lbl == i), axis=0)) for i in range(1, n + 1)])


def plot_rectangles(img, points, bbox_shape, border = 3, color = (255, 255, 255)):
    points = np.int16(points)[::, ::-1]
    res_img = np.int16(img.copy())
    for pt in points:
        cv2.rectangle(res_img, (pt[0] - bbox_shape[0] // 2, pt[1] - bbox_shape[1] // 2),
                      (pt[0] + bbox_shape[0] // 2, pt[1] + bbox_shape[1] // 2), color, border)
    return res_img

def plot_img(img, cmap='gray'):
    plt.figure(figsize=(12,6))
    plt.imshow(img, cmap=cmap)
    plt.axis('off')
    plt.show()

def crop(img):
    x, y = 3338, 1709
    h, w = 60, 60
    return img[y-h:y, x-w:x]

def get_city_centers(img, all_img):
    template = all_img
    template = crop(template)

    return apply_template(img, template, True)

def apply_template(img, template, invert=False):
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    temp_gray = cv2.cvtColor(template, cv2.COLOR_RGB2GRAY)

    if invert:
        img_gray = cv2.bitwise_not(img_gray)
        temp_gray = cv2.bitwise_not(temp_gray)

    corr_skimage = match_template(img_gray, temp_gray, pad_input=True)
    return np.int64(get_local_centers(corr_skimage, 0.5))

def apply_morfology(mask):
    mask_int = mask.astype(np.uint8)
    kernel = np.ones((4, 4),np.uint8)
    mask_int_eroded = cv2.erode(mask_int, kernel,iterations = 1)

    kernel = np.ones((10,10))
    mask_red_opened = cv2.morphologyEx(mask_int_eroded, cv2.MORPH_OPEN, kernel)

    mask_int_dilated = cv2.dilate(mask_red_opened, kernel,iterations = 1)

    kernel = np.ones((15,15))
    mask_red_opened2 = cv2.morphologyEx(mask_int_dilated, cv2.MORPH_OPEN, kernel)
    mask_int_dilated2 = cv2.dilate(mask_red_opened2, kernel,iterations = 1)
    return mask_int_dilated2

def apply_black_morfology(mask):
    mask_int = mask.astype(np.uint8)
    kernel = np.ones((2, 2),np.uint8)
    mask_int_eroded = cv2.erode(mask_int, kernel,iterations = 1)
    
    kernel = np.ones((6,6))
    mask_red_opened = cv2.morphologyEx(mask_int_eroded, cv2.MORPH_OPEN, kernel)
    kernel = np.ones((11,11))


    mask_int_dilated = cv2.dilate(mask_red_opened, kernel,iterations = 1)

    kernel = np.ones((15,15))
    mask_red_opened2 = cv2.morphologyEx(mask_int_dilated, cv2.MORPH_OPEN, kernel)
    mask_int_dilated2 = cv2.dilate(mask_red_opened2, kernel,iterations = 1)
    return mask_int_dilated2

def calc_trains(contours, aver_len):
    num = 0
    for c in contours:
        num += round(cv2.arcLength(c,True) / aver_len)
    return num

def get_trains_colored(img):
    temp_img = img.copy()
    HLS = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    SAT = HLS[:, :, 2]
    mask =  (SAT > 75) 
    temp_img[mask==0] = (0,0,0)
    return temp_img

def get_wagons(img, mask, template, len = 380, cut = True, black = False):
    #no empty fields
    mask = mask.astype(np.uint8)
    if cut:
        points = apply_template(img, template)
        mask = plot_rectangles(mask, points, (110, 110), -1, (0, 0, 0))

    #morfolgy
    if black:
        mask = apply_black_morfology(mask)
    else:
        mask = apply_morfology(mask)
    #contours
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return calc_trains(contours, 350)

def get_red_wagons(img, HUE, all_img):
    mask = ((HUE > 170) & (HUE < 190) & (HUE != 177))
    template = all_img
    template = template[..., ::-1]
    template = template[1680:1708, 1070:1100]
    return get_wagons(img, mask, template)

def get_blue_wagons(img, HUE, all_img):
    img[0:100, :] = (0,0,0)
    img[2436:, :] = (0,0,0)
    img[:, 0:110] = (0,0,0)
    img[:, 3726:] = (0,0,0)
    mask = ((HUE > 100) & (HUE < 110))
    template = all_img
    template = template[..., ::-1]
    template = template[2250:2280, 205:230]
    return get_wagons(img, mask, template)

def get_black_wagons(img):
    HLS = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    LIGHT = HLS[:, :, 1]
    SAT = HLS[:, :, 2]
    img[0:150, :] = (255, 255, 255)
    img[2436:, :] = (255, 255, 255)
    img[:, 0:110] = (255, 255, 255)
    img[:, 3726:] = (255, 255, 255)
    mask = (LIGHT < 30) & (SAT < 35)
    return get_wagons(img, mask, img, 400, False, True)

def get_yellow_wagons(img, HUE, all_img):
    mask = ((HUE > 23) & (HUE < 30))
    template = all_img
    template = template[..., ::-1]
    template = template[2107:2136 , 574:601]
    return get_wagons(img, mask, template)

def get_green_wagons(img, HUE):
    mask = ((HUE > 70) & (HUE < 85)).astype(np.uint8)
    return get_wagons(img, mask, img, 330, False)



def predict_image(img: np.ndarray) -> (Union[np.ndarray, list], dict, dict):
    img = img[..., ::-1] 
    black_n = get_black_wagons(img)
    all_img = cv2.imread("/autograder/source/train/all.jpg") 
    # all_img = cv2.imread("train/all.jpg") 
    
    city_centers = get_city_centers(img, all_img)
    colored_trains_img = get_trains_colored(img)
    HLS = cv2.cvtColor(colored_trains_img, cv2.COLOR_RGB2HLS)
    HUE = HLS[:, :, 0]  


    red_n = get_red_wagons(img, HUE, all_img)
    blue_n = get_blue_wagons(img, HUE, all_img)
    yellow_n = get_yellow_wagons(img, HUE, all_img)
    green_n = get_green_wagons(img, HUE)
    
    n_trains = {'blue': blue_n, 'green': green_n, 'black': black_n, 'yellow': yellow_n, 'red': red_n}
    scores = {'blue': blue_n, 'green': green_n, 'black': black_n, 'yellow': yellow_n, 'red': red_n}
    return city_centers, n_trains, scores
