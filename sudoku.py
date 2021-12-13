import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.feature import canny
from matplotlib import cm
from keras.models import load_model

import matplotlib.pyplot as plt

def get_mask(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    blurred = cv2.GaussianBlur(img_gray.copy(), (9,9),0)
    threshed = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 5)
    
    threshed_not = cv2.bitwise_not(threshed)


    kernel = np.ones((10,10))
    th_not_dial = cv2.dilate(threshed_not, kernel, iterations = 1)

    contours, hierarchy = cv2.findContours(th_not_dial, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    black = np.zeros(img_gray.shape)
    img_sudoku= black.copy()
    sudokus = []
    sud = []
    for i in range(4):
        maxper = 0
        sudoku = -1
        s_n = -1
        for n, c in enumerate(contours):
            per = cv2.arcLength(c, True)
            apr = cv2.approxPolyDP(c, 0.01*per, True)
            if n in sudokus:
                continue
            if len(apr) == 4 and maxper < per:
                s_n = n
                sudoku = apr
                maxper=per
                sud.append(sudoku)
        sudokus.append(s_n)
        img_sudoku= cv2.drawContours(img_sudoku.copy(), [sudoku], -1, 1, -1)
    
    maxper = 0
    sudoku = -1
    s_n = -1
    for n, c in enumerate(contours):
        per = cv2.arcLength(c, True)
        apr = cv2.approxPolyDP(c, 0.01*per, True)
        if len(apr) == 4 and maxper < per:
            s_n = n
            sudoku = apr
            maxper=per
    return img_sudoku, sudoku

def predict_image(image: np.ndarray) -> (np.ndarray, list):
    sudoku_digits = [
        np.int16([[-1, -1, -1, -1, -1, -1, -1, -1, -1],
                  [-1, -1, -1,  8,  9,  4, -1, -1, -1],
                  [-1, -1, -1,  6, -1,  1, -1, -1, -1],
                  [-1,  6,  5,  1, -1,  9,  7,  8, -1],
                  [-1,  1, -1, -1, -1, -1, -1,  3, -1],
                  [-1,  3,  9,  4, -1,  5,  6,  1, -1],
                  [-1, -1, -1,  8, -1,  2, -1, -1, -1],
                  [-1, -1, -1,  9,  1,  3, -1, -1, -1],
                  [-1, -1, -1, -1, -1, -1, -1, -1, -1]]),
    ]
    mask, sudoku = get_mask(image)
    # print(sudoku)
    corners = [(corner[0][0], corner[0][1]) for corner in sudoku]
    ordered = np.asarray(corners).reshape((4,2))
    new = np.zeros((4,1,2), dtype=np.int32)
    add= ordered.sum(1)
    new[0] = ordered[np.argmin(add)]
    new[3] = ordered[np.argmax(add)]
    diff = np.diff(ordered, axis=1)
    width, height = 28*9, 28*9
    new[1] = ordered[np.argmin(diff)]
    new[2] = ordered[np.argmax(diff)]
    # print(new)
    dimensions = np.array([[0, 0], [width, 0], [0, height], [width, height]], dtype="float32")
    ordered_corners = np.float32(new)
    grid = cv2.getPerspectiveTransform(ordered_corners, dimensions)
    # print(grid)
    warped = cv2.warpPerspective(image, grid, (width, height))
    # plt.figure(dpi=150)
    # plt.imshow(image, cmap="gray")
    # plt.show()
    cells = []
    for r in np.vsplit(warped, 9):
        for c in np.hsplit(r, 9):
            cells.append(c)

    model = load_model("mnist.h5")
    digits = []
    for cell in cells:
        
        cif = cell.copy()
        g_cif = np.dot(cif[...,:3], [0.299, 0.587, 0.114])
        img = g_cif.reshape(1,28,28,1)
        img = img/255.0
        res = model.predict([img])[0]
        digits.append(np.argmax(res))
    # print(digits)
    return mask, np.asarray(sudoku_digits, dtype=np.int16)


# img = cv2.imread("train/train_0.jpg")
# mask, ss = predict_image(img)
# cv2.imwrite("test.jpg", mask)