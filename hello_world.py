import numpy as np
import cv2

def predict(img: np.ndarray, model_path: str) -> np.ndarray:
    model = np.load(model_path)
    img -= model[0]
    img = np.round(img / model[1])
    return img

def train(img: np.ndarray, save_model_path: str) -> None:
    model = np.array([img.mean(), img.std()], np.uint8)
    np.save(save_model_path, model)

