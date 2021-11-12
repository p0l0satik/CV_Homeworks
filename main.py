from _template_solution import *

if __name__ == "__main__":
    img = cv2.imread("train/all.jpg")
    print(predict_image(img))