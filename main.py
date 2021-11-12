from hello_world import *

if __name__ == "__main__":
    img = cv2.imread('train/pepega.png')
    train(img, "model.npy")
    new_img = predict(img, "model.npy")
    cv2.imwrite('train/new_pepega.png', new_img)