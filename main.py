from ticket_to_ride import *

if __name__ == "__main__":
    img = cv2.imread("train/all.jpg")
    print('     {"blue": 36, "green": 33, "black": 28, "yellow": 26, "red": 32}')
    print(predict_image(img))
    print()


    img = cv2.imread("train/black_red_yellow.jpg")
    print('     {"blue": 0, "green": 0, "black": 37, "yellow": 24, "red": 30}')
    print(predict_image(img))
    print()

    img = cv2.imread("train/black_blue_green.jpg")
    print('     {"blue": 40, "green": 37, "black": 37, "yellow": 0, "red": 0}')
    print(predict_image(img))
    print()


    img = cv2.imread("train/red_green_blue.jpg")
    print('     {"blue": 38, "green": 38, "black": 0, "yellow": 0, "red": 40}')
    print(predict_image(img))
    print()


    img = cv2.imread("train/red_green_blue_inaccurate.jpg")
    print('     {"blue": 38, "green": 38, "black": 0, "yellow": 0, "red": 40}')


    print(predict_image(img))
    print()
