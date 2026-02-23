import numpy as np
import os
import cv2 as cv
import cv2
import multiprocessing as mp
import pandas as pd


def write_csv():
    columns = ['names', 'x', 'y', 'w', 'h']
    try:
        with open("tmp4.csv", 'x', encoding='utf-8-sig') as f:
            pd.DataFrame(columns=columns).to_csv(f, index=False)
    except Exception as e:
        print(f"File_{'tmp4.csv'}, err_type: {e.args}")


# def image_write(q_save:mp.Queue):
#     while True:
#         if not q_save.empty():
#             images, names_file = q_save.get()
#             print(names_file)
#             cv.imwrite(path_save_image + names_file + ".jpg", images)

#
# def run_prc_save(daemon=True):
#     prc = mp.Process(target=image_write, args=(), daemon=daemon)
#     prc.start()


if __name__ == '__main__':
    kernel = np.array([[0, -1, 0],
                       [-1, 8, -1],
                       [0, -1, 0]])
    path = f"/home/usr/Изображения/tmp3/"
    lst_image = sorted(os.listdir(path))[150:]  #[400:1700, ::] [347:593, ::]
    for i, k in enumerate(lst_image):
        #image_1, image_2 = cv.cvtColor(cv2.filter2D(cv.imread(path+lst_image[i])[::, ::], -1, kernel), cv.COLOR_BGR2GRAY), cv.cvtColor(cv2.filter2D(cv.imread(path+lst_image[i+1])[::, ::], -1, kernel), cv.COLOR_BGR2GRAY)
        #print(image_2.shape, lst_image[i], lst_image[i+1])
        #
        image_1 = cv.cvtColor(cv2.filter2D(cv.imread(path + lst_image[i])[::, ::], -1, kernel),
                                       cv.COLOR_BGR2GRAY)
        #binar_image(image_1, image_2)

        _, thresh = cv2.threshold(image_1, 200, 255, cv2.THRESH_BINARY_INV)

        # Поиск контуров
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Предположим, что самый большой контур — это наш квадрат
        largest_contour = max(contours, key=cv2.contourArea)

        # Получение ограничивающего прямоугольника (x, y, w, h) # t_usec,x,y,w,h
        x, y, w, h = cv2.boundingRect(largest_contour)
        # data_row = {
        #     'names': k[:16],
        #     'x': [x],
        #     'y': [y],
        #     'w': [w],
        #     'h': [h],
        # }
        # df = pd.DataFrame(data_row, columns=['names', 'x', 'y', 'w', 'h'])
        # df.to_csv("tmp4.csv", mode='a', header=False, index=False, encoding='utf-8-sig')
        # Отобразим результат
        cv2.rectangle(image_1, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow('Detected Square', cv.resize(image_1, (800, 800)))
        cv2.waitKey(1)
        # lab = cv2.cvtColor(image_1, cv2.COLOR_BGR2LAB)
        # l, a, b = cv2.split(lab)
        # clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        # l2 = clahe.apply(l)
        # lab = cv2.merge((l2, a, b))
        # img_clahe = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        #
        # cv.imshow("Mask", cv.resize(image_2, (800, 800)))
        # cv.waitKey(100)