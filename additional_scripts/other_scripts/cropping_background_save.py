import os
import time

import cv2 as cv
import multiprocessing as mp
from geometry_lib import *


def write_lables(path_save, names):

    with open(path_save+names+".txt", "w") as file:
        #file.write("")
        #print(f"Lables_save:{path_save+names}")
        pass

def image_write(images, path_save, name):
    cv.imwrite(path_save+name+".jpg", images)



def cropped_image(images, names_image, path_save, area_list, window=(288, 288), overlay=(0.1, 0.1)):
    try:
        cord_cropp = calc_scan_areas(area_list, window_w_h=window, overlay=overlay)
        #print(cord_cropp)
        for j, m in enumerate(cord_cropp):

                crop_image = images[m[1]:m[3], m[0]:m[2]]
                image_write(crop_image, path_save, names_image+f"_{str(j)}")
                write_lables(path_save, names_image+f"_{str(j)}")
    except Exception as e:
        print(f"ERRR: {e.args}, name: {names_image}, path: {path}")

if __name__ == '__main__':
    path = f"/home/usr/Изображения/background_2/" # path to frame or videos
    path_save = f"/home/usr/Изображения/background/"
    flag = False
    cap = os.listdir(path) if flag == False else cv.VideoCapture(path)

    if flag:
        c = 0
        name_gen = f"cropp_background_"
        while cap.isOpened():
            flg, images = cap.read()
            c += 1
            if flg:
                h, w, _ = images.shape
                print((h, w, _), c)
                cropped_image(images, name_gen+str(c), path_save, [0, 0, w, h])
    else:
        for i, k in enumerate(cap):
            images = cv.imread(path+k)
            h, w, _ = images.shape
            print((h, w, _), i)
            cropped_image(images, k, path_save,[0, 0, w, h])



