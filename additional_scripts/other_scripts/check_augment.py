import os
import cv2 as cv
import numpy


path = f"/home/usr/Видео/Записи экрана/img/"

lst = os.listdir(path)

for i, k in enumerate(lst):
    print(k)

    if k[-3::] != "txt":
        images = cv.imread(path+k)
        H, W, _ = images.shape
        print(H, W)
    lst_det = []
    with open(path+k[:-3]+"txt", "r") as file:
        for j in file:
            ls = j.split()
            lst_det.append([int(ls[0]), float(ls[1]), float(ls[2]), float(ls[3]), float(ls[4])])
    print(lst_det)
    for j, m in enumerate(lst_det):
        x_cnt, y_cnt, w, h = m[1]*W, m[2]*H, m[3]*W, m[4]*H
        x_lft, y_lft, x_rig, y_rig = int(x_cnt - (w/2)), int(y_cnt - (h/2)), int(x_cnt + (w/2)), int(y_cnt + (h/2))

        cv.rectangle(images, (x_lft, y_lft),
                      (x_rig, y_rig), (255, 0, 0), 6)
    cv.imshow("frame", images)
    cv.waitKey(0)