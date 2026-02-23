import time

from ultralytics import YOLO
import torch
import numpy as np
import cv2



if __name__ == '__main__':


    model = YOLO("yolo11n-seg.pt")  # load an official model
    #print(model.state_dict())
    #results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
    results = model.predict(source="https://ultralytics.com/images/bus.jpg", show=True)
    # for result in results:
    #     xy = result.masks.xy  # mask in polygon format
    #     xyn = result.masks.xyn  # normalized
    #     masks = result.masks.data  # mask in matrix format (num_objects x H x W)
    #     print(masks)

    # Или для ручной отрисовки

    # for r in results:
    #     img = np.copy(r.orig_img)
    #
    #     # Получаем контуры масок: results.masks.xy — список контуров каждого объекта (float)
    #     for contour in r.masks.xy:
    #         print(contour)
    #         poly = contour.astype(np.int32).reshape((-1, 1, 2))
    #         print(poly)
    #         # Рисуем контур
    #         cv2.polylines(img, [poly], isClosed=True, color=(0, 255, 0), thickness=2)
    #
    #     cv2.imshow("Contours", img)
    #     cv2.waitKey(0)

    seg_head = model.model.model[-1]

    print("Segmentation head:")
    print(seg_head)

    # Параметры слоя сегментации
    # for name, param in seg_head.named_parameters():
    #     print(name, param.shape)