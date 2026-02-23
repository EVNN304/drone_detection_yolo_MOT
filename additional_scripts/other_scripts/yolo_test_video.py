from ultralytics import YOLO
import torch
# Загрузить модель
from Moution_detect import Moution_detect


from geometry_lib import *
import copy
import multiprocessing as mp


# if __name__ == '__main__':
#
#
#
#     model = YOLO("last.pt")
#
#
#     image = cv2.imread('planes.jpg')
#     results = model(image)
#
#     for result in results:
#         boxes = result.boxes  # Объект Boxes для вывода bbox
#         # masks = result.masks  # Объект Masks для вывода масок сегментации
#         # keypoints = result.keypoints  # Объект Keypoints для вывода поз
#         # probs = result.probs
#         object_type = boxes.cls
#         rect = boxes.data
#         print("FUCK",rect[0])
#         cv2.rectangle(image, (int(rect[0][0]), int(rect[0][1])), (int(rect[0][2]), int(rect[0][3])), (0, 0, 0), 4)
#         cv2.putText(image,f'{int(rect[0][5])}[{rect[0][4]}',(int(rect[0][0]),int(rect[0][1])-10),3,1.3,(0,0,0),2)
#
#
#
#     cv2.imshow("frame", image)
#     cv2.waitKey(10000000)


def track_objects_from_stream(q_in, q_to_neural):
    size_nn_w, size_nn_h = 288, 288
    rescale_w, rescale_h = 600, 600
    p_w, p_h = 0.51, 0.51
    while True:

        neuro_image, contour_boxes, frame_count, img_mout = q_in.get()
        #neuro_image = cv.cvtColor(neuro_image, cv.COLOR_BGR2GRAY)
        print("countor_bboxes", contour_boxes, neuro_image.shape)
        lst_track = contour_boxes
        if (q_to_neural.qsize() == 0):
            print("LST_TACK", lst_track)
            for i, k in enumerate(lst_track):
                Xcnt, Ycnt, W, H = int(k[0] + ((k[2] - k[0])/2)), int(k[1] + ((k[3] - k[1])/2)), k[2] - k[0], k[3] - k[1]
                print("per_w_h", W/size_nn_w, H/size_nn_h, W/size_nn_w < p_w, H/size_nn_h < p_h)
                if (W/size_nn_w < p_w) and (H/size_nn_h < p_h):
                    obl_detection = cover_pt_by_area((Xcnt, Ycnt), area_w_h=[size_nn_w, size_nn_h], limit_box=[0, 0, 1920, 1080])  # Получаем координаты нарезаемой области
                    print("OBJ_DET_288x288", obl_detection)
                    q_to_neural.put((neuro_image[obl_detection[1]:obl_detection[3], obl_detection[0]:obl_detection[2]], [obl_detection[0], obl_detection[1], obl_detection[2], obl_detection[3]], [frame_count, len([[obl_detection[0], obl_detection[1], obl_detection[2], obl_detection[3]]]), 1]))
                    print("shapes_288x288", neuro_image[obl_detection[1]:obl_detection[3], obl_detection[0]:obl_detection[2]].shape)
                else:
                    obl_detection = cover_pt_by_area((Xcnt, Ycnt), area_w_h=[rescale_w, rescale_h], limit_box=[0, 0, 1920, 1080])  # Получаем координаты нарезаемой области
                    print("OBJ_DET_600x600", obl_detection)
                    q_to_neural.put((neuro_image[obl_detection[1]:obl_detection[3], obl_detection[0]:obl_detection[2]], [obl_detection[0], obl_detection[1], obl_detection[2], obl_detection[3]], [frame_count, len([[obl_detection[0], obl_detection[1], obl_detection[2], obl_detection[3]]]), 1]))
                    print("shapes_600x600", neuro_image[obl_detection[1]:obl_detection[3], obl_detection[0]:obl_detection[2]].shape)


def start_tracking(get_tracker, q_to_neuro:mp.Queue):

    pr_tracking = mp.Process(target=track_objects_from_stream,args=(get_tracker, q_to_neuro))
    pr_tracking.start()

def collect_n_cast_neuro(q_dets:mp.Queue):
    model = YOLO("best_11.pt")
    while True:
        arg = q_dets.get()
        images, cords, idx = arg[0], arg[1], arg[2]
        results = model(images, conf=0.6)
        print("Result_DETECTION", results[0].boxes.xyxy.cpu().tolist(), results[0].boxes.cls.cpu().tolist(), results[0].boxes.conf.cpu().tolist())
        for result in results:
            # boxes = result.boxes
            # rect = boxes.data
            # print("FUCK", rect.items())

            annotated_frame = result.plot()

            #cv.rectangle(images, (int(rect[0][0]), int(rect[0][1])), (int(rect[0][2]), int(rect[0][3])), (0, 0, 0), 4)
        #     cv.putText(images,f'{int(rect[0][5])}[{rect[0][4]}',(int(rect[0][0]),int(rect[0][1])-10),3,1.3,(0,0,0),2)

        cv.imshow("frame", annotated_frame)
        cv.waitKey(1)



if __name__ == '__main__':


    # Очередь обработчика нейросети. Размер должен быть немного больше максимального количества областей
    q_to_neuro = mp.Queue(50)
    # Выходная очередь нейросети, из которой забираются все обнаружения
    q_from_neuro = mp.Queue(8)

    q_to_tracker = mp.Queue(1)
    get_tracker = mp.Queue(1)
    q_from_tracker = None

    #Инициализация и запуск обработчиков нейросети:

    neuro_data_collector = mp.Process(target=collect_n_cast_neuro, args=(q_to_neuro,))
    neuro_data_collector.start()



    cam = cv.VideoCapture("2_n.avi") # "2_n.avi"


    Moution_detect(q_to_tracker, get_tracker)
    start_tracking(get_tracker, q_to_neuro)

    frame_count = 0

    while cam.isOpened():
        frame_valid, image = cam.read()
        print(image.shape)
        if frame_valid:
            frame_count += 1
            if q_to_tracker.empty():
                q_to_tracker.put((copy.deepcopy(image), frame_count))

            cv.imshow('original', cv.resize(image, (600, 600)))
            cv.waitKey(1)
