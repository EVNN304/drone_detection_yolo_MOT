import os

from ultralytics import YOLO
import torch
# Загрузить модель
import cv2
import multiprocessing as mp
import time


def neural_detect(q_to_neural:mp.Queue):
    #model = YOLO(f"/home/usr/PycharmProjects/yolo_proj/ultralytics/runs/detect/train/weights/best.pt")
    #model = YOLO(f"bb.pt")
    model = YOLO(f"/home/usr/Загрузки/best.pt")

    dct_name = model.names
    print(dct_name)
    while True:
        if not q_to_neural.empty():
            frame = q_to_neural.get()
            results = model(frame)

            for bbox, clss, cnf in zip(results[0].boxes.xyxy.cpu().tolist(), results[0].boxes.cls.cpu().tolist(), results[0].boxes.conf.cpu().tolist()):

                print("res", bbox, clss, cnf)
                #annotated_frame = result.plot()

                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 4)
                cv2.putText(frame,f'{dct_name[int(clss)]}[{round(cnf, 2)}]',(int(bbox[0]),int(bbox[1])-10),3,1.3,(0,0,255),2)
            #
            cv2.imshow("frame", cv2.resize(frame, (1000, 1000)))
            cv2.waitKey(1)

if __name__ == '__main__':


    q_to_neural = mp.Queue(maxsize=1)
    prc = mp.Process(target=neural_detect, args=(q_to_neural,), daemon=False)
    prc.start()
    flag = True

    if flag:
        cap = cv2.VideoCapture("/home/usr/Загрузки/Telegram Desktop/IMG_5198.MOV") #

        # Loop through the video frames
        while cap.isOpened():
            # Read a frame from the video
            success, frame = cap.read()
            print("frame_shape", frame.shape)
            if q_to_neural.empty() and success:
                q_to_neural.put(frame)

            time.sleep(0.15)
    else:
        pth = f"/home/usr/Загрузки/dataset_ground_vehicle/filtering_dataset/VISDRONE/VisDrone2019-DET-train/images/"
        lst = os.listdir(pth)
        for i, k in enumerate(lst):
            images = cv2.imread(pth+k)
            print("frame_shape", images.shape)
            if q_to_neural.empty():
                q_to_neural.put(images)

            time.sleep(0.45)
