from ultralytics import YOLO
import torch
# Загрузить модель
import cv2
import multiprocessing as mp



def neural_detect(q_to_neural:mp.Queue):
    model = YOLO(f"/home/usr/PycharmProjects/yolo_proj/ultralytics/runs/detect/train/weights/best.pt")
    print(model.names)
    while True:
        if not q_to_neural.empty():
            frame = q_to_neural.get()
            results = model(frame)
            print(results)




if __name__ == '__main__':


    q_to_neural = mp.Queue(maxsize=1)
    prc = mp.Process(target=neural_detect, args=(q_to_neural,), daemon=False)
    prc.start()

    cap = cv2.VideoCapture("/home/usr/Загрузки/Telegram Desktop/Т-90М с КАЗ Арена-М.mp4")

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if q_to_neural.empty():
            q_to_neural.put(frame)

