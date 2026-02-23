from ultralytics import YOLO
import torch
# Загрузить модель
import cv2
from collections import defaultdict
import numpy as np
# def run():
#     torch.multiprocessing.freeze_support()
#     print('loop')
#
#
# if __name__ == '__main__':
#     run()
#
#     #model = YOLO('args.yaml')  # создать новую модель из YAML
#     # model = YOLO("yolov8x.pt")
#     #model = YOLO("args.yaml").load("last.pt")
#     #model = YOLO("best.pt")
#
#
#
#
#
#     model = YOLO('best.pt')
#     results = model.track(source="2_n.avi", conf=0.5, iou=0.5, show=True, tracker="bytetrack.yaml")


if __name__ == '__main__':
    model = YOLO("yolo11l.pt")



    #results = model.track(source="2_n.avi", conf=0.3, iou=0.5, show=True, persist=True)
      # no arguments needed, dataset and settings remembered


    cap = cv2.VideoCapture("2_n.avi")

    track_history = defaultdict(lambda: [])
    while cap.isOpened():
        ret, frame = cap.read()

        if ret:
            results = model.track(frame, persist=True)

            # Get the boxes and track IDs
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Plot the tracks
            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                track = track_history[track_id]
                track.append((float(x), float(y)))  # x, y center point
                if len(track) > 30:  # retain 90 tracks for 90 frames
                    track.pop(0)

                # Draw the tracking lines
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

            # Display the annotated frame
            cv2.imshow("YOLOv8 Tracking", annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break