from ultralytics import YOLO
import torch
import cv2 as cv
from collections import defaultdict
import numpy as np



if __name__ == '__main__':


    model = YOLO("best.pt")


    #results = model.track("2_n.avi", conf=0.5, iou=0.5, show=True)



    # Load the YOLO11 model

    # Open the video file
    cap = cv.VideoCapture("/home/usr/Видео/anti_uav_video/Anti-UAV-RGBT/train/20190925_194211_1_7/infrared.mp4")

    # Store the track history
    track_history = defaultdict(lambda: [])

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLO11 tracking on the frame, persisting tracks between frames  imgsz=(512, 640)
            result = model.track(frame, conf=0.5, persist=True, imgsz=(640, 640), tracker="bytetrack.yaml")[0]

            # Get the boxes and track IDs
            if result.boxes and result.boxes.is_track:
                boxes = result.boxes.xywh.cpu()
                track_ids = result.boxes.id.int().cpu().tolist()

                # Visualize the result on the frame
                frame = result.plot()

                # Plot the tracks
                for box, track_id in zip(boxes, track_ids):
                    x, y, w, h = box
                    track = track_history[track_id]
                    track.append((float(x), float(y)))  # x, y center point
                    if len(track) > 30:  # retain 30 tracks for 30 frames
                        track.pop(0)

                    # Draw the tracking lines
                    points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                    cv.polylines(frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

            # Display the annotated frame
            cv.imshow("YOLO11 Tracking", frame)

            # Break the loop if 'q' is pressed
            if cv.waitKey(30) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    cap.release()
    cv.destroyAllWindows()