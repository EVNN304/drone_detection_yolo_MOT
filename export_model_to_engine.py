
from ultralytics import YOLO


model = YOLO('/home/usr/PycharmProjects/yolo_proj/ultralytics/final_weights_train/ground_to_air/best_yolo11x_288x288_batch_64.pt')
model.export(format='engine', imgsz=1280, half=True, device=0)
