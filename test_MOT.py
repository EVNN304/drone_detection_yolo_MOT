# import cv2
# import torch
# import numpy as np
# from pathlib import Path
# from boxmot import BoostTrack
# from torchvision.models.detection import (fasterrcnn_resnet50_fpn_v2,FasterRCNN_ResNet50_FPN_V2_Weights as Weights)
#
# # Set device
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
# # Load detector with pretrained weights and preprocessing transforms
# weights = Weights.DEFAULT
# detector = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.5)
# detector.to(device).eval()
# transform = weights.transforms()
#
# # Initialize tracker
# tracker = BoostTrack(reid_weights=Path('/home/usr/PycharmProjects/yolo_proj/ultralytics/pre_train_weights_yolo/yolo12x.pt'), device=device, half=True)
#
# # Start video capture
# cap = cv2.VideoCapture(f"/media/usr/sdcard/DCIM/100MEDIA/DJI_0572.MP4")
#
# with torch.inference_mode():
#     while True:
#         success, frame = cap.read()
#         if not success:
#             break
#
#         # Convert frame to RGB and prepare for detector
#         rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         tensor = torch.from_numpy(rgb).permute(2, 0, 1).to(torch.uint8)
#         input_tensor = transform(tensor).to(device)
#
#         # Run detection
#         output = detector([input_tensor])[0]
#         scores = output['scores'].cpu().numpy()
#         keep = scores >= 0.5
#
#         # Prepare detections for tracking
#         boxes = output['boxes'][keep].cpu().numpy()
#         labels = output['labels'][keep].cpu().numpy()
#         filtered_scores = scores[keep]
#         detections = np.concatenate([boxes, filtered_scores[:, None], labels[:, None]], axis=1)
#
#         # Update tracker and draw results
#         #   INPUT:  M X (x, y, x, y, conf, cls)
#         #   OUTPUT: M X (x, y, x, y, id, conf, cls, ind)
#         res = tracker.update(detections, frame)
#         tracker.plot_results(frame, show_trajectories=True)
#
#         # Show output
#         cv2.imshow('BoXMOT + Torchvision', frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#
# # Clean up
# cap.release()
# cv2.destroyAllWindows()\


import cv2
import torch
import time
import numpy as np
from pathlib import Path
from boxmot import BoostTrack, BotSort, StrongSort, DeepOcSort, ByteTrack, HybridSort, OcSort
from ultralytics import YOLO  # Добавляем импорт YOLO

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Initialize YOLOv11x detector with your custom weights
yolo_model = YOLO('/home/usr/PycharmProjects/yolo_proj/ultralytics/final_weights_train/ground_to_air/best_yolo11x_288x288_batch_64.pt')  # <-- УКАЖИТЕ ПУТЬ К ВАШИМ ВЕСАМ osnet_x0_25_msmt17.pt  /home/usr/PycharmProjects/yolo_proj/ultralytics/final_weights_train/ground_to_air/best_yolo11x_288x288_batch_64.pt  /home/usr/PycharmProjects/yolo_proj/ultralytics/runs/detect/train20/weights/best.pt

# Initialize tracker
# tracker = BoostTrack(
#     reid_weights=Path('osnet_x0_25_msmt17.pt'),
#     device=device,
#     half=True
# )
tracker = HybridSort( reid_weights=Path('osnet_x0_25_msmt17.pt'),
    device=0,
    half=True
)#osnet_x0_25_msmt17.pt    ##### HybridSort ведет себя стабильнее относительно сохранения id трека
# Start video capture

# tracker = HybridSort(reid_weights=None,
#     device=0,
#     half=True,
#     with_reid=False
# )

  # Без reid_weights




cap = cv2.VideoCapture(f"/home/usr/Видео/stock-footage-drone-formation-flying-in-the-air-for-detection.webm")
start_time = time.time()
frame_count = 0
while True:
    success, frame = cap.read()
    if not success:
        break

    # Run YOLO detection
    results = yolo_model(
        frame,
        imgsz=1280,  # Размер изображения для инференса
        conf=0.5,  # Порог уверенности
        iou=0.7,  # Порог NMS IoU
        device=device,  # Устройство (cuda/cpu)
        verbose=False  # Отключаем подробный вывод
    )[0]

    # Convert YOLO results to tracking format
    # Формат YOLO: xyxy, conf, cls
    boxes = results.boxes.xyxy.cpu().numpy()
    confs = results.boxes.conf.cpu().numpy()
    classes = results.boxes.cls.cpu().numpy()

    # Prepare detections for tracking: M X (x, y, x, y, conf, cls)
    detections = np.concatenate([
        boxes,
        confs[:, None],
        classes[:, None]
    ], axis=1)
    print(detections)
    # Update tracker and draw results
    #   INPUT:  M X (x, y, x, y, conf, cls)
    #   OUTPUT: M X (x, y, x, y, id, conf, cls, ind)
    res = tracker.update(detections, frame)
    tracker.plot_results(frame, show_trajectories=True)
    frame_count += 1
    current_time = time.time()
    fps = frame_count / (current_time - start_time)
    print("FPS", fps)
    cv2.putText(frame, f'FPS: {fps:.1f}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    # Show output
    cv2.imshow('BoXMOT + YOLOv11x', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()