import cv2
import torch
import numpy as np
from pathlib import Path
from boxmot import HybridSort
from ultralytics import YOLO
import time
import threading
import queue
from collections import deque

# Исправление для FutureWarning: используем правильный autocast
if torch.__version__ >= '2.0.0':
    from torch.amp import autocast
else:
    from torch.cuda.amp import autocast

# Включите CUDA оптимизации
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True


class OptimizedVideoReader:
    """Оптимизированный видеоридер с предзагрузкой кадров"""

    def __init__(self, video_path, buffer_size=10):
        self.video_path = video_path
        self.frame_buffer = deque(maxlen=buffer_size)
        self.running = True
        self.total_frames = 0
        self.cap = None
        self.single_threaded = True  # Режим для избежания ошибки async_lock

    def start(self):
        """Запуск чтения видео в отдельном потоке"""
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {self.video_path}")

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        print(f"Video info: {self.total_frames} frames, {self.fps:.1f} FPS")

        return self

    # !!! ИЗМЕНЕНИЕ 1: Полностью убираем многопоточность - используем только однопоточное чтение !!!
    def read_frame(self):
        """Чтение кадра в однопоточном режиме"""
        if not self.cap.isOpened():
            return None, None

        ret, frame = self.cap.read()
        if not ret:
            self.running = False
            return None, None

        # !!! ИЗМЕНЕНИЕ 2: Корректный расчет индекса кадра !!!
        frame_idx = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        return frame_idx, frame

    def release(self):
        self.running = False
        if self.cap is not None:
            self.cap.release()


def optimized_frame_processing(frame, yolo_model, tracker, device, imgsz=800):
    """Оптимизированная обработка кадра"""
    with torch.no_grad():
        with autocast(device_type='cuda' if device.type == 'cuda' else 'cpu', enabled=(device.type == 'cuda')):
            # YOLO детекция
            results = yolo_model(
                frame,
                imgsz=imgsz,
                conf=0.5,
                iou=0.6,
                device=device,
                half=True,
                verbose=False,
                agnostic_nms=True
            )[0]

    # Эффективная подготовка данных
    if len(results.boxes) > 0:
        boxes = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy()

        mask = confs > 0.5
        if np.any(mask):
            boxes = boxes[mask]
            confs = confs[mask]
            classes = classes[mask]
            detections = np.column_stack((boxes, confs, classes))
        else:
            detections = np.empty((0, 6))
    else:
        detections = np.empty((0, 6))

    tracker.update(detections, frame)
    return results


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")

    print("Loading YOLO model...")
    yolo_model = YOLO('/home/usr/PycharmProjects/yolo_proj/ultralytics/final_weights_train/ground_to_air/best_yolo11x_288x288_batch_64.pt')

    print("Loading tracker...")
    tracker = HybridSort(
        reid_weights=Path('osnet_x0_25_msmt17.pt'),
        device=0,
        half=True,
    )

    print("Warming up models...")
    torch.cuda.empty_cache()
    print("Warm-up completed!")

    print("Starting video capture...")
    video_reader = OptimizedVideoReader("/home/usr/PycharmProjects/yolo_proj/ultralytics/videos/ground_to_air/2_n.avi").start()

    cv2.namedWindow('High-Speed Tracking', cv2.WINDOW_NORMAL)

    start_time = time.time()
    frame_count = 0
    fps_values = []
    last_frame_idx = -1
    imgsz = 1280

    try:
        while video_reader.running:
            # !!! ИЗМЕНЕНИЕ 3: УДАЛЯЕМ СТРОКУ, КОТОРАЯ ПЕРЕЗАПИСЫВАЛА frame_data !!!
            # Эта строка была причиной зависания и ошибок:
            # frame_data = video_reader.get_frame()

            frame_idx, frame = video_reader.read_frame()
            if frame is None:
                break

            # Пропуск дубликатов кадров
            if frame_idx <= last_frame_idx:
                continue
            last_frame_idx = frame_idx

            num_active_tracks = len(tracker.active_tracks)

            results = optimized_frame_processing(frame, yolo_model, tracker, device, imgsz)

            show_trajectories = (frame_count % 2 == 0)
            tracker.plot_results(frame, show_trajectories=show_trajectories)

            frame_count += 1
            current_time = time.time()
            elapsed = current_time - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            fps_values.append(fps)

            display_fps = np.mean(fps_values[-5:]) if len(fps_values) >= 5 else fps
            cv2.putText(frame, f'FPS: {display_fps:.1f} (imgsz={imgsz})',
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f'Tracks: {num_active_tracks}',
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f'Frame: {frame_idx}/{video_reader.total_frames}',
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow('High-Speed Tracking', frame)

            # !!! ИЗМЕНЕНИЕ 4: Раскомментируем выход по клавишам !!!
            key = cv2.waitKey(1)
            if key == ord('q') or key == 27:  # 27 = Esc
                print("User interrupted processing")
                break

            time.sleep(0.001)

            if frame_count % 50 == 0:
                torch.cuda.empty_cache()

    except KeyboardInterrupt:
        print("Processing interrupted by user")
    finally:
        video_reader.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)

        total_time = time.time() - start_time
        avg_fps = frame_count / total_time if total_time > 0 else 0

        print("\n=== Performance Summary ===")
        print(f"Total frames processed: {frame_count}")
        print(f"Total processing time: {total_time:.2f} seconds")
        print(f"Average FPS: {avg_fps:.1f}")
        if fps_values:
            print(f"Peak FPS: {max(fps_values):.1f}")
            print(f"Min FPS: {min(fps_values):.1f}")

        print(f"\nFinal track count: {len(tracker.active_tracks)}")
        print(f"Total unique tracks: {tracker.track_id_count}")

        print("\nProcessing completed successfully!")