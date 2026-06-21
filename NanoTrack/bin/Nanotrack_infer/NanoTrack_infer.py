import importlib
import os
import time
import cv2
import torch
import numpy as np
import multiprocessing as mp
from collections import OrderedDict
from render_utils import BlockRenderer
from loggers import *

from NanoTrack.nanotrack.core.config import cfg
from NanoTrack.nanotrack.models.model_builder import ModelBuilder
from NanoTrack.nanotrack.tracker.tracker_builder import build_tracker
from NanoTrack.nanotrack.utils.model_load import load_pretrain

torch.set_num_threads(1)


class NanoTrackWrapper:
    """Обёртка для NanoTrack в стиле STARK: очередь, валидация, рендер."""

    def __init__(self, q_frame: mp.Queue, name: str, parameter_name: str, dataset_name: str,
                 run_id: int = None, display_name: str = None, result_only=False,
                 render_cfg=None, class_names=None, config_path: str = None, snapshot_path: str = None):

        self.name = name
        self.parameter_name = parameter_name
        self.dataset_name = dataset_name
        self.q_frame = q_frame
        self.renderer = BlockRenderer(render_cfg or {}, class_names)
        self.config_path = config_path or "/home/usr/PycharmProjects/Siam_tracker/SiamTrackers/NanoTrack/models/config/configv3.yaml"
        self.snapshot_path = snapshot_path or "/home/usr/PycharmProjects/Siam_tracker/SiamTrackers/NanoTrack/checkpoint_e56.pth"

        self.logger = logging.getLogger(__name__)

        # === НАСТРОЙКИ ВАЛИДАЦИИ  ===
        self.frame_w, self.frame_h = 1920, 1080
        self.iou_threshold = 0.5
        self.max_consecutive_failures = 7
        self.consecutive_failures = 0
        self.grace_period_frames = 10
        self.frames_since_init = 0
        self.min_track_duration = 15
        self.frames_tracked = 0
        self.mot_confidence_threshold = 0.5
        self.smoothing_alpha = 0.7
        self.smoothed_bbox = None
        self.active_track_id = -1
        self.id_mismatch_count = 0
        self.max_id_mismatches = 3
        self.target_lost_alert = False
        self.mot_lost_frames = 0


    def set_iou_threshold(self, val:float):
        self.iou_threshold = val


    def set_max_consecutive_failures(self, val:int):
        self.max_consecutive_failures = val

    def set_smoothing_alpha(self, val:float):
        self.smoothing_alpha = val

    def set_max_id_mismatches(self, val:int):
        self.max_id_mismatches = val

    def set_grace_period_frames(self, val:int):
        self.grace_period_frames = val

    def set_min_track_duration(self, val:int):
        self.min_track_duration = val

    def set_mot_confidence_threshold(self, val:float):
        self.mot_confidence_threshold = val

    def set_config_path(self, val:str):
        self.config_path = val

    def set_snapshot_path(self, val:str):
        self.snapshot_path = val


    def _init_tracker(self):
        """Инициализация модели NanoTrack."""
        cfg.merge_from_file(self.config_path)
        cfg.CUDA = torch.cuda.is_available() and cfg.CUDA
        device = torch.device('cuda' if cfg.CUDA else 'cpu')

        model = ModelBuilder()
        model = load_pretrain(model, self.snapshot_path)
        if cfg.CUDA:
            model = model.cuda().eval()

        tracker = build_tracker(model)
        return tracker

    def _calculate_iou(self, box1, box2):
        if not box1 or not box2: return 0.0
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        xi1, yi1 = max(x1, x2), max(y1, y2)
        xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
        inter_w, inter_h = max(0, xi2 - xi1), max(0, yi2 - yi1)
        inter_area = inter_w * inter_h
        union_area = w1 * h1 + w2 * h2 - inter_area
        return inter_area / (union_area + 1e-9)

    def _check_bbox_sanity(self, bbox):
        if not bbox or len(bbox) != 4: return False
        x, y, w, h = bbox
        area = w * h
        frame_area = self.frame_w * self.frame_h
        if x < -w * 2 or y < -h * 2 or x > self.frame_w + w * 2 or y > self.frame_h + h * 2: return False
        if area > frame_area * 0.55: return False
        return True

    def _validate_with_mot(self, stark_bbox, mot_data):
        if isinstance(mot_data, dict):
            mot_bbox = mot_data.get('bbox', [])
            mot_conf = mot_data.get('conf', 1.0)
            mot_id = mot_data.get('track_id', -1)
            mot_lost = mot_data.get('lost', False)
            self.mot_lost_frames = mot_data.get('lost_frames', 0)
        else:
            mot_bbox = mot_data if mot_data else []
            mot_conf, mot_id, mot_lost = 1.0, -1, False
            self.mot_lost_frames = 0

        if self.frames_since_init < self.grace_period_frames:
            self.consecutive_failures = 0
            self.id_mismatch_count = 0
            return True
        if not mot_bbox or mot_lost:
            self.consecutive_failures = 0
            self.id_mismatch_count = 0
            return True
        if mot_conf < self.mot_confidence_threshold:
            self.consecutive_failures = 0
            self.id_mismatch_count = 0
            return True

        id_match = True
        if self.active_track_id != -1 and mot_id != -1:
            if self.active_track_id != mot_id:
                id_match = False
                self.id_mismatch_count += 1
                self.logger.info(f"⚠️ NanoTrack: ID mismatch! Active: {self.active_track_id}, MOT: {mot_id}")
            else:
                self.id_mismatch_count = 0
        elif mot_id != -1:
            self.active_track_id = mot_id

        iou = self._calculate_iou(stark_bbox, mot_bbox)

        if id_match and iou >= self.iou_threshold:
            self.consecutive_failures = 0
            return True
        if id_match and iou < self.iou_threshold:
            self.consecutive_failures += 1
            if self.consecutive_failures >= self.max_consecutive_failures:
                self.logger.info(f"⚠️ NanoTrack: Low IoU ({iou:.2f}) for {self.consecutive_failures} frames. Reset.")
                self.target_lost_alert = True
                return False
            return True
        if not id_match and iou >= self.iou_threshold:
            self.logger.info(f"✅ NanoTrack: ID changed but IoU high. Update ID: {self.active_track_id} → {mot_id}")
            self.active_track_id = mot_id
            self.consecutive_failures = 0
            self.id_mismatch_count = 0
            return True
        if not id_match and iou < self.iou_threshold:
            if self.id_mismatch_count >= self.max_id_mismatches:
                self.logger.info(f"❌ NanoTrack: ID mismatch + low IoU ({iou:.2f}). RESET.")
                self.target_lost_alert = True
                return False
            self.consecutive_failures += 1
            return True
        self.consecutive_failures = 0
        return True

    def _smooth_bbox(self, bbox):
        if self.smoothed_bbox is None:
            self.smoothed_bbox = list(bbox)
            return bbox
        smoothed = []
        for i in range(4):
            val = self.smoothing_alpha * bbox[i] + (1 - self.smoothing_alpha) * self.smoothed_bbox[i]
            smoothed.append(val)
            self.smoothed_bbox[i] = val
        return smoothed

    def _extract_bbox_from_data(self, data):
        if isinstance(data, dict):
            return data.get('bbox', [])
        elif isinstance(data, list):
            return data if len(data) == 4 else []
        return []

    def run_process(self, daemon=True):
        return mp.Process(target=self.run_video, args=(), daemon=daemon)

    def run_video(self):
        """Основной цикл трекинга — идентичен STARK по логике."""
        if not logging.getLogger().handlers: setup_logging()

        tracker = self._init_tracker()
        output_boxes = []
        is_tracking = False

        def _build_init_info(box):
            return {'init_bbox': box}

        # === Цикл инициализации ===
        while True:
            if not self.q_frame.empty():
                frame, data = self.q_frame.get()
                bbox = self._extract_bbox_from_data(data)
                if bbox:
                    track_id = data.get('track_id', -1) if isinstance(data, dict) else -1
                    mot_lost = data.get('lost', False) if isinstance(data, dict) else False
                    self.logger.info(f"✅ NanoTrack: Init target Box={bbox}, ID={track_id}")

                    # Инициализация NanoTrack
                    init_rect = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
                    tracker.init(frame, init_rect)

                    is_tracking = True
                    output_boxes.append(bbox)
                    self.active_track_id = track_id
                    self.target_lost_alert = mot_lost
                    self.frames_since_init = 0
                    self.frames_tracked = 0
                    self.consecutive_failures = 0
                    self.id_mismatch_count = 0
                    self.smoothed_bbox = None
                    break

        # === Основной цикл трекинга ===
        start_time = time.time()
        frame_count = 0

        while True:
            if not self.q_frame.empty():
                frame, data = self.q_frame.get()
                if frame is None: break

                # Парсинг MOT-данных
                if isinstance(data, dict):
                    mot_bbox = data.get('bbox', [])
                    mot_conf = data.get('conf', 1.0)
                    mot_lost = data.get('lost', False)
                    mot_lost_frames = data.get('lost_frames', 0)
                    mot_data = data
                else:
                    mot_bbox = data if data else []
                    mot_conf, mot_lost, mot_lost_frames = 1.0, False, 0
                    mot_data = mot_bbox

                frame_disp = frame.copy()

                if is_tracking:
                    # Трекинг NanoTrack
                    outputs = tracker.track(frame)
                    nano_bbox = list(map(int, outputs.get('bbox', []))) if 'bbox' in outputs else None

                    if nano_bbox:
                        self.frames_since_init += 1
                        self.frames_tracked += 1

                        # Проверка адекватности
                        if not self._check_bbox_sanity(nano_bbox):
                            self.logger.info("⚠️ NanoTrack: Invalid bbox. Reset.")
                            is_tracking = False
                            self.smoothed_bbox = None
                            self.active_track_id = -1
                            self.target_lost_alert = True
                            continue

                        # Сверка с MOT
                        if self.frames_tracked >= self.min_track_duration:
                            if not self._validate_with_mot(nano_bbox, mot_data):
                                self.logger.info("⚠️ NanoTrack: MOT validation failed. Reset.")
                                is_tracking = False
                                self.smoothed_bbox = None
                                self.active_track_id = -1
                                continue
                        else:
                            self.consecutive_failures = 0
                            self.id_mismatch_count = 0

                        # Сглаживание
                        smoothed = self._smooth_bbox(nano_bbox)
                        output_boxes.append([int(s) for s in smoothed])
                    else:
                        is_tracking = False
                        self.active_track_id = -1
                        self.target_lost_alert = True
                        continue
                else:
                    # Ожидание новой инициализации от MOT
                    mot_bbox_check = self._extract_bbox_from_data(data)
                    if mot_bbox_check:
                        track_id = data.get('track_id', -1) if isinstance(data, dict) else -1
                        mot_lost = data.get('lost', False) if isinstance(data, dict) else False
                        init_rect = [int(mot_bbox_check[0]), int(mot_bbox_check[1]), int(mot_bbox_check[2]),
                                     int(mot_bbox_check[3])]
                        tracker.init(frame, init_rect)
                        is_tracking = True
                        output_boxes.append(mot_bbox_check)
                        self.active_track_id = track_id
                        self.target_lost_alert = mot_lost
                        self.consecutive_failures = 0
                        self.id_mismatch_count = 0
                        self.smoothed_bbox = None
                        self.logger.info(f"✅ NanoTrack: New target acquired ID={track_id}")

                # === Рендеринг телеметрии ===
                frame_count += 1
                current_time = time.time()
                fps = 1.0 / (current_time - start_time) if frame_count > 0 else 0.0
                start_time = current_time

                telemetry = {
                    'track_id': self.active_track_id,
                    'is_tracking': is_tracking,
                    'consecutive_failures': self.consecutive_failures,
                    'max_consecutive_failures': self.max_consecutive_failures,
                    'mot_lost_frames': mot_lost_frames,
                    'lost_alert': self.target_lost_alert,
                    'mot_lost': mot_lost,
                    'bbox': self._smooth_bbox(nano_bbox) if is_tracking and 'nano_bbox' in locals() else None,
                }
                self.renderer.show_stark_telemetry(frame_disp, telemetry, fps)