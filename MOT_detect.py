import copy
import time
import torch
import multiprocessing as mp
import numpy as np
from pathlib import Path
from render_utils import BlockRenderer
from boxmot import BoostTrack, BotSort, StrongSort, DeepOcSort, ByteTrack, HybridSort, OcSort
from loggers import *
from shm_utils import SharedFrameReader, SharedFrameWriter


class MOT:
    def __init__(self, q_to_mot: mp.Queue, q_to_stark: mp.Queue, q_to_qwen: mp.Queue,
                 render_cfg=None, class_names=None, method_mot=0, tracker_params=None,
                 shm_yolo_to_mot_name_0=None, shm_yolo_to_mot_name_1=None,
                 shm_yolo_to_mot_free_queue=None, shm_yolo_to_mot_ready_queue=None,
                 shm_mot_to_vlm_name_0=None, shm_mot_to_vlm_name_1=None,
                 shm_mot_to_vlm_free_queue=None, shm_mot_to_vlm_ready_queue=None,
                 shm_mot_to_sot_name_0=None, shm_mot_to_sot_name_1=None,
                 shm_mot_to_sot_free_queue=None, shm_mot_to_sot_ready_queue=None,
                 frame_h=1080, frame_w=1920,
                 vlm_enabled=False,
                 vlm_send_interval=30,
                 vlm_max_objects=3):
        self.q_mot = q_to_mot
        self.mot_method = method_mot
        self.q_to_stark = q_to_stark
        self.q_to_qwen = q_to_qwen
        self.render_cfg = render_cfg
        self.renderer = BlockRenderer(render_cfg or {}, class_names)

        self.mot = None
        self.path_reid = 'osnet_x0_25_msmt17.pt'
        self.flag_show_track = False
        self.half = True
        self.device = 0

        # PTZ логика
        self.locked_track_id = None
        self.lock_enabled = False
        self.target_lost_frames = 0
        self.max_lost_frames = 30
        self.last_known_bbox = None

        self.tracker_params = tracker_params or {}
        self.logger = logging.getLogger(__name__)

        # Размеры кадра
        self.frame_h = frame_h
        self.frame_w = frame_w

        # Shared Memory для чтения от YOLO
        self.shm_yolo_to_mot_name_0 = shm_yolo_to_mot_name_0
        self.shm_yolo_to_mot_name_1 = shm_yolo_to_mot_name_1
        self.shm_yolo_to_mot_free_queue = shm_yolo_to_mot_free_queue
        self.shm_yolo_to_mot_ready_queue = shm_yolo_to_mot_ready_queue

        # Shared Memory для отправки в VLM
        self.vlm_enabled = vlm_enabled
        self.vlm_send_interval = vlm_send_interval
        self.vlm_max_objects = vlm_max_objects

        self.shm_mot_to_vlm_name_0 = shm_mot_to_vlm_name_0
        self.shm_mot_to_vlm_name_1 = shm_mot_to_vlm_name_1
        self.shm_mot_to_vlm_free_queue = shm_mot_to_vlm_free_queue
        self.shm_mot_to_vlm_ready_queue = shm_mot_to_vlm_ready_queue

        # Shared Memory для отправки в SOT трекер (STARK/NanoTrack)
        self.shm_mot_to_sot_name_0 = shm_mot_to_sot_name_0
        self.shm_mot_to_sot_name_1 = shm_mot_to_sot_name_1
        self.shm_mot_to_sot_free_queue = shm_mot_to_sot_free_queue
        self.shm_mot_to_sot_ready_queue = shm_mot_to_sot_ready_queue

        self.shm_reader = None
        self.shm_vlm_writer = None
        self.shm_sot_writer = None
        self.vlm_counter = 0
        self.vlm_send_interval = 30
        self.vlm_max_objects = 3

    def set_mot_method(self, val: int):
        self.mot_method = val

    def set_path_weights_reid(self, val: str):
        self.path_reid = val
        self.logger.info(f"Choice_weights_reid:{self.path_reid}")

    def set_flag_half(self, val: bool):
        self.half = val

    def set_device(self, val: int):
        self.device = val

    def set_flag_show_track(self, val: bool):
        self.flag_show_track = val

    def set_max_lost_frames(self, val: int):
        self.max_lost_frames = val

    def choice_mot(self):
        base_kwargs = {
            "reid_weights": Path(self.path_reid),
            "device": torch.device("cuda:0") if self.device == 0 else torch.device("cpu"),
            "half": bool(self.half)
        }
        final_kwargs = {**base_kwargs, **self.tracker_params}
        dict_mot = {
            0: (BoostTrack(**final_kwargs), "Choice method mot BoostTrack!"),
            1: (BotSort(**final_kwargs), "Choice method mot BotSort!"),
            2: (StrongSort(**final_kwargs), "Choice method mot StrongSort!"),
            3: (DeepOcSort(**final_kwargs), "Choice method mot DeepOcSort!"),
            4: (ByteTrack(**final_kwargs), "Choice method mot ByteTrack!"),
            5: (HybridSort(**final_kwargs), "Choice method mot HybridSort!"),
            6: (OcSort(**final_kwargs), "Choice method mot OcSort!")
        }
        try:
            self.mot, st = dict_mot[self.mot_method]
            self.logger.info(f"✅ {st}")
            return self.mot, True
        except TypeError as e:
            self.logger.warning(f"⚠️ Трекер не принимает часть параметров: {e}")
            self.logger.info("🔄 Fallback: инициализация только с базовыми параметрами")
            try:
                self.mot, st = {0: (BoostTrack(**base_kwargs), "BoostTrack"), 1: (BotSort(**base_kwargs), "BotSort"),
                                2: (StrongSort(**base_kwargs), "StrongSort"),
                                3: (DeepOcSort(**base_kwargs), "DeepOcSort"),
                                4: (ByteTrack(**base_kwargs), "ByteTrack"),
                                5: (HybridSort(**base_kwargs), "HybridSort"), 6: (OcSort(**base_kwargs), "OcSort")}[
                    self.mot_method]
                return self.mot, True
            except:
                self.mot = HybridSort(**base_kwargs)
                return self.mot, True
        except Exception as e:
            self.logger.error(f"❌ Errrr_choice_mot: {e.args}")
            self.mot = HybridSort(**base_kwargs)
            return self.mot, True

    def run_process(self, daemon=True):
        prc = mp.Process(target=self.main_worker, args=(), daemon=daemon)
        prc.start()
        return prc

    def main_worker(self):
        if not logging.getLogger().handlers:
            setup_logging()
        tracker, flg = self.choice_mot()
        frame_count = 0
        start_time = time.time()

        while flg:
            if not self.q_mot.empty():
                try:
                    task = self.q_mot.get()

                    # === ЧТЕНИЕ КАДРА ИЗ SHARED MEMORY ===
                    if isinstance(task, dict) and 'predictions' in task:
                        if self.shm_reader is None:
                            self.shm_reader = SharedFrameReader(
                                self.frame_h, self.frame_w,  # ← ИСПРАВЛЕНО
                                self.shm_yolo_to_mot_name_0,
                                self.shm_yolo_to_mot_name_1,
                                self.shm_yolo_to_mot_free_queue,
                                self.shm_yolo_to_mot_ready_queue
                            )
                            self.logger.info(f"✅ MOT: подключён к Shared Memory от YOLO ({self.frame_w}x{self.frame_h})")
                        frame = self.shm_reader.read_frame()
                        predictions = task['predictions']
                    else:
                        frame, predictions = task

                    h, w, _ = frame.shape
                    frame_stark = copy.deepcopy(frame)

                    # === ФОРМИРУЕМ DETS ДЛЯ MOT ===
                    dets = []
                    for k in predictions:
                        x_lft, y_lft = k.left_top()
                        x_rgh, y_rgh = k.right_bottom()
                        conf, obj_cls = k.p, k.obj_class
                        dets.append([x_lft, y_lft, x_rgh, y_rgh, conf, obj_cls])

                    # === ОБНОВЛЕНИЕ MOT ТРЕКОВ ===
                    res = tracker.update(np.array(dets), frame)

                    # === РАБОТАЕМ С MOT ТРЕКАМИ ===
                    mot_tracks = []
                    vlm_qwen = [] if self.vlm_enabled else None

                    for track in res:
                        x1, y1, x2, y2, track_id = track[:5]
                        track_conf = track[5] if len(track) > 5 else 1.0
                        cls = int(track[-2])
                        if cls == 0 or cls == 2:
                            bbox = [x1, y1, x2 - x1, y2 - y1]
                            if vlm_qwen is not None:
                                vlm_qwen.append([x1, y1, x2, y2])
                            x_cnt = x1 + (x2 - x1) / 2
                            y_cnt = y1 + (y2 - y1) / 2
                            dx = x_cnt - (w // 2)
                            dy = y_cnt - (h // 2)
                            euclid_dist = (dx * dx + dy * dy) ** 0.5
                            mot_tracks.append({
                                'track_id': int(track_id),
                                'bbox': bbox,
                                'conf': track_conf,
                                'distance': euclid_dist,
                                'center': (x_cnt, y_cnt),
                                'class': cls
                            })

                    # === VLM ЛОГИКА С SHARED MEMORY ===
                    if self.vlm_enabled:
                        self.vlm_counter += 1
                        if (self.vlm_counter % self.vlm_send_interval == 0 and vlm_qwen
                                and self.shm_mot_to_vlm_name_0 is not None and not self.q_to_qwen.full()):
                            if self.shm_vlm_writer is None:
                                self.shm_vlm_writer = SharedFrameWriter(
                                    h, w,  # ← Используем реальные размеры из кадра
                                    self.shm_mot_to_vlm_name_0,
                                    self.shm_mot_to_vlm_name_1,
                                    self.shm_mot_to_vlm_free_queue,
                                    self.shm_mot_to_vlm_ready_queue
                                )
                            self.shm_vlm_writer.write_frame(frame)
                            self.q_to_qwen.put({
                                'frame_idx': frame_count,
                                'bboxes': vlm_qwen[:self.vlm_max_objects],
                                'frame_h': h,
                                'frame_w': w
                            })

                    # === ВЫБОР ЛУЧШЕЙ ЦЕЛИ ===
                    best_target = None
                    if self.lock_enabled and self.locked_track_id is not None:
                        for track in mot_tracks:
                            if track['track_id'] == self.locked_track_id:
                                best_target = track
                                self.target_lost_frames = 0
                                self.last_known_bbox = track['bbox']
                                break
                        if best_target is None:
                            self.target_lost_frames += 1
                            if self.target_lost_frames > self.max_lost_frames:
                                self.locked_track_id = None
                                self.target_lost_frames = 0
                    else:
                        if mot_tracks:
                            mot_tracks.sort(key=lambda x: x['distance'] * (1.0 - x['conf']))
                            best_target = mot_tracks[0]
                            self.locked_track_id = best_target['track_id']
                            self.target_lost_frames = 0
                            self.lock_enabled = True
                            self.last_known_bbox = best_target['bbox']

                    # === ОТПРАВКА В SOT ТРЕКЕР (STARK/NanoTrack) ЧЕРЕЗ SHARED MEMORY ===
                    if self.q_to_stark.empty():
                        if best_target:
                            data_packet = {
                                'bbox': best_target['bbox'],
                                'conf': best_target['conf'],
                                'track_id': best_target['track_id'],
                                'lost': False
                            }
                        else:
                            data_packet = {
                                'bbox': self.last_known_bbox if self.last_known_bbox else [],
                                'conf': 0.0,
                                'track_id': self.locked_track_id if self.locked_track_id else -1,
                                'lost': True,
                                'lost_frames': self.target_lost_frames
                            }

                        # НОВЫЙ РЕЖИМ: кадр в SHM, data_packet в Queue
                        if self.shm_mot_to_sot_name_0 is not None:
                            if self.shm_sot_writer is None:
                                self.shm_sot_writer = SharedFrameWriter(
                                    h, w,  # ← Используем реальные размеры из кадра
                                    self.shm_mot_to_sot_name_0, self.shm_mot_to_sot_name_1,
                                    self.shm_mot_to_sot_free_queue, self.shm_mot_to_sot_ready_queue
                                )
                                self.logger.info("✅ MOT: SharedFrameWriter для SOT создан")
                            self.shm_sot_writer.write_frame(frame_stark)
                            self.q_to_stark.put(data_packet)
                        else:
                            # Fallback: старый режим через Queue
                            self.q_to_stark.put([frame_stark, data_packet])

                    # === ВИЗУАЛИЗАЦИЯ ===
                    tracker.plot_results(frame, show_trajectories=self.renderer.show_trajectories)

                    frame_count += 1
                    current_time = time.time()
                    fps = 1.0 / (current_time - start_time) if frame_count > 0 else 0.0
                    start_time = current_time
                    self.logger.info(f"Fps_mot:{fps}")
                    self.renderer.show_with_mot_status(
                        frame,
                        locked_id=self.locked_track_id,
                        lost_frames=self.target_lost_frames,
                        max_lost_frames=self.max_lost_frames,
                        fps=fps
                    )

                except Exception as e:
                    self.logger.error(f"Errrrrrrrrrrr_MOT_prc: {e.args}")