import copy

import time
import torch

import multiprocessing as mp
import numpy as np
from pathlib import Path
from render_utils import BlockRenderer

from boxmot import BoostTrack, BotSort, StrongSort, DeepOcSort, ByteTrack, HybridSort, OcSort
from loggers import *


class MOT:
    def __init__(self, q_to_mot: mp.Queue, q_to_stark: mp.Queue, q_to_qwen:mp.Queue, render_cfg=None, class_names=None, method_mot=0, tracker_params=None):
        self.q_mot = q_to_mot
        self.mot_method = method_mot
        self.q_to_stark = q_to_stark
        self.q_to_qwen = q_to_qwen ### new

        self.render_cfg = render_cfg
        self.renderer = BlockRenderer(render_cfg or {}, class_names)

        self.mot = None
        self.path_reid = 'osnet_x0_25_msmt17.pt'
        self.flag_show_track = False
        self.half = True
        self.device = 0
        # === PTZ ЛОГИКА: Блокировка на цель ===
        self.locked_track_id = None  # ID заблокированной цели
        self.lock_enabled = False  # Включить блокировку
        self.target_lost_frames = 0  # Счётчик кадров без цели
        self.max_lost_frames = 30  # После 30 кадров — переключаемся
        self.last_known_bbox = None  # Последняя известная позиция (для гимбала)

        self.tracker_params = tracker_params or {}
        self.logger = logging.getLogger(__name__)



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
        """Установить лимит кадров до переключения на новую цель"""
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
                self.mot, st = {
                    0: (BoostTrack(**base_kwargs), "BoostTrack (fallback)"),
                    1: (BotSort(**base_kwargs), "BotSort (fallback)"),
                    2: (StrongSort(**base_kwargs), "StrongSort (fallback)"),
                    3: (DeepOcSort(**base_kwargs), "DeepOcSort (fallback)"),
                    4: (ByteTrack(**base_kwargs), "ByteTrack (fallback)"),
                    5: (HybridSort(**base_kwargs), "HybridSort (fallback)"),
                    6: (OcSort(**base_kwargs), "OcSort (fallback)")
                }[self.mot_method]
                return self.mot, True
            except:
                self.mot = HybridSort(**base_kwargs)
                return self.mot, True

        except Exception as e:
            self.logger.error(f"❌ Errrr_choice_mot: {e.args}")
            self.mot = HybridSort(**base_kwargs)  # Дефолт на гибрид
            return self.mot, True

    def run_process(self, daemon=True):

        prc = mp.Process(target=self.main_worker, args=(), daemon=daemon)
        prc.start()
        return prc


    def main_worker(self):
        if not logging.getLogger().handlers: setup_logging()  # ← только для spawn
        tracker, flg = self.choice_mot()
        frame_count = 0
        start_time = time.time()
        while flg:
            if not self.q_mot.empty():
                try:
                    frame, detections = self.q_mot.get()
                    h, w, _ = frame.shape
                    frame_stark = copy.deepcopy(frame)
                    # === ФОРМИРУЕМ DETS ДЛЯ MOT ===
                    dets = []
                    for k in detections:
                        x_lft, y_lft = k.left_top()
                        x_rgh, y_rgh = k.right_bottom()
                        conf, obj_cls = k.p, k.obj_class

                        # Фильтруем только опасные классы (0 и 2)
                        #if obj_cls == 0 or obj_cls == 2:
                        dets.append([x_lft, y_lft, x_rgh, y_rgh, conf, obj_cls])

                    # === ОБНОВЛЕНИЕ MOT ТРЕКОВ ===
                    res = tracker.update(np.array(dets), frame)

                    # === РАБОТАЕМ НАПРЯМУЮ С MOT ТРЕКАМИ (не с детекциями!) ===
                    mot_tracks = []

                    #vlm_qwen = [] ### new
                    for track in res:
                        x1, y1, x2, y2, track_id = track[:5]
                        # BoxMOT может возвращать confidence в 6-м элементе
                        track_conf = track[5] if len(track) > 5 else 1.0
                        cls = int(track[-2])
                        if cls == 0 or cls == 2:
                            # Конвертируем в [x, y, w, h]
                            bbox = [x1, y1, x2 - x1, y2 - y1]
                            #vlm_qwen.append([x1, y1, x2, y2])
                            # Считаем расстояние до центра кадра
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

                    best_target = None
                    self.logger.info(f"MOT_TRACKS: {mot_tracks}")
                    if self.lock_enabled and self.locked_track_id is not None:
                        # РЕЖИМ БЛОКИРОВКИ: ищем ТОЛЬКО наш track_id
                        for track in mot_tracks:
                            if track['track_id'] == self.locked_track_id:
                                best_target = track
                                self.target_lost_frames = 0
                                self.last_known_bbox = track['bbox']
                                break

                        # Если не нашли — цель потеряна
                        if best_target is None:
                            self.target_lost_frames += 1
                            self.logger.info(f"⚠️ TARGET LOST! ID={self.locked_track_id}, frames={self.target_lost_frames}/{self.max_lost_frames}")

                            # Если потеряли больше max_lost_frames — сбрасываем блокировку
                            if self.target_lost_frames > self.max_lost_frames:
                                self.logger.info(f"🔓 UNLOCKED: Цель не найдена {self.target_lost_frames} кадров, переключаемся на новую")
                                self.locked_track_id = None
                                self.target_lost_frames = 0
                                #self.lock_enabled = False
                    else:
                        # РЕЖИМ БЕЗ БЛОКИРОВКИ: выбираем лучший трек по дистанции до центра
                        if mot_tracks:
                            mot_tracks.sort(key=lambda x: x['distance'] * (1.0 - x['conf']))
                            best_target = mot_tracks[0]
                            self.locked_track_id = best_target['track_id']
                            self.target_lost_frames = 0
                            self.lock_enabled = True

                            self.last_known_bbox = best_target['bbox']
                            self.logger.info(f"🔒 LOCKED: Новая цель ID={self.locked_track_id}, dist={best_target['distance']:.1f}")

                    # === ОТПРАВКА В STARK ===
                    if self.q_to_stark.empty():
                        if best_target:
                            data_packet = {
                                'bbox': best_target['bbox'],
                                'conf': best_target['conf'],
                                'track_id': best_target['track_id'],
                                'lost': False
                            }
                        else:
                            # Отправляем последний известный бокс (для гимбала)
                            data_packet = {
                                'bbox': self.last_known_bbox if self.last_known_bbox else [],
                                'conf': 0.0,
                                'track_id': self.locked_track_id if self.locked_track_id else -1,
                                'lost': True,
                                'lost_frames': self.target_lost_frames
                            }

                        self.q_to_stark.put([frame_stark, data_packet])

                    #if self.q_to_stark.empty():   ### new
                    #    self.q_to_qwen.put([frame_stark, vlm_qwen])   ### new
                    # === ВИЗУАЛИЗАЦИЯ ===
                    tracker.plot_results(frame, show_trajectories=self.renderer.show_trajectories)

                    # Отображаем статус блокировки


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