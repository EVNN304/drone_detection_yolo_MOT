import importlib
import os
from collections import OrderedDict
from lib.test.evaluation.environment import env_settings
from render_utils import BlockRenderer

import time
import cv2 as cv
import multiprocessing as mp
from lib.utils.lmdb_utils import decode_img
from loggers import *
from shm_utils import SharedFrameReader  # ← ДОБАВЛЕН ИМПОРТ


def trackerlist(name: str, parameter_name: str, dataset_name: str, run_ids=None, display_name: str = None,
                result_only=False):
    """Generate list of trackers."""
    if run_ids is None or isinstance(run_ids, int):
        run_ids = [run_id]
    return [Tracker(name, parameter_name, dataset_name, run_id, display_name, result_only) for run_id in run_ids]


class Tracker:
    """Wraps the tracker for evaluation and running purposes."""

    def __init__(self, q_frame: mp.Queue, name: str, parameter_name: str, dataset_name: str,
                 run_id: int = None, display_name: str = None, result_only=False,
                 render_cfg=None, class_names=None,
                 shm_mot_to_sot_name_0=None, shm_mot_to_sot_name_1=None,  # ← ДОБАВЛЕНО
                 shm_mot_to_sot_free_queue=None, shm_mot_to_sot_ready_queue=None,  # ← ДОБАВЛЕНО
                 frame_h=1080, frame_w=1920):  # ← ДОБАВЛЕНО
        assert run_id is None or isinstance(run_id, int)

        self.name = name
        self.parameter_name = parameter_name
        self.dataset_name = dataset_name
        self.run_id = run_id
        self.display_name = display_name
        self.q_frame = q_frame
        self.renderer = BlockRenderer(render_cfg or {}, class_names)

        # === SHARED MEMORY ДЛЯ ЧТЕНИЯ КАДРА ОТ MOT ===
        self.shm_mot_to_sot_name_0 = shm_mot_to_sot_name_0
        self.shm_mot_to_sot_name_1 = shm_mot_to_sot_name_1
        self.shm_mot_to_sot_free_queue = shm_mot_to_sot_free_queue
        self.shm_mot_to_sot_ready_queue = shm_mot_to_sot_ready_queue
        self.frame_h = frame_h
        self.frame_w = frame_w
        self.shm_reader = None  # Создастся при первом использовании

        print(f"PARAM ", self.name, self.parameter_name, self.dataset_name)

        env = env_settings()
        if self.run_id is None:
            self.results_dir = '{}/{}/{}'.format(env.results_path, self.name, self.parameter_name)
        else:
            self.results_dir = '{}/{}/{}_{:03d}'.format(env.results_path, self.name, self.parameter_name, self.run_id)
        if result_only:
            self.results_dir = '{}/{}'.format(env.results_path, self.name)

        tracker_module_abspath = f"/home/usr/PycharmProjects/STARK/Stark/lib/models/stark/stark_s.py"
        if os.path.isfile(tracker_module_abspath):
            tracker_module = importlib.import_module('lib.test.tracker.{}'.format(self.name))
            self.tracker_class = tracker_module.get_tracker_class()
        else:
            self.tracker_class = None

        # === НАСТРОЙКИ ВАЛИДАЦИИ (PTZ OPTIMIZED) ===
        # Переопределяем frame_w, frame_h если переданы
        if frame_w and frame_h:
            self.frame_w, self.frame_h = frame_w, frame_h

        # 1. IoU порог
        self.iou_threshold = 0.5

        # 2. Hysteresis: требуем 7 подряд плохих кадров для сброса
        self.max_consecutive_failures = 7
        self.consecutive_failures = 0

        # 3. Grace Period: первые 10 кадров не валидируем
        self.grace_period_frames = 10
        self.frames_since_init = 0

        # 4. Минимальная длительность трека перед разрешением сброса
        self.min_track_duration = 15
        self.frames_tracked = 0

        # 5. Порог уверенности MOT
        self.mot_confidence_threshold = 0.5

        # 6. Сглаживание (EMA alpha)
        self.smoothing_alpha = 0.7
        self.smoothed_bbox = None

        # === 7. MOT TRACK ID ===
        self.active_track_id = -1
        self.id_mismatch_count = 0
        self.max_id_mismatches = 3

        # === 8. PTZ ЛОГИКА ===
        self.target_lost_alert = False
        self.mot_lost_frames = 0

        self.logger = logging.getLogger(__name__)

    def create_tracker(self, params):
        tracker = self.tracker_class(params, self.dataset_name)
        return tracker

    def run_process(self, daemon=True):
        prc = mp.Process(target=self.run_video, args=(), daemon=daemon)
        prc.start()
        return prc

    def run_sequence(self, seq, debug=None):
        params = self.get_parameters()
        debug_ = debug
        if debug is None:
            debug_ = getattr(params, 'debug', 0)
        params.debug = debug_
        init_info = seq.init_info()
        tracker = self.create_tracker(params)
        output = self._track_sequence(tracker, seq, init_info)
        return output

    def _track_sequence(self, tracker, seq, init_info):
        output = {'target_bbox': [], 'time': []}
        if tracker.params.save_all_boxes:
            output['all_boxes'] = []
            output['all_scores'] = []

        def _store_outputs(tracker_out: dict, defaults=None):
            defaults = {} if defaults is None else defaults
            for key in output.keys():
                val = tracker_out.get(key, defaults.get(key, None))
                if key in tracker_out or val is not None:
                    output[key].append(val)

        image = self._read_image(seq.frames[0])
        start_time = time.time()
        out = tracker.initialize(image, init_info)
        if out is None:
            out = {}

        prev_output = OrderedDict(out)
        init_default = {'target_bbox': init_info.get('init_bbox'), 'time': time.time() - start_time}
        if tracker.params.save_all_boxes:
            init_default['all_boxes'] = out['all_boxes']
            init_default['all_scores'] = out['all_scores']

        _store_outputs(out, init_default)

        for frame_num, frame_path in enumerate(seq.frames[1:], start=1):
            image = self._read_image(frame_path)
            start_time = time.time()
            info = seq.frame_info(frame_num)
            info['previous_output'] = prev_output
            out = tracker.track(image, info)
            prev_output = OrderedDict(out)
            _store_outputs(out, {'time': time.time() - start_time})

        for key in ['target_bbox', 'all_boxes', 'all_scores']:
            if key in output and len(output[key]) <= 1:
                output.pop(key)

        return output

    # === МЕТОДЫ ВАЛИДАЦИИ ===

    def _calculate_iou(self, box1, box2):
        """Расчёт IoU между двумя боксами. Box format: [x, y, w, h]"""
        if not box1 or not box2:
            return 0.0
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        xi1, yi1 = max(x1, x2), max(y1, y2)
        xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)

        inter_w = max(0, xi2 - xi1)
        inter_h = max(0, yi2 - yi1)
        inter_area = inter_w * inter_h

        area1 = w1 * h1
        area2 = w2 * h2
        union_area = area1 + area2 - inter_area

        return inter_area / (union_area + 1e-9)

    def _check_bbox_sanity(self, bbox):
        """Проверка 'здоровья' бокса STARK."""
        if not bbox or len(bbox) != 4:
            return False

        x, y, w, h = bbox
        area = w * h
        frame_area = self.frame_w * self.frame_h

        if x < -w * 2 or y < -h * 2 or x > self.frame_w + w * 2 or y > self.frame_h + h * 2:
            return False

        if area > frame_area * 0.55:
            return False

        return True

    def _validate_with_mot(self, stark_bbox, mot_data):
        """Сверка с MOT с использованием Hysteresis + Track ID."""

        if isinstance(mot_data, dict):
            mot_bbox = mot_data.get('bbox', [])
            mot_conf = mot_data.get('conf', 1.0)
            mot_id = mot_data.get('track_id', -1)
            mot_lost = mot_data.get('lost', False)
            self.mot_lost_frames = mot_data.get('lost_frames', 0)
        else:
            mot_bbox = mot_data if mot_data else []
            mot_conf = 1.0
            mot_id = -1
            mot_lost = False
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
                self.logger.info(f"⚠️ STARK: ID не совпадает! Активный: {self.active_track_id}, MOT: {mot_id}")
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
                self.logger.info(
                    f"⚠️ STARK: IoU низкий ({iou:.2f}) в течение {self.consecutive_failures} кадров. Сброс.")
                self.target_lost_alert = True
                return False
            else:
                self.logger.info(
                    f"⚠️ STARK: IoU низкий ({iou:.2f}), кадр {self.consecutive_failures}/{self.max_consecutive_failures}. Ждем...")
                return True

        if not id_match and iou >= self.iou_threshold:
            self.logger.info(f"✅ STARK: ID изменился, но IoU высокий. Обновляем ID: {self.active_track_id} → {mot_id}")
            self.active_track_id = mot_id
            self.consecutive_failures = 0
            self.id_mismatch_count = 0
            return True

        if not id_match and iou < self.iou_threshold:
            if self.id_mismatch_count >= self.max_id_mismatches:
                self.logger.info(f"❌ STARK: ID не совпадает + IoU низкий ({iou:.2f}). СБРОС ТРЕКА.")
                self.target_lost_alert = True
                return False
            else:
                self.logger.info(
                    f"⚠️ STARK: ID не совпадает, но ждем подтверждения ({self.id_mismatch_count}/{self.max_id_mismatches})")
                self.consecutive_failures += 1
                return True

        self.consecutive_failures = 0
        return True

    def _smooth_bbox(self, bbox):
        """Сглаживание координат бокса (Exponential Moving Average)"""
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
        """Извлечение bbox из данных очереди."""
        if isinstance(data, dict):
            return data.get('bbox', [])
        elif isinstance(data, list):
            return data if len(data) == 4 else []
        else:
            return []

    def _get_frame_and_data(self):
        """
        Универсальный метод получения кадра и данных.
        Поддерживает оба формата:
        - Новый (SHM): data_packet из Queue + кадр из SHM
        - Старый (Queue): [frame, data_packet] из Queue
        """
        item = self.q_frame.get()

        # === НОВЫЙ РЕЖИМ: SHM ===
        if isinstance(item, dict):
            # data_packet пришёл отдельно, кадр читаем из SHM
            if self.shm_reader is None:
                if self.shm_mot_to_sot_name_0:
                    self.shm_reader = SharedFrameReader(
                        self.frame_h, self.frame_w,
                        self.shm_mot_to_sot_name_0,
                        self.shm_mot_to_sot_name_1,
                        self.shm_mot_to_sot_free_queue,
                        self.shm_mot_to_sot_ready_queue
                    )
                    self.logger.info(f"✅ STARK: подключён к Shared Memory от MOT ({self.frame_w}x{self.frame_h})")
                else:
                    self.logger.error("❌ STARK: SHM не настроен!")
                    return None, None

            frame = self.shm_reader.read_frame()
            return frame, item

        # === СТАРЫЙ РЕЖИМ: Queue ===
        elif isinstance(item, list) and len(item) == 2:
            frame, data = item
            return frame, data

        else:
            self.logger.error(f"❌ STARK: Неизвестный формат данных: {type(item)}")
            return None, None

    def run_video(self):
        """Run the tracker with the videofile."""
        params = self.get_parameters()
        params.tracker_name = self.name
        params.param_name = self.parameter_name
        multiobj_mode = getattr(params, 'multiobj_mode', getattr(self.tracker_class, 'multiobj_mode', 'default'))

        if multiobj_mode == 'default':
            tracker = self.create_tracker(params)
        elif multiobj_mode == 'parallel':
            from lib.test.tracker.basetracker import MultiObjectWrapper
            tracker = MultiObjectWrapper(self.tracker_class, params, None, fast_load=True)
        else:
            raise ValueError('Unknown multi object mode {}'.format(multiobj_mode))

        output_boxes = []
        is_tracking = False

        if not logging.getLogger().handlers:
            setup_logging()

        def _build_init_info(box):
            return {'init_bbox': box}

        # 1. Цикл ожидания инициализации
        while True:
            if not self.q_frame.empty():
                frame, data = self._get_frame_and_data()
                if frame is None:
                    continue

                bbox = self._extract_bbox_from_data(data)

                if bbox:
                    if isinstance(data, dict):
                        track_id = data.get('track_id', -1)
                        mot_lost = data.get('lost', False)
                    else:
                        track_id = -1
                        mot_lost = False

                    self.logger.info(f"✅ STARK: Инициализация цели Box={bbox}, ID={track_id}")
                    tracker.initialize(frame, _build_init_info(bbox))
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

        # 2. Основной цикл трекинга
        start_time = time.time()
        frame_count = 0
        state = None  # Для telemetry

        while True:
            if not self.q_frame.empty():
                frame, data = self._get_frame_and_data()
                if frame is None:
                    break

                if isinstance(data, dict):
                    mot_bbox = data.get('bbox', [])
                    mot_conf = data.get('conf', 1.0)
                    mot_id = data.get('track_id', -1)
                    mot_lost = data.get('lost', False)
                    mot_lost_frames = data.get('lost_frames', 0)
                    mot_data = {'bbox': mot_bbox, 'conf': mot_conf, 'track_id': mot_id,
                                'lost': mot_lost, 'lost_frames': mot_lost_frames}
                else:
                    mot_bbox = data if data else []
                    mot_conf = 1.0
                    mot_id = -1
                    mot_lost = False
                    mot_lost_frames = 0
                    mot_data = mot_bbox

                frame_disp = frame.copy()

                if is_tracking:
                    out = tracker.track(frame)
                    stark_bbox = out.get('target_bbox', None)

                    if stark_bbox:
                        self.frames_since_init += 1
                        self.frames_tracked += 1

                        if not self._check_bbox_sanity(stark_bbox):
                            self.logger.info("⚠️ STARK: Бокс неадекватен (разнос). Сброс трека.")
                            is_tracking = False
                            self.smoothed_bbox = None
                            self.active_track_id = -1
                            self.target_lost_alert = True
                            continue

                        if self.frames_tracked >= self.min_track_duration:
                            if not self._validate_with_mot(stark_bbox, mot_data):
                                self.logger.info("⚠️ STARK: Валидация с MOT не пройдена. Сброс трека.")
                                is_tracking = False
                                self.smoothed_bbox = None
                                self.active_track_id = -1
                                continue
                        else:
                            self.consecutive_failures = 0
                            self.id_mismatch_count = 0

                        smoothed_state = self._smooth_bbox(stark_bbox)
                        state = [int(s) for s in smoothed_state]
                        output_boxes.append(state)
                    else:
                        is_tracking = False
                        self.active_track_id = -1
                        self.target_lost_alert = True
                        continue
                else:
                    mot_bbox_check = self._extract_bbox_from_data(data)
                    if mot_bbox_check:
                        if isinstance(data, dict):
                            track_id = data.get('track_id', -1)
                            mot_lost = data.get('lost', False)
                        else:
                            track_id = -1
                            mot_lost = False

                        tracker.initialize(frame, _build_init_info(mot_bbox_check))
                        is_tracking = True
                        output_boxes.append(mot_bbox_check)
                        self.active_track_id = track_id
                        self.target_lost_alert = mot_lost
                        self.consecutive_failures = 0
                        self.id_mismatch_count = 0
                        self.smoothed_bbox = None
                        self.logger.info(f"✅ STARK: Новая цель захвачена ID={track_id}")

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
                    'bbox': state if state else [],
                }

                self.renderer.show_stark_telemetry(frame_disp, telemetry, fps)

    def get_parameters(self):
        """Get parameters."""
        param_module = importlib.import_module('lib.test.parameter.{}'.format(self.name))
        params = param_module.parameters(self.parameter_name)
        return params

    def _read_image(self, image_file: str):
        if isinstance(image_file, str):
            im = cv.imread(image_file)
            return cv.cvtColor(im, cv.COLOR_BGR2RGB)
        elif isinstance(image_file, list) and len(image_file) == 2:
            return decode_img(image_file[0], image_file[1])
        else:
            raise ValueError("type of image_file should be str or list")


if __name__ == '__main__':
    tracker = Tracker("stark_s", "baseline", "video")