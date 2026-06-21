# config_router.py
import yaml
import logging
import os
import inspect
from convert_weights_TRT import Converter

logger = logging.getLogger(__name__)

class ConfigRouter:
    def __init__(self, path="pipeline_config.yaml"):
        with open(path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)
        logger.info("✅ Config loaded from %s", path)

    def apply_to_yolo(self, yolo_obj):
        m = self.cfg["model"]

        # 🔧 Сначала конвертируем веса если нужно
        original_path = m["yolo_weights"]
        m["yolo_weights"] = self.get_converted_weights_path(original_path)

        # Теперь применяем параметры к уже готовым весам
        yolo_obj.set_path_model(m["yolo_weights"])
        yolo_obj.set_size_inp_layers(m["img_size"])
        yolo_obj.set_conf_model(m["conf_threshold"])
        yolo_obj.set_nms_type(m["nms_type"])
        if m.get("nms_params"):
            yolo_obj.set_nms_params(**m["nms_params"])

        if "half_flag" in m:
            yolo_obj.set_half_flag(m["half_flag"])
        if "verbose" in m:
            yolo_obj.set_verbose(m["verbose"])

        if "device" in m and hasattr(yolo_obj, "set_device"):
            yolo_obj.set_device(m["device"])

        logger.info("📦 YOLO model config applied")

    def apply_to_mot(self, mot_obj):
        m = self.cfg["mot"]
        mot_obj.set_mot_method(m["method"])
        mot_obj.set_path_weights_reid(m["reid_weights"])
        mot_obj.set_flag_half(m["half"])
        mot_obj.set_flag_show_track(m["show_track"])
        mot_obj.set_max_lost_frames(m["max_lost_frames"])
        logger.info("📦 MOT config applied")

    def apply_to_tracker(self, tracker_obj):
        t = self.cfg["tracker"]
        logger.info("📦 Tracker config applied")

    def get_selected_tracker(self) -> tuple[str, dict]:
        """
        Возвращает (имя_трекера, конфиг_трекера).
        Логика выбора:
        1. Если только один enabled=true → он
        2. Если оба true → первый в priority_order
        3. Если оба false → default_tracker
        """
        tracker_cfg = self.cfg.get("tracker", {})
        priority = tracker_cfg.get("priority_order", ["nanotrack", "stark_s"])
        default = tracker_cfg.get("default_tracker", "stark_s")

        # Собираем список включённых трекеров
        enabled_trackers = []
        for name in ["stark_s", "nanotrack"]:
            if tracker_cfg.get(name, {}).get("enabled", False):
                enabled_trackers.append(name)

        # Выбираем по приоритету
        if len(enabled_trackers) == 1:
            selected = enabled_trackers[0]
        elif len(enabled_trackers) > 1:
            # Коллизия: берём первый из priority_order
            selected = next((t for t in priority if t in enabled_trackers), default)
            logger.warning(f"⚠️ Tracker collision: both enabled. Selected '{selected}' by priority.")
        else:
            # Ни одного не включено → дефолт
            selected = default
            logger.info(f"ℹ️ No tracker enabled. Using default: '{selected}'")

        return selected, tracker_cfg.get(selected, {})


    def get_converted_weights_path(self, original_path: str) -> str:
        """
        Возвращает путь к весам: либо оригинальный .pt, либо сконвертированный .engine.
        Конвертация происходит только если:
        - включена в конфиге (tensorrt.enabled: true)
        - исходный файл имеет расширение .pt
        - engine ещё не существует рядом
        """
        trt_cfg = self.cfg.get("tensorrt", {})

        # 1. Если конвертация выключена → возвращаем как есть
        if not trt_cfg.get("enabled", False):
            return original_path

        # 2. Если уже .engine → не трогаем (защита от исключения)
        if original_path.lower().endswith(".engine"):
            logger.info(f"✅ TRT: Already engine format: {original_path}")
            return original_path

        # 3. Если engine уже есть рядом → не конвертируем заново (кэш)
        engine_path = os.path.splitext(original_path)[0] + ".engine"
        if os.path.exists(engine_path):
            logger.info(f"✅ TRT: Engine exists, skipping conversion: {engine_path}")
            return engine_path

        # 4. Запуск конвертации
        logger.info(f"🔧 TRT: Converting {original_path} → {engine_path} ...")
        try:
            converter = Converter(original_path)

            # Пробрасываем ВСЕ параметры из конфига
            converter.set_export_param("format", trt_cfg.get("format", "engine"))
            converter.set_export_param("device", trt_cfg.get("device", 0))
            converter.set_export_param("half", trt_cfg.get("half", True))
            converter.set_export_param("int8", trt_cfg.get("int8", False))
            converter.set_export_param("dynamic", trt_cfg.get("dynamic", True))
            converter.set_export_param("simplify", trt_cfg.get("simplify", True))
            converter.set_export_param("workspace", trt_cfg.get("workspace", 1.0))
            converter.set_export_param("imgsz", tuple(trt_cfg.get("imgsz", [288, 288])))
            converter.set_export_param("batch", trt_cfg.get("batch", 16))
            converter.set_export_param("nms", trt_cfg.get("nms", False))
            converter.set_export_param("verbose", trt_cfg.get("verbose", True))

            converter.start_to_convert()

            if os.path.exists(engine_path):
                logger.info(f"✅ TRT: Conversion successful. Using {engine_path}")
                return engine_path
            else:
                logger.warning(f"⚠️ TRT: Conversion finished but engine not found. Fallback to {original_path}")
                return original_path

        except Exception as e:
            logger.error(f"❌ TRT: Conversion failed: {e}", exc_info=True)
            return original_path  # Fallback на оригинал, пайплайн не упадёт



    def apply_to_recording(self, neuro_obj):
        r = self.cfg.get("recording", {})
        if r.get("names_files"):
            neuro_obj.set_name_save(r["names_files"])
        if r.get("name_folder"):
            neuro_obj.set_name_folder(r["name_folder"])
        # saved_mode обычно задаётся в __init__, но если есть сеттер:
        # if r.get("saved_mode") is not None and hasattr(neuro_obj, "set_saved_mode"):
        #     neuro_obj.set_saved_mode(r["saved_mode"])

        if "size_cut_w" in r and hasattr(neuro_obj, "set_size_cut_w"):
            neuro_obj.set_size_cut_w(r["size_cut_w"])
        if "size_cut_h" in r and hasattr(neuro_obj, "set_size_cut_h"):
            neuro_obj.set_size_cut_h(r["size_cut_h"])
        if "min_conf" in r and hasattr(neuro_obj, "set_min_conf"):
            neuro_obj.set_min_conf(r["min_conf"])

        logger.info("📦 Recording config applied")

    def get_mot_tracker_params(self, method: int) -> dict:
        """Собирает параметры: базовые + специфичные для выбранного метода."""
        mot_cfg = self.cfg.get("mot", {})
        base = mot_cfg.get("base", {})
        specific = mot_cfg.get("params", {}).get(method, {})
        return {**base, **specific}



    def get_input_cfg(self):
        return self.cfg["input"]

    def get_render_cfg(self):
        return self.cfg["render"]

    def get_inference_cfg(self):
        """Возвращает настройки инференса с дефолтами."""
        defaults = {
            "batch_size": 16,
            "half_precision": True,
            "device": "cuda:0",
            "augment": False,
            "visualize": False
        }
        cfg = self.cfg.get("inference", {})
        return {**defaults, **cfg}

    def get_render_cfg(self):
        """Возвращает настройки рендера."""
        return self.cfg.get("render", {})

    def get_vlm_cfg(self) -> dict:
        """Возвращает конфигурацию VLM с дефолтами."""
        defaults = {
            "enabled": False,
            "send_interval": 30,
            "max_objects": 3,
            "device": "cuda:1",
            "model_id": "Qwen/Qwen2.5-VL-3B-Instruct",
            "crop_size": 288
        }
        vlm_cfg = self.cfg.get("vlm", {})
        return {**defaults, **vlm_cfg}

    def is_vlm_enabled(self) -> bool:
        """Проверяет, включен ли VLM."""
        return self.cfg.get("vlm", {}).get("enabled", False)