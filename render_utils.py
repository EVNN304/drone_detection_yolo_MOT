# render_utils.py — ПОЛНЫЙ РАБОЧИЙ КОД
import cv2
import logging
logger = logging.getLogger(__name__)

class BlockRenderer:
    def __init__(self, cfg: dict, class_names: dict = None):
        self.enabled = cfg.get("enabled", True)
        self.window_name = cfg.get("window_name", "Render Window")
        self.resize_factor = cfg.get("resize_factor", 1.0)
        self.show_trajectories = cfg.get("show_trajectories", False)
        self.waitkey_delay = cfg.get("waitkey_delay", 1)  # 🔹 По умолчанию 1 мс для реального времени
        self.class_names = class_names or {}
        self._window_created = False

    def _ensure_window(self):
        if not self._window_created and self.enabled:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            self._window_created = True
            logger.info(f"🎨 Created window: '{self.window_name}'")

    def _resize_and_show(self, img):
        if not self.enabled:
            return cv2.waitKey(1)
        if self.resize_factor != 1.0:
            h, w = img.shape[:2]
            img = cv2.resize(img, (int(w * self.resize_factor), int(h * self.resize_factor)))
        self._ensure_window()
        cv2.imshow(self.window_name, img)
        return cv2.waitKey(self.waitkey_delay)  # 🔹 Используем конфиг-значение

    def show(self, frame, overlay_text: str = ""):
        """Базовый режим (YOLO, Neuro)"""
        if not self.enabled or frame is None:
            return cv2.waitKey(1)
        img = frame.copy()
        if overlay_text:
            cv2.putText(img, overlay_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        return self._resize_and_show(img)

    # render_utils.py — ИСПРАВЛЕННЫЙ МЕТОД
    def show_with_mot_status(self, frame, locked_id, lost_frames, max_lost_frames, fps):
        """MOT: сохраняет нативные цвета BoxMOT + статусы"""
        if not self.enabled or frame is None:
            return cv2.waitKey(1)
        img = frame.copy()  # BoxMOT уже нарисовал треки — не трогаем

        # Ваши статусы (те же координаты и цвета, что у вас были)
        y = 30
        cv2.putText(img, f"FPS: {fps:.1f}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        y += 35
        cv2.putText(img, f"LOCK: {locked_id if locked_id is not None else 'NONE'}", (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 0, 0), 2)
        y += 35
        if lost_frames > 0:
            cv2.putText(img, f"LOST: {lost_frames}/{max_lost_frames}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), 2)

        return self._resize_and_show(img)

    def show_with_stark_status(self, frame, bbox=None, track_id=-1, fps=0.0, status_text=""):
        """STARK: красный бокс + статус"""
        if not self.enabled or frame is None:
            return cv2.waitKey(1)
        img = frame.copy()
        if bbox:
            x, y, w, h = map(int, bbox)
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 3)
            cv2.putText(img, f"STARK:{track_id}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        y = 30
        if status_text:
            cv2.putText(img, status_text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2); y += 30
        cv2.putText(img, f"FPS: {fps:.1f}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return self._resize_and_show(img)

    def show_stark_telemetry(self, frame, telemetry: dict, fps: float = 0.0):
        """
        Отрисовка STARK с телеметрией — БЕЗ хоткеев, полностью автоматически.
        """
        if not self.enabled or frame is None:
            return cv2.waitKey(1)  # просто пропускаем, если окно выключено

        img = frame.copy()
        font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
        y_pos = 30

        bbox = telemetry.get('bbox')
        if bbox:
            x, y, w, h = map(int, bbox)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.putText(img, f"ID:{telemetry.get('track_id', -1) - 1}", (x, y - 8), font, 0.5, (0, 255, 0), 1)

        alert = " ⚠️ LOST!" if telemetry.get('lost_alert', False) or telemetry.get('mot_lost', False) else ""
        state_text = "Tracking!" if telemetry.get('is_tracking', False) else "Searching..."

        lines = [
            f"ID:{telemetry.get('track_id', -1) - 1} | {state_text}",
            f"Fail:{telemetry.get('consecutive_failures', 0)}/{telemetry.get('max_consecutive_failures', 7)} | MOT Lost:{telemetry.get('mot_lost_frames', 0)}{alert}",
        ]

        for line in lines:
            color = (0, 255, 0) if "Lost" not in line else (0, 0, 255)
            cv2.putText(img, line, (20, y_pos), font, scale, color, thick)
            y_pos += 25

        cv2.putText(img, f"FPS: {fps:.1f}", (10, 80), font, 0.7, (0, 255, 255), 2)

        return self._resize_and_show(img)  # ← здесь уже есть resize + waitkey_delay из конфига


    def show_video_stream(self, frame, status_text="", fps=0.0):
        """для основного видеопотока"""
        if not self.enabled or frame is None:
            return cv2.waitKey(1)
        img = frame.copy()
        y = 30
        if status_text:
            cv2.putText(img, status_text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2); y += 30
        cv2.putText(img, f"FPS: {fps:.1f}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return self._resize_and_show(img)

    def destroy(self):
        if self._window_created:
            cv2.destroyWindow(self.window_name)
            self._window_created = False