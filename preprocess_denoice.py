import cv2
import numpy as np
from typing import Tuple, Optional, Union


def preprocess_moving_camera_frame(
        image: np.ndarray,
        motion_level: str = 'auto',
        enable_stabilization: bool = False,
        stabilization_params: dict = None,
        enable_edge_preservation: bool = True,
        detection_focus: str = 'drones'  # 'drones', 'general', 'small_objects'
) -> np.ndarray:
    """
    Специализированная предобработка для движущейся камеры на поворотном устройстве.
    Оптимизирована для сохранения деталей дронов при движении камеры.

    Args:
        image: Входное изображение (BGR формат)
        motion_level: 'auto', 'low', 'medium', 'high' - уровень движения камеры
        enable_stabilization: Включить цифровую стабилизацию (только для очень медленного движения)
        stabilization_params: Параметры стабилизации {'max_shift': 10, 'smooth_factor': 0.2}
        enable_edge_preservation: Сохранять края для лучшей детекции дронов
        detection_focus: 'drones' (оптимизация для дронов), 'general', 'small_objects'

    Returns:
        Предобработанное изображение в BGR формате
    """

    if image is None or image.size == 0:
        return image

    # Создаем копию, чтобы не изменять оригинал
    processed = image.copy().astype(np.float32)  # Работаем во float32 для точности
    h, w = processed.shape[:2]

    # ========== ШАГ 1: АВТОМАТИЧЕСКАЯ ОЦЕНКА ДВИЖЕНИЯ ==========
    current_motion_level = motion_level
    if motion_level == 'auto':
        current_motion_level = estimate_camera_motion_level(processed.astype(np.uint8), prev_frame=None)

    # ========== ШАГ 2: СТАБИЛИЗАЦИЯ (ТОЛЬКО ДЛЯ МЕДЛЕННОГО ДВИЖЕНИЯ) ==========


    # ========== ШАГ 3: АДАПТИВНОЕ ШУМОПОДАВЛЕНИЕ ДЛЯ ДВИЖУЩЕЙСЯ КАМЕРЫ ==========
    processed = apply_motion_robust_denoising(
        processed.astype(np.uint8),
        motion_level=current_motion_level,
        detection_focus=detection_focus
    ).astype(np.float32)

    # ========== ШАГ 4: УСИЛЕНИЕ КОНТРАСТА С ФОКУСОМ НА ДРОНАХ ==========
    processed = apply_drone_optimized_contrast(
        processed.astype(np.uint8),
        motion_level=current_motion_level,
        detection_focus=detection_focus
    ).astype(np.float32)

    # ========== ШАГ 5: ЗАЩИТА КРАЕВ И МЕЛКИХ ДЕТАЛЕЙ ==========
    if enable_edge_preservation:
        processed = preserve_drone_edges(processed.astype(np.uint8), detection_focus=detection_focus).astype(np.float32)

    # ========== ШАГ 6: АДАПТИВНАЯ НОРМАЛИЗАЦИЯ ЯРКОСТИ ==========
    processed = adaptive_brightness_normalization(processed.astype(np.uint8), motion_level=current_motion_level).astype(
        np.float32)

    # ========== ФИНАЛЬНАЯ КОНВЕРТАЦИЯ ==========
    processed = np.clip(processed, 0, 255).astype(np.uint8)

    return processed


# ========== ИСПРАВЛЕННАЯ ФУНКЦИЯ СОХРАНЕНИЯ КРАЕВ ==========

def preserve_drone_edges(image: np.ndarray, detection_focus: str = 'drones') -> np.ndarray:
    """
    Сохранение и усиление краев для лучшей детекции дронов.
    Исправленная версия с правильной обработкой типов данных.
    """
    if detection_focus != 'drones' or image is None or image.size == 0:
        return image.copy()

    # Создаем копию для работы
    img_float = image.astype(np.float32)

    # 1. Умеренное нерезкое маскирование
    blurred = cv2.GaussianBlur(image, (3, 3), 0.8)
    blurred_float = blurred.astype(np.float32)

    # Применяем нерезкое маскирование с фиксированными коэффициентами
    sharpened = cv2.addWeighted(img_float, 1.4, blurred_float, -0.4, 0)
    sharpened = np.clip(sharpened, 0, 255)

    # 2. Обнаружение краев
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Адаптивные параметры Canny в зависимости от разрешения
    height, width = image.shape[:2]
    if height > 480:  # Высокое разрешение
        low_thresh = 40
        high_thresh = 100
    else:  # Низкое разрешение
        low_thresh = 30
        high_thresh = 80

    edges = cv2.Canny(gray, low_thresh, high_thresh)

    # 3. Создание маски краев с размытием для плавного перехода
    edge_mask = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
    edge_mask = cv2.GaussianBlur(edge_mask.astype(np.float32), (5, 5), 1.0)
    edge_mask = edge_mask / 255.0  # Нормализация в [0, 1]
    edge_mask = np.clip(edge_mask, 0, 1)  # Убедимся, что значения в правильном диапазоне

    # 4. Расширяем маску до 3 каналов
    if len(edge_mask.shape) == 2:
        edge_mask_3ch = np.stack([edge_mask, edge_mask, edge_mask], axis=2)
    else:
        edge_mask_3ch = edge_mask

    # 5. Плавное смешивание с использованием маски
    # Коэффициент усиления для краев (0.3 = 30% дополнительного усиления на краях)
    edge_boost_factor = 0.3

    # Вычисляем итоговое изображение
    result = img_float * (1 - edge_mask_3ch * edge_boost_factor) + sharpened * (edge_mask_3ch * edge_boost_factor)

    # 6. Нормализация и конвертация обратно в uint8
    result = np.clip(result, 0, 255).astype(np.uint8)

    return result


# ========== ИСПРАВЛЕННЫЕ ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ==========

def estimate_camera_motion_level(current_frame: np.ndarray, prev_frame: Optional[np.ndarray] = None) -> str:
    """
    Оценка уровня движения камеры на основе анализа кадра.
    """
    if current_frame is None or current_frame.size == 0:
        return 'medium'

    gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

    # Оценка резкости (движение обычно снижает резкость)
    try:
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    except:
        laplacian_var = 50  # Значение по умолчанию

    # Оценка уровня шума
    noise_level = np.std(gray) / 255.0 if gray.size > 0 else 0.1

    # Эвристическая оценка
    if laplacian_var > 100 and noise_level < 0.1:
        return 'low'
    elif laplacian_var > 50 and noise_level < 0.15:
        return 'medium'
    else:
        return 'high'


def apply_motion_robust_denoising(
        image: np.ndarray,
        motion_level: str = 'medium',
        detection_focus: str = 'drones'
) -> np.ndarray:
    """
    Шумоподавление, устойчивое к движению камеры.
    """
    if image is None or image.size == 0:
        return image.copy()

    h, w = image.shape[:2]
    is_small_fragment = h < 128 or w < 128

    try:
        if motion_level == 'low':
            if detection_focus == 'drones':
                return cv2.bilateralFilter(image, 5, 40, 40)
            else:
                return cv2.medianBlur(image, 3)

        elif motion_level == 'medium':
            if detection_focus == 'drones':
                temp = cv2.medianBlur(image, 3)
                return cv2.bilateralFilter(temp, 7, 50, 50)
            else:
                return cv2.bilateralFilter(image, 7, 60, 60)

        else:  # high motion
            if is_small_fragment:
                if detection_focus == 'drones':
                    return cv2.GaussianBlur(image, (3, 3), 0.5)
                else:
                    return cv2.medianBlur(image, 3)
            else:
                if detection_focus == 'drones':
                    return apply_drone_focused_denoising(image)
                else:
                    return cv2.GaussianBlur(image, (3, 3), 1.0)

    except Exception as e:
        print(f"Error in denoising: {e}")
        return image.copy()


def apply_drone_focused_denoising(image: np.ndarray) -> np.ndarray:
    """
    Специализированное шумоподавление с фокусом на сохранении дронов.
    """
    if image is None or image.size == 0:
        return image.copy()

    try:
        # Простой, но эффективный подход для движущейся камеры
        # Легкое медианное размытие для удаления артефактов
        temp = cv2.medianBlur(image, 3)
        # Билатеральная фильтрация для сохранения краев
        return cv2.bilateralFilter(temp, 5, 30, 30)

    except Exception as e:
        print(f"Error in drone focused denoising: {e}")
        return image.copy()


def apply_drone_optimized_contrast(
        image: np.ndarray,
        motion_level: str = 'medium',
        detection_focus: str = 'drones'
) -> np.ndarray:
    """
    Усиление контраста с фокусом на детекции дронов.
    """
    if image is None or image.size == 0:
        return image.copy()

    try:
        if detection_focus != 'drones':
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l_clahe = clahe.apply(l)
            lab_clahe = cv2.merge((l_clahe, a, b))
            return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Адаптивные параметры CLAHE
        if motion_level == 'low':
            clip_limit = 1.8
            tile_size = (10, 10)
        elif motion_level == 'medium':
            clip_limit = 2.2
            tile_size = (8, 8)
        else:  # high motion
            clip_limit = 2.5
            tile_size = (6, 6)

        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
        l_clahe = clahe.apply(l)

        # Дополнительное усиление для верхних частей кадра
        height = l_clahe.shape[0]
        if height > 0:
            # Создаем вертикальный градиент для усиления
            y_coords = np.linspace(1.2, 1.0, height)[:, np.newaxis]
            l_enhanced = l_clahe.astype(np.float32) * y_coords.astype(np.float32)
            l_enhanced = np.clip(l_enhanced, 0, 255).astype(np.uint8)
        else:
            l_enhanced = l_clahe

        lab_enhanced = cv2.merge((l_enhanced, a, b))
        return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    except Exception as e:
        print(f"Error in contrast enhancement: {e}")
        return image.copy()


def adaptive_brightness_normalization(image: np.ndarray, motion_level: str = 'medium') -> np.ndarray:
    """
    Адаптивная нормализация яркости.
    """
    if image is None or image.size == 0:
        return image.copy()

    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if gray.size == 0:
            return image.copy()

        avg_brightness = np.mean(gray)

        # Целевая яркость зависит от движения
        if motion_level == 'low':
            target_brightness = 120
        elif motion_level == 'medium':
            target_brightness = 110
        else:  # high motion
            target_brightness = 100

        # Адаптивный коэффициент с ограничениями
        if avg_brightness > 10:
            alpha = target_brightness / avg_brightness
            alpha = np.clip(alpha, 0.7, 1.5)

            # Применяем преобразование
            result = image.astype(np.float32) * alpha
            result = np.clip(result, 0, 255).astype(np.uint8)

            return result

        return image.copy()

    except Exception as e:
        print(f"Error in brightness normalization: {e}")
        return image.copy()


# ========== УЛЬТРА-БЫСТРАЯ И НАДЕЖНАЯ ВЕРСИЯ ДЛЯ РЕАЛЬНОГО ВРЕМЕНИ ==========

def preprocess_moving_camera_realtime(image: np.ndarray) -> np.ndarray:
    """
    Сверхбыстрая и надежная версия для реального времени.
    Минимальная обработка с максимальной стабильностью.
    """
    if image is None or image.size == 0:
        return image.copy() if image is not None else np.zeros((480, 640, 3), dtype=np.uint8)

    try:
        # Работаем с копией
        processed = image.copy()

        h, w = processed.shape[:2]
        is_low_res = min(h, w) < 720

        # 1. БЫСТРОЕ МЕДИАННОЕ РАЗМЫТИЕ (только для высоких разрешений)
        if not is_low_res and max(h, w) > 1080:
            processed = cv2.medianBlur(processed, 3)

        # 2. БИЛАТЕРАЛЬНАЯ ФИЛЬТРАЦИЯ С МИНИМАЛЬНЫМИ ПАРАМЕТРАМИ
        if is_low_res:
            processed = cv2.bilateralFilter(processed, 3, 20, 20)
        else:
            processed = cv2.bilateralFilter(processed, 2, 15, 15)

        # 3. ОПТИМИЗИРОВАННАЯ CLAHE ДЛЯ ДРОНОВ
        lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
        if lab.shape[2] == 3:  # Проверка количества каналов
            l, a, b = cv2.split(lab)

            # Быстрая CLAHE
            clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(6, 6))
            l_clahe = clahe.apply(l)

            # Легкое усиление верхней части
            height = l_clahe.shape[0]
            if height > 0:
                top_enhancement = np.ones((height, l_clahe.shape[1]), dtype=np.float32)
                y_coords = np.linspace(1.1, 1.0, height)[:, np.newaxis]
                top_enhancement = top_enhancement * y_coords

                l_enhanced = l_clahe.astype(np.float32) * top_enhancement
                l_enhanced = np.clip(l_enhanced, 0, 255).astype(np.uint8)
            else:
                l_enhanced = l_clahe

            lab_enhanced = cv2.merge([l_enhanced, a, b])
            processed = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

        # 4. УМЕРЕННОЕ УСИЛЕНИЕ КОНТРАСТА
        processed = cv2.convertScaleAbs(processed, alpha=1.1, beta=2)

        return processed

    except Exception as e:
        print(f"Error in realtime preprocessing: {e}")
        return image.copy()