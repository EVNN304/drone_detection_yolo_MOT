🚁 Aerial Object Detection & Tracking System (YOLOv11x-26x + HybridSort)
Система детекции и сопровождения воздушных объектов (дроны, птицы, самолеты) в реальном времени. Проект использует архитектуру на основе YOLOv11x-26x для детекции, технику SAHI-batching (разбиение кадра на кропы) для повышения точности на мелких объектах и HybridSort для многообъектного трекинга (MOT) с кастомной ReID моделью, обученной на воздушных целях.
🌟 Основные возможности

    Детекция объектов: Дроны, птицы, самолеты.
    Модель детекции: Ultralytics YOLOv11x-26x (поддержка .pt и .engine).
    SAHI Batching: Автоматическое разбиение кадра на перекрывающиеся области для улучшения детекции мелких объектов.
    MOT (Multi-Object Tracking): Поддержка 7+ алгоритмов трекинга (HybridSort, ByteTrack, BotSort и др.).
    Custom ReID: Интеграция собственной модели ReID для улучшения ассоциации треков воздушных целей.
    Мультипроцессинг: Разделение потоков видеозахвата, инференса нейросети и трекинга для максимального FPS.
    NMS Алгоритмы: Возможность выбора стратегии подавления дубликатов (Classic, Soft-NMS, WBF, DIoU-NMS и др.).
    
🏗 Архитектура системы
Система состоит из трех основных процессов, связанных через очереди multiprocessing.Queue:

    Video Process: Захват видео (RTSP, файл, камера), разбиение кадра на кропы (SAHI logic).
    YOLO Process: Инференс модели на батчах кропов, применение NMS, сборка координат.
    MOT Process: Получение детекций, ассоциация треков (HybridSort), отрисовка результатов.

📋 Требования

    Python: 3.8 - 3.10
    GPU: NVIDIA CUDA (рекомендуется для real-time работы)
    OS: Linux (Ubuntu 20.04/22.04 tested), Windows (требуется адаптация путей)
    
зависимости (requirements.txt)

  torch>=2.0.0
  torchvision>=0.15.0
  ultralytics>=8.0.0
  opencv-python>=4.8.0
  numpy>=1.24.0
  boxmot>=2.0.0
  
🚀 Установка
    Клонирование репозитория:
    git clone <your-repo-url>
    cd <repo-folder>
    
    python3 -m venv venv
    source venv/bin/activate

    pip install -r requirements.txt

Загрузка моделей:
Вам необходимо скачать веса для детектора и ReID модели.

    YOLO Detector: Положите файл .pt или .engine в папку weights/.
        Пример: best_yolo11x_288x288_batch_64.pt
    ReID Model: Положите веса в папку weights/reid/.
        Пример: osnet_x0_25_reconverted.onnx или .pt
⚙️ Конфигурация
    Основной файл запуска: yolo_sahi_batching_MOT.py.

    Выбор алгоритма MOT
В файле MOT_detect.py доступны следующие методы (параметр set_mot_method):

    0: BoostTrack
    1: BotSort
    2: StrongSort
    3: DeepOcSort
    4: ByteTrack
    5: HybridSort (✅ Рекомендуется для воздушных целей)
    6: OcSort

    Выбор алгоритма NMS
    В файле yolo_batch_main_mot.py доступны:

    classic: Classic NMS
    soft: Soft-NMS
    wbf: Weighted Box Fusion
    diou: DIoU-NMS
    adaptive: Adaptive NMS
    cluster: Cluster NMS
    nmm: Non-Maximum Merge
    greedynmm: Greedy NMM


▶️ Запуск

    Откройте yolo_sahi_batching_MOT.py.
    Отредактируйте пути к моделям и видео в блоке if __name__ == '__main__':
      path_video = "rtsp://192.168.75.179:8554/operator/h264/tv" # Или путь к файлу
      cl.set_path_model("/path/to/your/best_yolo11x.pt")
      mot_start_proc.set_path_weights_reid('/path/to/your/reid_model.onnx')
      mot_start_proc.set_mot_method(5) # HybridSort
    Запустите скрипт:
        python yolo_sahi_batching_MOT.py

⚠️ Важные примечания

    Зависимости additional_scripts: В коде присутствуют импорты из модулей additional_scripts.save_modules и additional_scripts.main_src.protocols. Убедитесь, что эти папки присутствуют в проекте или адаптируйте импорты под свою структуру.
    Пути к файлам: В коде жестко прописаны пути (например, /home/usr/...). Обязательно замените их на актуальные для вашей системы перед запуском.
    Производительность: Для достижения высокого FPS рекомендуется использовать TensorRT (.engine) для YOLO и включать half_flag=True (FP16), если ваша GPU поддерживает это.
    SAHI Logic: Функция calc_scan_areas используется для генерации координат кропов. Убедитесь, что она импортирована корректно.

🤝 Contributing
Проект открыт для доработок. Если вы нашли ошибку или хотите добавить новый алгоритм трекинга, пожалуйста, создайте Pull Request.


🔮 Планы развития проекта
Ближайшие обновления (2026)
1. STARK Single Object Tracking 🔥
Планируется интеграция трекера STARK (Spatial-Temporal Anchor-free Representation Learning for Visual Tracking) для сопровождения одиночных целей:

    Преимущества STARK:
        Высокая точность при трекинге быстро маневрирующих объектов
        Устойчивость к окклюзиям и выходам за кадр
        Работает в реальном времени (>60 FPS на GPU)
        Отличная работа с мелкими объектами на больших дистанциях
    Сценарии использования:
        Приоритетное сопровождение конкретной цели (дрона, самолета)
        Режим "lock-on" для выбранного объекта
        Долгосрочное сопровождение целей за пределами поля зрения детектора
        Переключение между целями без потери трека


   
📸 Демонстрация работы
![Drone Detection 00-39-30](examples/Снимок%20экрана%20от%202026-03-02%2000-39-30.png)

![Drone Detection 00-39-38](examples/Снимок%20экрана%20от%202026-03-02%2000-39-38.png)

![Bird Tracking 00-43-14](examples/Снимок%20экрана%20от%202026-03-02%2000-43-14.png)

![Bird Tracking 00-43-22](examples/Снимок%20экрана%20от%202026-03-02%2000-43-22.png)

![Aircraft Detection 00-44-22](examples/Снимок%20экрана%20от%202026-03-02%2000-44-22.png)

![Aircraft Detection 00-44-28](examples/Снимок%20экрана%20от%202026-03-02%2000-44-28.png)

![MOT Tracking 00-45-17](examples/Снимок%20экрана%20от%202026-03-02%2000-45-17.png)

![MOT Tracking 00-45-22](examples/Снимок%20экрана%20от%202026-03-02%2000-45-22.png)

![HybridSort MOT 00-46-04](examples/Снимок%20экрана%20от%202026-03-02%2000-46-04.png)

![HybridSort MOT 00-46-11](examples/Снимок%20экрана%20от%202026-03-02%2000-46-11.png)

![SAHI Batching 00-46-59](examples/Снимок%20экрана%20от%202026-03-02%2000-46-59.png)

![System Overview 00-48-51](examples/Снимок%20экрана%20от%202026-03-02%2000-48-51.png)
