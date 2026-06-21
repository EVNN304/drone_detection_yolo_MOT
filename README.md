# 🚁 Aerial Object Detection & Tracking System

**Комплексная система детекции и сопровождения воздушных объектов в реальном времени.**

Проект объединяет **YOLOv11x-26x** для детекции, **SAHI-batching** для работы с мелкими объектами, **HybridSort** для многообъектного трекинга, **STARK/NanoTrack** для точного сопровождения одиночных целей и высокопроизводительную архитектуру на базе **Shared Memory** для передачи кадров между процессами.

---

## 🌟 Основные возможности

### Детекция
- **Детекция объектов**: Дроны, птицы, самолёты
- **Модель детекции**: Ultralytics YOLOv11x-26x (поддержка `.pt` и `.engine` TensorRT)
- **SAHI Batching**: Автоматическое разбиение кадра на перекрывающиеся кропы для улучшения детекции мелких объектов
- **NMS Алгоритмы**: 8 стратегий подавления дубликатов (Classic, Soft-NMS, WBF, DIoU-NMS, Adaptive, Cluster, NMM, Greedy NMM)

### Трекинг
- **MOT (Multi-Object Tracking)**: Поддержка 7+ алгоритмов через BoxMOT (HybridSort, ByteTrack, BotSort, StrongSort, DeepOcSort, BoostTrack, OcSort)
- **SOT (Single Object Tracking)**: STARK и NanoTrack для точного сопровождения выбранной цели
- **Custom ReID**: Интеграция собственной модели ReID, обученной на воздушных целях
- **PTZ логика**: Режим lock-on с автоматическим переключением между целями при потере трека

### Производительность
- **Мультипроцессинг**: 6 изолированных процессов (VideoCapture, YOLO, MOT, SOT, Neuro, Watchdog)
- **Watchdog**: Автоматический перезапуск упавших процессов

---
    
🏗 Архитектура системы
![Screenshot from 2026-06-22 02-42-09](examples/Screenshot%from%2026-06-22%02-42-09.png)

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

    Для работы STARK и NanoTrack их нужно установить отдельно:
      STARK (рекомендуется)
        cd /path/to/your/projects
        git clone https://github.com/researchmm/Stark.git
        cd Stark
        pip install -r requirements.txt
      NanoTrack (легковесная альтернатива)
        cd /path/to/your/projects
        git clone https://github.com/microsoft/SiamTracker.git
        cd SiamTrackers
        pip install -r requirements.txt

  После установки положите скрипты-обёртки yolo_mot_stark.py и NanoTrack_infer.py из этого проекта в соответствующие директории трекеров, либо адаптируйте пути в pipeline_config.yaml.
    
Загрузка моделей:
Вам необходимо скачать веса для детектора и ReID модели.

    YOLO Detector: Положите файл .pt или .engine в папку weights/.
        Пример: best_yolo11x_288x288_batch_64.pt
    ReID Model: Положите веса в папку weights/reid/.
        Пример: osnet_x0_25_reconverted.onnx или .pt

    STARK/NanoTrack: Веса скачиваются согласно инструкциям в их репозиториях

    
⚙️ Конфигурация

    Вся настройка системы осуществляется через pipeline_config.yaml — единый файл конфигурации.
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


    Выбор SOT трекера
      В секции tracker:
        Установите enabled: true для нужного трекера
        Приоритет задаётся в priority_order

▶️ Запуск

    # 1. Отредактируйте pipeline_config.yaml под свои пути и настройки
         nano pipeline_config.yaml

    # 2. Запустите систему
         python yolo_sahi_batching_MOT.py

         
    Система автоматически:

        Сконвертирует YOLO веса в TensorRT (если включено)
        Запустит все процессы
        Подключит выбранный SOT трекер
        Запустит watchdog для мониторинга процессов



    
⚠️ Важные примечания

    Зависимости additional_scripts: В коде присутствуют импорты из модулей additional_scripts.save_modules и additional_scripts.main_src.protocols. Убедитесь, что эти папки присутствуют в проекте или адаптируйте импорты под свою структуру.
    Пути к файлам: В коде жестко прописаны пути (например, /home/usr/...). Обязательно замените их на актуальные для вашей системы перед запуском.
    Производительность: Для достижения высокого FPS рекомендуется использовать TensorRT (.engine) для YOLO и включать half_flag=True (FP16), если ваша GPU поддерживает это.
    SAHI Logic: Функция calc_scan_areas используется для генерации координат кропов. Убедитесь, что она импортирована корректно.

🤝 Contributing
Проект открыт для доработок. Если вы нашли ошибку или хотите добавить новый алгоритм трекинга, пожалуйста, создайте Pull Request.


🔮 Реализованные модули:

   ✅ STARK Single Object Tracking
   Интегрирован трекер STARK (Spatial-Temporal Anchor-free Representation Learning) для сопровождения одиночных целей:

        Высокая точность при трекинге быстро маневрирующих объектов
        Устойчивость к окклюзиям и выходам за кадр
        Режим "lock-on" для выбранного объекта
        Валидация трека через IoU с MOT и EMA-сглаживание
        📥 Установка: https://github.com/researchmm/Stark
        
  ✅ NanoTrack Single Object Tracking
  Интегрирован легковесный трекер NanoTrack для ресурсовограниченных систем:

        Минимальные требования к GPU
        Высокая скорость (>100 FPS)
        Альтернатива STARK для embedded-устройств
       📥 Установка: https://github.com/microsoft/SiamTracker



   
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
