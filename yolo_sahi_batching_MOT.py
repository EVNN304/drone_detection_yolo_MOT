import sys
sys.path.insert(0, '/home/usr/PycharmProjects/STARK/Stark')
sys.path.insert(0, '/home/usr/PycharmProjects/Siam_tracker/SiamTrackers')
from loggers import *
logger = logging.getLogger(__name__)
import signal
import yaml
import multiprocessing as mp
import threading as thr
from preprocess_denoice import *
from config_router import ConfigRouter
from shm_utils import (SharedFrameBuffer, SharedFrameWriter, SharedFrameReader,
                       SharedCropsBuffer, SharedCropsWriter, SharedSimpleFrameBuffer)

STOP = mp.Event()


def _loop(proc_configs, interval=2.0, max_restarts=3):
    while not STOP.is_set():
        for cfg in proc_configs:
            p = cfg['proc']
            if p is None or p.is_alive():
                continue
            logger.warning(f"⚠️ Процесс '{cfg['name']}' упал (exitcode={p.exitcode})")
            try:
                p.join(timeout=1)
            except:
                pass
            if cfg['restarts'] < max_restarts:
                cfg['restarts'] += 1
                logger.info(f"🔄 Перезапуск '{cfg['name']}' ({cfg['restarts']}/{max_restarts})")
                cfg['proc'] = mp.Process(target=cfg['target'], args=cfg['args'],
                                         daemon=cfg['daemon'], name=cfg['name'])
                cfg['proc'].start()
            else:
                logger.critical(f"💀 Лимит рестартов для '{cfg['name']}'. Остановка пайплайна.")
                STOP.set()
        STOP.wait(interval)


def setup_signals(proc_configs):
    def handler(sig, frame):
        logger.warning(f"🛑 Сигнал {sig}. Корректное завершение...")
        STOP.set()
        for cfg in proc_configs:
            if cfg['proc'] and cfg['proc'].is_alive():
                cfg['proc'].terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)


def start_watchdog(procs, interval=2.0, max_restarts=3):
    thr.Thread(target=_loop, args=(procs, interval, max_restarts), daemon=True, name="Watchdog").start()


def set_cam_param(cap, set_w, set_h):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, set_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, set_h)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap.set(cv2.CAP_PROP_FPS, 30)


def process_video(shm_name_0, shm_name_1, free_queue, ready_queue, q_set: mp.Queue,
                  flag_set, path_video, frame_h, frame_w, realtime_queue: bool = False):
    """Чтение видео/камеры и запись в Shared Memory"""
    cap = cv2.VideoCapture(path_video)
    set_w, set_h = None, None
    if flag_set:
        try:
            set_w, set_h = q_set.get()
            set_cam_param(cap, set_w, set_h)
        except Exception as e:
            logger.error(f"Errr_set_param_cam:{e.args}")

    writer = SharedFrameWriter(frame_h, frame_w, shm_name_0, shm_name_1, free_queue, ready_queue)
    c = 0
    while True:
        try:
            flag, image = cap.read()
            h, w, _ = image.shape
            if flag:
                c = 0
                if realtime_queue:
                    if not free_queue.empty():
                        writer.write_frame(image)
                else:
                    writer.write_frame(image)
        except Exception as e:
            logger.error(f"Errr_connect_cam_or_video: {e.args}")
            cap.release()
            cv2.destroyAllWindows()
            time.sleep(3)
            cap = cv2.VideoCapture(path_video)
            if flag_set and cap.isOpened():
                set_cam_param(cap, set_w, set_h)
            c += 1
            logger.info(f"Reconn_cam/video: {cap.isOpened()}, count try connect: {c}")
            continue


if __name__ == '__main__':
    setup_logging()
    mp.set_start_method('spawn', force=True)

    router = ConfigRouter("pipeline_config.yaml")
    cfg = router.cfg

    in_cfg = cfg["input"]
    sahi_cfg = cfg["sahi"]
    pipeline_cfg = cfg["pipeline"]

    crop_w, crop_h = sahi_cfg["crop_w"], sahi_cfg["crop_h"]
    overlay_w, overlay_h = sahi_cfg["overlay_w"], sahi_cfg["overlay_h"]
    set_w, set_h = in_cfg["resolution"]
    cap_flag_set = in_cfg.get("cap_flag_set", False)
    path_video = in_cfg["path"]
    pth = in_cfg.get("folder_path", "")
    flag_video = in_cfg["mode"] in ("video", "rtsp")
    stop_timeout = pipeline_cfg["stop_timeout_sec"]
    max_retries = pipeline_cfg["max_queue_retries"]
    render_cfg = router.cfg.get("render", {})
    rec_cfg = cfg.get("recording", {})
    realtime_flag = in_cfg.get("realtime_queue", False)
    selected_tracker, tracker_params = router.get_selected_tracker()

    # === VLM КОНФИГ ===
    vlm_cfg = router.get_vlm_cfg()
    vlm_enabled = router.is_vlm_enabled()
    logger.info(f"🤖 VLM: {'ВКЛЮЧЕН' if vlm_enabled else 'ВЫКЛЮЧЕН'}")


    logger.info(f"📂 Mode: {in_cfg['mode']} | SAHI: {crop_w}x{crop_h} | Timeout: {stop_timeout}s")
    mot_method = cfg["mot"]["method"]
    mot_params = router.get_mot_tracker_params(mot_method)

    from yolo_batch_main_mot import *
    from init_Yolo_for_sahi_batches_v2 import *
    from MOT_detect import MOT
    from lib.test.evaluation.yolo_mot_stark_shm import Tracker
    from NanoTrack.bin.NanoTrack_infer_shm import NanoTrackWrapper
    from VLM_Qwen_class import VLM_Qwen

    procs = []
    renderer_video = BlockRenderer(render_cfg.get("video_stream", {}))

    # === ОПРЕДЕЛЯЕМ РАЗМЕР КАДРА ===
    if flag_video:
        cap = cv2.VideoCapture(path_video)
        if cap_flag_set:
            set_cam_param(cap, set_w, set_h)
        ret, img = cap.read()
        cap.release()
        if not ret:
            logger.critical("❌ Нет кадров от источника.")
            exit(1)
        h_p, w_p, _ = img.shape
    else:
        lst = os.listdir(pth)
        img = cv2.imread(pth + lst[0])
        h_p, w_p, _ = img.shape

    FRAME_H, FRAME_W = h_p, w_p
    logger.info(f"📐 Размер кадра: {FRAME_W}x{FRAME_H}")

    # === СОЗДАЕМ SHARED MEMORY БУФЕРЫ ===
    # 1. VideoCapture → Главный процесс
    shm_video_capture = SharedFrameBuffer(FRAME_H, FRAME_W, name_prefix="shm_video_capture")
    main_to_yolo_free_queue = mp.Queue(maxsize=2)
    main_to_yolo_free_queue.put(1)
    main_to_yolo_free_queue.put(1)
    main_to_yolo_ready_queue = mp.Queue(maxsize=2)

    # 2. Главный процесс → YOLO (кропы)
    num_crops = len(calc_scan_areas([0, 0, w_p, h_p], window_w_h=(crop_w, crop_h), overlay=(overlay_w, overlay_h)))
    shm_crops_to_yolo = SharedCropsBuffer(num_crops, crop_h, crop_w, name_prefix="shm_crops_to_yolo")

    # 3. YOLO → MOT
    shm_yolo_to_mot = SharedFrameBuffer(FRAME_H, FRAME_W, name_prefix="shm_yolo_to_mot")
    yolo_to_mot_free_queue = mp.Queue(maxsize=2)
    yolo_to_mot_free_queue.put(1)
    yolo_to_mot_free_queue.put(1)
    yolo_to_mot_ready_queue = mp.Queue(maxsize=2)

    # 4. YOLO → Neuro collector
    shm_yolo_to_neuro = SharedSimpleFrameBuffer(FRAME_H, FRAME_W, name_prefix="shm_yolo_to_neuro")

    # 5. MOT → VLM
    if vlm_enabled:
        shm_mot_to_vlm = SharedFrameBuffer(FRAME_H, FRAME_W, name_prefix="shm_mot_to_vlm")
        mot_to_vlm_free_queue = mp.Queue(maxsize=2)
        mot_to_vlm_free_queue.put(1)
        mot_to_vlm_free_queue.put(1)
        mot_to_vlm_ready_queue = mp.Queue(maxsize=2)
    else:
        shm_mot_to_vlm = None
        q_to_qwen = None
    # 6. MOT → SOT трекер (STARK/NanoTrack) ← НОВОЕ
    shm_mot_to_sot = SharedFrameBuffer(FRAME_H, FRAME_W, name_prefix="shm_mot_to_sot")
    mot_to_sot_free_queue = mp.Queue(maxsize=2)
    mot_to_sot_free_queue.put(1)
    mot_to_sot_free_queue.put(1)
    mot_to_sot_ready_queue = mp.Queue(maxsize=2)

    # === ОЧЕРЕДИ (только для маленьких данных) ===
    q_set = mp.Queue(maxsize=1)
    q_to_mot = mp.Queue(maxsize=1)
    q_out = mp.Queue(maxsize=1)
    q_send_names = mp.Queue(maxsize=1)
    q_to_stark = mp.Queue(maxsize=1)  # ← увеличен для SHM
    q_to_qwen = mp.Queue(maxsize=1)
    q_coords = mp.Queue(maxsize=1)

    # === YOLO ===
    cl = Yolo_batches(q_to_mot, q_out, q_send_names, q_coords,
                      shm_main_to_yolo_name_0=shm_video_capture.name_0,
                      shm_main_to_yolo_name_1=shm_video_capture.name_1,
                      shm_main_to_yolo_free_queue=main_to_yolo_free_queue,
                      shm_main_to_yolo_ready_queue=main_to_yolo_ready_queue,
                      shm_crops_name=shm_crops_to_yolo.name,
                      num_crops=num_crops,
                      crop_h=crop_h, crop_w=crop_w,
                      shm_mot_name_0=shm_yolo_to_mot.name_0,
                      shm_mot_name_1=shm_yolo_to_mot.name_1,
                      shm_mot_free_queue=yolo_to_mot_free_queue,
                      shm_mot_ready_queue=yolo_to_mot_ready_queue,
                      shm_neuro_name=shm_yolo_to_neuro.name,
                      frame_h=FRAME_H, frame_w=FRAME_W)
    cl.set_nms_type(cfg["model"]["nms_type"])
    router.apply_to_yolo(cl)
    logger.info(f"SIZE_LAYERS: {cl.get_size_inp_layers()}")
    prc_main = cl.process_start()
    procs.append({'proc': prc_main, 'target': cl.main_func, 'args': (), 'daemon': True, 'name': "YOLO_Batch", 'restarts': 0})
    names_classes = q_send_names.get()

    # === MOT ===
    mot_start_proc = MOT(q_to_mot, q_to_stark, q_to_qwen,
                         render_cfg=render_cfg.get("mot"),
                         class_names=names_classes,
                         tracker_params=mot_params,
                         shm_yolo_to_mot_name_0=shm_yolo_to_mot.name_0,
                         shm_yolo_to_mot_name_1=shm_yolo_to_mot.name_1,
                         shm_yolo_to_mot_free_queue=yolo_to_mot_free_queue,
                         shm_yolo_to_mot_ready_queue=yolo_to_mot_ready_queue,
                         # VLM параметры (None если выключен)
                         shm_mot_to_vlm_name_0=shm_mot_to_vlm.name_0 if shm_mot_to_vlm else None,
                         shm_mot_to_vlm_name_1=shm_mot_to_vlm.name_1 if shm_mot_to_vlm else None,
                         shm_mot_to_vlm_free_queue=mot_to_vlm_free_queue if vlm_enabled else None,
                         shm_mot_to_vlm_ready_queue=mot_to_vlm_ready_queue if vlm_enabled else None,
                         shm_mot_to_sot_name_0=shm_mot_to_sot.name_0,
                         shm_mot_to_sot_name_1=shm_mot_to_sot.name_1,
                         shm_mot_to_sot_free_queue=mot_to_sot_free_queue,
                         shm_mot_to_sot_ready_queue=mot_to_sot_ready_queue,
                         frame_h=FRAME_H, frame_w=FRAME_W,
                         vlm_enabled=vlm_enabled,  # ← НОВЫЙ ПАРАМЕТР
                         vlm_send_interval=vlm_cfg.get("send_interval", 30),
                         vlm_max_objects=vlm_cfg.get("max_objects", 3))
    router.apply_to_mot(mot_start_proc)
    mot_prc = mot_start_proc.run_process()
    procs.append({'proc': mot_prc, 'target': mot_start_proc.main_worker, 'args': (), 'daemon': True, 'name': "MOT_Tracker", 'restarts': 0})

    # === SOT ТРЕКЕР ===
    if selected_tracker == "stark_s":
        tracker = Tracker(
            q_to_stark,
            selected_tracker,
            tracker_params.get("mode", "baseline"),
            "video",
            render_cfg=render_cfg.get("stark"),
            class_names=names_classes,
            shm_mot_to_sot_name_0=shm_mot_to_sot.name_0,
            shm_mot_to_sot_name_1=shm_mot_to_sot.name_1,
            shm_mot_to_sot_free_queue=mot_to_sot_free_queue,
            shm_mot_to_sot_ready_queue=mot_to_sot_ready_queue,
            frame_h=FRAME_H, frame_w=FRAME_W
        )
        logger.info(f"🎯 Using tracker: STARK (mode={tracker_params.get('mode')})")
    elif selected_tracker == "nanotrack":
        tracker = NanoTrackWrapper(
            q_frame=q_to_stark,
            name=selected_tracker,
            parameter_name=tracker_params.get("mode", "baseline"),
            dataset_name="video",
            render_cfg=render_cfg.get("stark"),
            class_names=names_classes,
            config_path=tracker_params.get("config_path"),
            snapshot_path=tracker_params.get("snapshot_path"),
            shm_mot_to_sot_name_0=shm_mot_to_sot.name_0,
            shm_mot_to_sot_name_1=shm_mot_to_sot.name_1,
            shm_mot_to_sot_free_queue=mot_to_sot_free_queue,
            shm_mot_to_sot_ready_queue=mot_to_sot_ready_queue,
            frame_h=FRAME_H, frame_w=FRAME_W
        )
        logger.info(f"🎯 Using tracker: NanoTrack (config={tracker_params.get('config_path')})")
    else:
        raise ValueError(f"Unknown tracker selected: {selected_tracker}")

    stark_prc = tracker.run_process()
    procs.append({'proc': stark_prc, 'target': tracker.run_video, 'args': (), 'daemon': True,
                  'name': f"{selected_tracker}_Tracker", 'restarts': 0})
    logger.info(f"Classes_load_model: {names_classes}")

    # === NEURO COLLECTOR ===
    neuro_start_proc = Yolo_inits_batch(q_out,
                                        names_files=rec_cfg.get("names_files", "default_"),
                                        saved_mode=rec_cfg.get("saved_mode"),
                                        name_folder=rec_cfg.get("name_folder", "./saves"),
                                        classes_names=names_classes,
                                        render_cfg=render_cfg.get("neuro"),
                                        shm_neuro_name=shm_yolo_to_neuro.name,
                                        frame_h=FRAME_H, frame_w=FRAME_W)
    neuro_start_proc.set_size_cut_w(rec_cfg.get("size_cut_w"))
    neuro_start_proc.set_size_cut_h(rec_cfg.get("size_cut_h"))
    neuro_start_proc.set_min_conf(rec_cfg.get("min_conf"))
    router.apply_to_recording(neuro_start_proc)
    neuro_data_collector = neuro_start_proc.run_nets()
    procs.append({'proc': neuro_data_collector, 'target': neuro_start_proc.collect_n_cast_neuro,
                  'args': (q_out, neuro_start_proc.saved_mode), 'daemon': True, 'name': "Neuro_Collector", 'restarts': 0})

    # === VLM ===
    if vlm_enabled:
        from VLM_Qwen_class import VLM_Qwen

        prc_vlm = VLM_Qwen(q_to_qwen,
                           shm_mot_to_vlm_name_0=shm_mot_to_vlm.name_0,
                           shm_mot_to_vlm_name_1=shm_mot_to_vlm.name_1,
                           shm_mot_to_vlm_free_queue=mot_to_vlm_free_queue,
                           shm_mot_to_vlm_ready_queue=mot_to_vlm_ready_queue,
                           device=vlm_cfg.get("device", "cuda:1"),
                           model_id=vlm_cfg.get("model_id", "Qwen/Qwen2.5-VL-3B-Instruct"),
                           crop_size=vlm_cfg.get("crop_size", 288))
        vlm_process = prc_vlm.run_process()
        procs.append({'proc': vlm_process, 'target': prc_vlm.main_func, 'args': (), 'daemon': True,
                      'name': "VLM_Qwen", 'restarts': 0})
        logger.info("✅ VLM процесс запущен")
    else:
        logger.info("⏭️ VLM отключен!")

    # === VIDEO CAPTURE ===
    if flag_video:
        proc_vid = mp.Process(target=process_video,
                              args=(shm_video_capture.name_0, shm_video_capture.name_1,
                                    main_to_yolo_free_queue, main_to_yolo_ready_queue,
                                    q_set, cap_flag_set, path_video, FRAME_H, FRAME_W, realtime_flag),
                              daemon=False, name="VideoCapture")
        proc_vid.start()
        procs.append({'proc': proc_vid, 'target': process_video,
                      'args': (shm_video_capture.name_0, shm_video_capture.name_1,
                               main_to_yolo_free_queue, main_to_yolo_ready_queue,
                               q_set, cap_flag_set, path_video, FRAME_H, FRAME_W),
                      'daemon': False, 'name': "VideoCapture", 'restarts': 0})

        if cap_flag_set:
            q_set.put([set_w, set_h])
        time.sleep(4)

        # Создаем reader для чтения кадра из SHM в главном процессе
        video_reader = SharedFrameReader(FRAME_H, FRAME_W,
                                         shm_video_capture.name_0, shm_video_capture.name_1,
                                         main_to_yolo_free_queue, main_to_yolo_ready_queue)

        fps_timer = Timer()
        fps_timer.start()
        lst_cord = calc_scan_areas([0, 0, w_p, h_p], window_w_h=(crop_w, crop_h), overlay=(overlay_w, overlay_h))
        setup_signals(procs)
        start_watchdog(procs, interval=2.0, max_restarts=max_retries)

        while True:
            # Читаем оригинальный кадр из SHM (вместо q_video.get())
            image = video_reader.read_frame()
            image[900:1080, 20:350, :] = 0

            # Нарезаем кропы и записываем в SHM
            crops_list = []
            coords_list = []
            for i, k in enumerate(lst_cord):
                crop = image[k[1]:k[3], k[0]:k[2]]
                crops_list.append(crop)
                coords_list.append([k[0], k[1]])

            # Записываем кропы в SHM
            crops_writer = SharedCropsWriter(num_crops, crop_h, crop_w, shm_crops_to_yolo.name)
            crops_writer.write_crops(crops_list)

            # Отправляем только координаты через очередь
            if q_coords.empty():
                q_coords.put(coords_list)

            elapsed = fps_timer.stop()
            fps_timer.start()
            logger.info(f'capture fps: {1 / elapsed if elapsed != 0 else 1}')
            renderer_video.show_video_stream(
                image,
                status_text=f"Mode: {in_cfg['mode']} | SAHI crops: {len(lst_cord)}",
                fps=1 / elapsed if elapsed != 0 else 1
            )

    else:
        lst = os.listdir(pth)
        img = cv2.imread(pth + lst[0])
        h_p, w_p, _ = img.shape

        fps_timer = Timer()
        fps_timer.start()
        lst_cord = calc_scan_areas([0, 0, w_p, h_p], window_w_h=(crop_w, crop_h), overlay=(overlay_w, overlay_h))

        for i, k in enumerate(lst):
            image = cv2.imread(pth + k)
            crops_list = []
            coords_list = []
            for i, k in enumerate(lst_cord):
                crop = image[k[1]:k[3], k[0]:k[2]]
                crops_list.append(crop)
                coords_list.append([k[0], k[1]])

            crops_writer = SharedCropsWriter(num_crops, crop_h, crop_w, shm_crops_to_yolo.name)
            crops_writer.write_crops(crops_list)

            if q_coords.empty():
                q_coords.put(coords_list)

            elapsed = fps_timer.stop()
            fps_timer.start()
            logger.info(f'capture fps: {1 / elapsed if elapsed != 0 else 1}')
            renderer_video.show_video_stream(
                image,
                status_text=f"Mode: {in_cfg['mode']} | SAHI crops: {len(lst_cord)}",
                fps=1 / elapsed if elapsed != 0 else 1
            )

    # === ОСВОБОЖДЕНИЕ РЕСУРСОВ ===
    shm_video_capture.close()
    shm_crops_to_yolo.close()
    shm_yolo_to_mot.close()
    shm_yolo_to_neuro.close()
    if shm_mot_to_vlm:
        shm_mot_to_vlm.close()
    shm_mot_to_sot.close()
    logger.info("✅ Завершение работы")