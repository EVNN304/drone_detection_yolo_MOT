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


STOP = mp.Event()


def _loop(proc_configs, interval=2.0, max_restarts=3):
    while not STOP.is_set():
        for cfg in proc_configs:
            p = cfg['proc']
            if p is None or p.is_alive(): continue

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
        STOP.wait(interval)  # ждёт сигнал или таймаут

def setup_signals(proc_configs):
    def handler(sig, frame):
        logger.warning(f"🛑 Сигнал {sig}. Корректное завершение...")
        STOP.set()
        for cfg in proc_configs:
            if cfg['proc'] and cfg['proc'].is_alive(): cfg['proc'].terminate()
        sys.exit(0)
    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)

def start_watchdog(procs, interval=2.0, max_restarts=3):
    """
    proc_configs: список словарей вида:
    [{'proc': p, 'target': func, 'args': (), 'daemon': True, 'name': 'MyProc', 'restarts': 0}, ...]
    """

    thr.Thread(target=_loop, args=(procs, interval, max_restarts), daemon=True, name="Watchdog").start()



def set_cam_param(cap, set_w, set_h):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, set_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, set_h)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    #cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    #cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('H','2','6','4'))
    cap.set(cv2.CAP_PROP_FPS, 30)
    #print(cap.get(cv2.CAP_PROP_FPS), cap.getBackendName())

##
def process_video(q_vid:mp.Queue, q_set:mp.Queue, flag_set, path_video, realtime_queue: bool = False):


    cap = cv2.VideoCapture(path_video)
    set_w, set_h = None, None
    if flag_set:
        try:
            set_w, set_h = q_set.get()
            set_cam_param(cap, set_w, set_h)
        except Exception as e:
            logger.error(f"Errr_set_param_cam:{e.args}")

    c = 0

    while True:

        try:
            flag, image = cap.read()
            h, w, _ = image.shape
            if flag:
                c = 0
                if realtime_queue:
                    if q_vid.empty():
                        q_vid.put(image)
                else:
                    q_vid.put(image)

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
    realtime_flag = in_cfg.get("realtime_queue", False)  # читаем из конфига
    selected_tracker, tracker_params = router.get_selected_tracker()

    logger.info(f"📂 Mode: {in_cfg['mode']} | SAHI: {crop_w}x{crop_h} | Timeout: {stop_timeout}s")
    mot_method = cfg["mot"]["method"]
    mot_params = router.get_mot_tracker_params(mot_method)

    import sys
    from yolo_batch_main_mot import *
    from init_Yolo_for_sahi_batches_v2 import *
    from MOT_detect import MOT
    from lib.test.evaluation.yolo_mot_stark import Tracker
    from NanoTrack.bin.NanoTrack_infer import NanoTrackWrapper
    #from VLM_Qwen_class import *                   ### new

    # в path_video можно прописать rtsp поток камеры, либо пть к видео, либо номер usb камеры 0, 1, 2 и etc   ### rtsp://192.168.75.179:8554/operator/h264/tv
    #path_video = f"/home/usr/PycharmProjects/yolo_proj/ultralytics/final_weights_train/ground_to_air/best_yolo11x_288x288_batch_64.pt"
    #path_video = f"rtsp://192.168.75.175:8554/operator/h264/ir_w"
    #path_video = f"/home/usr/Загрузки/rook.mp4"
    #path_video = f"rtspsrc location=rtsp://192.168.75.175:8554/operator/h264/tv latency=0 ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! appsink drop=true max-buffers=1"
    #path_video = f"/home/usr/my_video-2.mkv"
    #path_video = f"/home/usr/Рабочий стол/video/DJI_0571.MP4"
    #path_video = f"/home/usr/PycharmProjects/yolo_proj/ultralytics/videos/ground_to_air/2_n.avi"
    #path_video = f"/home/usr/Видео/stock-footage-drone-shot-following-a-flock-of-birds-as-they-fly-together-on-a-clear-sunny-day.webm"
    #path_video = 0
    #path_video = f"/home/usr/Загрузки/Telegram Desktop/IMG_0958.MP4"
    #path_video = f"/home/usr/Загрузки/first_drone_by_rook.mp4"
    #path_video = f"/home/usr/Видео/stock-footage-swarm-of-drones-flying-at-dusk-a-group-of-drones-hovering-in-the-sky-at-dusk-showcasing-advanced.webm"
    #path_video = f"/home/usr/Видео/stock-footage-military-drone-flying-in-sunset-sky-army-drone-landing-on-military-base-uav-in-sunrise-clouds.webm"
    #path_video = f"rtsp://192.168.3.4:8554/raw/tv_w"
    #path_video = f"/home/usr/Документы/SHILKA/cameras_1803_2/2_TV.mp4"
    procs = []

    renderer_video = BlockRenderer(render_cfg.get("video_stream", {}))


    q_video = mp.Queue(maxsize=1)
    q_set = mp.Queue(maxsize=1)
    q_to_mot = mp.Queue(maxsize=1)
    q_frames, q_in, q_out = mp.Queue(maxsize=1), mp.Queue(maxsize=1), mp.Queue(maxsize=1)
    q_send_names = mp.Queue(maxsize=1)
    q_to_stark = mp.Queue(maxsize=1)

    q_to_qwen = mp.Queue(maxsize=1)    ### new

                                                              ## /home/usr/Рабочий стол/weights_yolo26/drone_iter_3_m/train23/weights/best_yolo26m_ground2air_288x288_btch64.pt
                                                                        # /home/usr/Рабочий стол/weights_yolo26/drone_iter_3_m/train23/weights/best.pt
    cl = Yolo_batches(q_frames, q_in, q_out, q_send_names, q_to_mot)   # /home/usr/PycharmProjects/yolo_proj/ultralytics/runs/detect/train20/weights/best.pt    /home/usr/Рабочий стол/weights_yolo26/drone_iter_2/train22/weights/best.pt
    cl.set_nms_type(cfg["model"]["nms_type"])
    router.apply_to_yolo(cl)                                           #### /home/usr/PycharmProjects/yolo_proj/ultralytics/final_weights_train/ground_to_air/best_yolo11x_288x288_batch_64.pt
    logger.info(f"SIZE_LAYERS: {cl.get_size_inp_layers()}")
    prc_main = cl.process_start()
    procs.append({'proc': prc_main, 'target': cl.main_func, 'args': (), 'daemon': True, 'name': "YOLO_Batch", 'restarts': 0})
    names_classes = q_send_names.get()

    mot_start_proc = MOT(q_to_mot, q_to_stark, q_to_qwen, render_cfg=render_cfg.get("mot"), class_names=names_classes, tracker_params=mot_params)
    router.apply_to_mot(mot_start_proc)
    mot_prc = mot_start_proc.run_process()
    procs.append({'proc': mot_prc, 'target': mot_start_proc.main_worker, 'args': (), 'daemon': True, 'name': "MOT_Tracker", 'restarts': 0})

    if selected_tracker == "stark_s":

        tracker = Tracker(
            q_to_stark,
            selected_tracker,
            tracker_params.get("mode", "baseline"),
            "video",
            render_cfg=render_cfg.get("stark"),
            class_names=names_classes
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
            snapshot_path=tracker_params.get("snapshot_path")
        )
        logger.info(f"🎯 Using tracker: NanoTrack (config={tracker_params.get('config_path')})")

    else:
        raise ValueError(f"Unknown tracker selected: {selected_tracker}")
    stark_prc = tracker.run_process()

    procs.append({'proc': stark_prc, 'target': tracker.run_video,  'args': (), 'daemon': True,  'name': f"{selected_tracker}_Tracker", 'restarts': 0})
    logger.info(f"Classes_load_model: {names_classes}")


    neuro_start_proc = Yolo_inits_batch(q_out, names_files=rec_cfg.get("names_files", "default_"), saved_mode=rec_cfg.get("saved_mode"), name_folder=rec_cfg.get("name_folder", "./saves"), classes_names=names_classes, render_cfg=render_cfg.get("neuro"))
    neuro_start_proc.set_size_cut_w(rec_cfg.get("size_cut_w"))
    neuro_start_proc.set_size_cut_h(rec_cfg.get("size_cut_h"))
    neuro_start_proc.set_min_conf(rec_cfg.get("min_conf"))
    router.apply_to_recording(neuro_start_proc)

    neuro_data_collector = neuro_start_proc.run_nets()
    procs.append({'proc': neuro_data_collector, 'target': neuro_start_proc.collect_n_cast_neuro, 'args': (q_out, neuro_start_proc.saved_mode), 'daemon': True, 'name': "Neuro_Collector", 'restarts': 0})

    #prc_vlm = VLM_Qwen(q_to_qwen)    ### new
    #prc_vlm.run_process()               #### new




    if flag_video:

        proc_vid = mp.Process(target=process_video, args=(q_video, q_set, cap_flag_set, path_video, realtime_flag), daemon=False, name="VideoCapture")
        proc_vid.start()
        procs.append({'proc': proc_vid, 'target': process_video,
                      'args': (q_video, q_set, cap_flag_set, path_video), 'daemon': False, 'name': "VideoCapture",
                      'restarts': 0})

        if cap_flag_set:
            q_set.put([set_w, set_h])
        time.sleep(4)
        if not q_video.empty():
            img = q_video.get()
            h_p, w_p, _ = img.shape
        else:
            flg, img = False, None
            logger.critical("❌ Нет кадров от источника.")
            STOP.set()
            exit(1)


        fps_timer = Timer()
        fps_timer.start()

        lst_cord = calc_scan_areas([0, 0, w_p, h_p], window_w_h=(crop_w, crop_h), overlay=(overlay_w, overlay_h))
        prev_frames_for_motion_est = []  # Для оценки движения (опционально)
        max_prev_frames = 1
        setup_signals(procs)
        start_watchdog(procs, interval=2.0,  max_restarts=max_retries)
        kernel = np.ones((4, 4), dtype=np.float32) / 17

        while True:
            if not q_video.empty():

                image = q_video.get()
                #image = cv2.flip(image, -1)
                image[900:1080, 20:350, :] = 0
                list_image, list_cropp_cord = [], []
                for i, k in enumerate(lst_cord):
                    #fragment_processed = cv.filter2D(image[k[1]:k[3], k[0]:k[2]], -1, kernel)
                    fragment_processed = image[k[1]:k[3], k[0]:k[2]]
                    list_image.append(fragment_processed)
                    list_cropp_cord.append([k[0], k[1]])

                list_image.append(image)
                list_cropp_cord.append([0, 0])
                if q_in.empty():
                    q_in.put([list_image, list_cropp_cord])

                elapsed = fps_timer.stop()
                fps_timer.start()
                logger.info(f'capture fps: {1 / elapsed if elapsed != 0 else 1}')
                renderer_video.show_video_stream(
                    image,
                    status_text=f"Mode: {in_cfg['mode']} | SAHI crops: {len(list_cropp_cord)}",
                    fps=1 / elapsed if elapsed != 0 else 1
                )


    else:
        lst = os.listdir(pth)

        img = cv2.imread(pth+lst[0])
        h_p, w_p, _ = img.shape

        fps_timer = Timer()
        fps_timer.start()

        lst_cord = calc_scan_areas([0, 0, w_p, h_p], window_w_h=(crop_w, crop_h), overlay=(overlay_w, overlay_h))
        for i, k in enumerate(lst):


            image = cv2.imread(pth+k)
            list_image, list_cropp_cord = [], []
            for i, k in enumerate(lst_cord):
                list_image.append(image[k[1]:k[3], k[0]:k[2]])
                list_cropp_cord.append([k[0], k[1]])
            list_image.append(image)
            list_cropp_cord.append([0, 0])
            if q_in.empty():
                q_in.put([list_image, list_cropp_cord])



            elapsed = fps_timer.stop()
            fps_timer.start()
            logger.info(f'capture fps: {1 / elapsed if elapsed != 0 else 1}')
            renderer_video.show_video_stream(
                image,
                status_text=f"Mode: {in_cfg['mode']} | SAHI crops: {len(list_cropp_cord)}",
                fps=1 / elapsed if elapsed != 0 else 1
            )