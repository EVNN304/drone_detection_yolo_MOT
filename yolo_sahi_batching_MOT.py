
import multiprocessing as mp
from preprocess_denoice import *




def set_cam_param(cap, set_w, set_h):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, set_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, set_h)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    #cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    #cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('H','2','6','4'))
    cap.set(cv2.CAP_PROP_FPS, 30)
    #print(cap.get(cv2.CAP_PROP_FPS), cap.getBackendName())

##
def process_video(q_vid:mp.Queue, q_set:mp.Queue, flag_set, path_video):


    cap = cv2.VideoCapture(path_video)
    set_w, set_h = None, None
    if flag_set:
        try:
            set_w, set_h = q_set.get()
            set_cam_param(cap, set_w, set_h)
        except Exception as e:
            print(f"Errr_set_param_cam:{e.args}")

    c = 0

    while True:
        try:
            flag, image = cap.read()
            if flag:
                c = 0
                if q_vid.empty():
                    q_vid.put(image)
        except Exception as e:
            print(f"Errr_connect_cam_or_video: {e.args}")
            cap.release()
            cv2.destroyAllWindows()
            time.sleep(3)

            cap = cv2.VideoCapture(path_video)
            if flag_set and cap.isOpened():
                set_cam_param(cap, set_w, set_h)
            c += 1
            print(f"Reconn_cam/video: {cap.isOpened()}, count try connect: {c}")
            continue

if __name__ == '__main__':

    mp.set_start_method('spawn', force=True)

    from yolo_batch_main_mot import *
    from init_Yolo_for_sahi_batches_v2 import *
    from MOT_detect import MOT
    crop_w, crop_h = 288, 288
    overlay_w, overlay_h = 0.1, 0.1
    set_w, set_h = 1920, 1200
    # в path_video можно прописать rtsp поток камеры, либо пть к видео, либо номер usb камеры 0, 1, 2 и etc   ### rtsp://192.168.75.179:8554/operator/h264/tv
    #path_video = f"/home/usr/PycharmProjects/yolo_proj/ultralytics/final_weights_train/ground_to_air/best_yolo11x_288x288_batch_64.pt"
    #path_video = f"rtsp://192.168.75.179:8554/operator/h264/tv"
    #path_video = f"/home/usr/Видео/stock-footage-drone-formation-flying-in-the-air-for-detection.webm"
    #path_video = f"rtspsrc location=rtsp://192.168.75.179:8554/operator/h264/tv latency=0 ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! appsink drop=true max-buffers=1"
    #path_video = f"/home/usr/my_video-2.mkv"
    #path_video = f"/home/usr/Рабочий стол/video/DJI_0571.MP4"
    #path_video = f"/home/usr/Видео/stock-footage-several-drones-flying-in-airspace-and-searching-for-enemy-positions-military-birds-performing.webm"
    #path_video = f"/home/usr/Видео/stock-footage-drone-shot-following-a-flock-of-birds-as-they-fly-together-on-a-clear-sunny-day.webm"
    #path_video = 2
    path_video = f"/home/usr/PycharmProjects/yolo_proj/ultralytics/videos/ground_to_air/2_n.avi"
    flag_use_denoise = False
    cap_flag_set = False
    flag_video = True
    pth = f"/home/usr/sahi3/" # путь к картинкам

    q_video = mp.Queue(maxsize=1)
    q_set = mp.Queue(maxsize=1)
    q_to_mot = mp.Queue(maxsize=1)
    q_frames, q_in, q_out = mp.Queue(maxsize=1), mp.Queue(maxsize=1), mp.Queue(maxsize=1)
    q_send_names = mp.Queue(maxsize=1)
    print(pth)
                                                                        # /home/usr/Рабочий стол/weights_yolo26/drone_iter_3_m/train23/weights/best.pt
    cl = Yolo_batches(q_frames, q_in, q_out, q_send_names, q_to_mot)   # /home/usr/PycharmProjects/yolo_proj/ultralytics/runs/detect/train20/weights/best.pt    /home/usr/Рабочий стол/weights_yolo26/drone_iter_2/train22/weights/best.pt
    cl.set_path_model("/home/usr/Рабочий стол/weights_yolo26/drone_iter_3_m/train23/weights/best.pt")   #### /home/usr/PycharmProjects/yolo_proj/ultralytics/final_weights_train/ground_to_air/best_yolo11x_288x288_batch_64.pt
    cl.set_size_inp_layers(288)
    cl.set_conf_model(0.6)
    print("SIZE_LAYERS", cl.get_size_inp_layers())
    cl.process_start()
    mot_start_proc = MOT(q_to_mot)
    mot_start_proc.set_path_weights_reid('osnet_x0_25_msmt17.pt')  # 'osnet_x0_25_msmt17.pt' /home/usr/Загрузки/resnet50_ (3).onnx
    #mot_start_proc.set_path_weights_reid('osnet_x0_25_msmt17.pt')
    mot_start_proc.set_mot_method(5)
    mot_start_proc.set_flag_half(True)
    mot_start_proc.set_flag_show_track(True)
    mot_start_proc.run_process()
    names_classes = q_send_names.get()

    print("Classes_load_model:", names_classes)


    neuro_start_proc = Yolo_inits_batch(q_out, names_files="", saved_mode=None, name_folder="/home/usr/Изображения/saver_detect_7777777", classes_naames=names_classes)
    cl.set_nms_type("classic")
    neuro_start_proc.run_nets()
    #### пофиксить режим сохраенения при выборе кроппа больше чем сама картинка

    kernel = np.ones((4, 4), dtype=np.float32) / 17
    if flag_video:

        proc_vid = mp.Process(target=process_video, args=(q_video, q_set, cap_flag_set, path_video), daemon=False)
        proc_vid.start()
        if cap_flag_set:
            q_set.put([set_w, set_h])
        time.sleep(4)
        if not q_video.empty():
            img = q_video.get()
            h_p, w_p, _ = img.shape
        else:
            flg, img = False, None
            exit(0)


        fps_timer = Timer()
        fps_timer.start()

        lst_cord = calc_scan_areas([0, 0, w_p, h_p], window_w_h=(crop_w, crop_h), overlay=(overlay_w, overlay_h))
        prev_frames_for_motion_est = []  # Для оценки движения (опционально)
        max_prev_frames = 1
        while True:
            if not q_video.empty():

                image = q_video.get()
                print(f"Image shape: {image.shape}, dtype: {image.dtype}")

                list_image, list_cropp_cord = [], []
                for i, k in enumerate(lst_cord):
                    fragment_processed = image[k[1]:k[3], k[0]:k[2]]

                    list_image.append(fragment_processed)
                    list_cropp_cord.append([k[0], k[1]])
                    #q_in.put((image[k[1]:k[3], k[0]:k[2]], [k[0], k[1], k[2], k[3]], [frame_count, len([[k[0], k[1], k[2], k[3]]]), 1, cnn]))
                list_image.append(image)
                list_cropp_cord.append([0, 0])
                if q_in.empty():
                    q_in.put([list_image, list_cropp_cord, image])

                # if prev_frames_for_motion_est is not None:
                #     prev_frames_for_motion_est.insert(0, image.copy())
                #     if len(prev_frames_for_motion_est) > max_prev_frames:
                #         prev_frames_for_motion_est.pop()
                elapsed = fps_timer.stop()
                fps_timer.start()
                print(f'capture fps: {1 / elapsed if elapsed != 0 else 1}')
                cv.imshow('original', cv.resize(image, (600, 600)))

                cv.waitKey(1)


    else:
        lst = os.listdir(pth)

        img = cv2.imread(pth+lst[0])
        h_p, w_p, _ = img.shape

        fps_timer = Timer()
        fps_timer.start()

        lst_cord = calc_scan_areas([0, 0, w_p, h_p], window_w_h=(crop_w, crop_h), overlay=(overlay_w, overlay_h))
        for i, k in enumerate(lst):


            image = cv2.imread(pth+k)
            print(image.shape)
            list_image, list_cropp_cord = [], []
            for i, k in enumerate(lst_cord):
                list_image.append(image[k[1]:k[3], k[0]:k[2]])
                list_cropp_cord.append([k[0], k[1]])
            list_image.append(image)
            list_cropp_cord.append([0, 0])
            if q_in.empty():
                q_in.put([list_image, list_cropp_cord, image])



            elapsed = fps_timer.stop()
            fps_timer.start()
            print(f'capture fps: {1 / elapsed if elapsed != 0 else 1}')
            cv.imshow('original', cv.resize(image, (600, 600)))

            cv.waitKey(0)