import time

from yolov8_lib_2 import Multi_net_process
from settings_storage import Server_settings_pack
from class_Cam_abstract import *


from init_Yolo_neural import *
from cropped_lib import *
from Eval.test2 import *


if __name__ == '__main__':

    settings_path = r'settings.xml'
    settings = Server_settings_pack()
    settings.load_from_file(settings_path)
    print('Загружаем настройки сервера...')
    settings.print()


    udp_neuro_self_address = settings.recg_server_address
    udp_neuro_dest_address = settings.recg_client_address

    video_server_interface = Mp_dev_interface(1, allow_loose=True)
    tcp_addr = settings.video_server_address



    q_frame_lucdi, q_rame_ir = mp.Queue(1), mp.Queue(1)


    opu_bundle_only = True
    process_permission = True
    frame_valid = False

    control_q_1 = mp.Queue(1)
    q_nano_track = mp.Queue(1)
    q_cord = mp.Queue(1)
    # Очередь обработчика нейросети. Размер должен быть немного больше максимального количества областей
    q_to_neuro_1 = mp.Queue(50)
    # Выходная очередь нейросети, из которой забираются все обнаружения
    q_from_neuro_1 = mp.Queue(8)
    # pr = mp.Process(target=fast_get_from_q, args= (q4mproc,))

    q_to_tracker_1 = mp.Queue(1)
    q_from_tracker = None



    # Инициализация и запуск обработчиков нейросети

    #neuro_start_proc = Yolo_inits(control_q_1, q_from_neuro_1, udp_neuro_self_address, udp_neuro_dest_address)
    #neuro_start_proc.saved_mode = mp.JoinableQueue(1)
    #neuro_start_proc.set_name_folder("//192.168.0.20/d/SERVER_192_168_0_16")


    #neuro_start_proc.run_nets()
    #neuro_start_proc.run_saver()


    neuro_processor = Multi_net_process(1, q_to_neuro_1, q_from_neuro_1)
    neuro_processor.config_run_nets()

    prc_1 = start_process(q_to_tracker_1, control_q_1, q_to_neuro_1, [416, 416], [0, 0, 1920, 1080])
    prc_2 = run_nets(control_q_1, q_from_neuro_1, q_cord)
    #prc_3 = run_main(q_nano_track, q_cord)


    # Настройка сетевого интерфейса для управления


    # Настройка сетевого интерфейса для вещания траекторий

    #####################################################



    im_size = settings.cam_frame_size
    meta = Image_meta(0, 0, settings.cam_frame_size, settings.cam_view_angles)



    fps_timer = Timer()
    fps_timer.start()

    to_tracker_cmd_list = {'settings': [10, 3, 16], 'channels_choice': [0, 1080, 1920]}





    # card_idx = [0]
    # print(len(card_idx))
    frame_count = 0
    class_cam_avi_test = Camera_Abstract(q_frame_lucdi)

    class_cam_avi_test.setter_ip_cam_adress('/home/usr/PycharmProjects/yolo_proj/ultralytics/2_n.avi')

    class_cam_avi_test.run_method_ip_cam()



    while True:

        q_frame_getter = q_frame_lucdi if to_tracker_cmd_list['channels_choice'][0] == 0 else q_rame_ir

        frame_count += 1
        if not q_frame_getter.empty():
            image = q_frame_getter.get()
            meta.timestamp = time.time()
            if q_to_tracker_1.empty():
                q_to_tracker_1.put((copy.deepcopy(image), copy.deepcopy(meta), frame_count))
            # if q_nano_track.empty():
            #     q_nano_track.put(image)

            elapsed = fps_timer.stop()
            fps_timer.start()
            print(image.shape)
            print(f'capture fps: {1 / elapsed if elapsed != 0 else 1}')
            cv.imshow('original', cv.resize(image, (1000, 1000)))

            cv.waitKey(1)




