

#from lucid_additional_lib import *
import multiprocessing as mp
import cv2 as cv
import time


class Camera_Abstract:
    def __init__(self, q_input:mp.Queue, ip_cam_adress="", set_ip_lucid="", serial_lucid=""):
        self.q_input = q_input
        self.show, self.time_delay = None, 1
        self.ip_cam_adress = ip_cam_adress
        self.set_ip_lucid, self.serial_lucid = set_ip_lucid, serial_lucid
        self.h, self.w = None, None


    def set_show(self, flag):
        self.show = flag

    def set_time_delay(self, val):
        self.time_delay = val

    def get_hw(self):
        return self.h, self.w

    def setter_ip_cam_adress(self, value):
        '''Здесь можно установить ip-адрес стандартной ip камеры,
        также здесь можно определить значение (от 0-10), для подключения к камере с соединением типа USB
        или же возпроизвести видеофайл с указанием пути к этому файлу'''

        self.ip_cam_adress = value

    def getter_ip_adress(self):
        return self.ip_cam_adress

    def setter_ip_cam_for_lucid(self, value):
        '''Здесь можно установить ip-адрес для камеры высогоко разрешения типа LUCID'''

        self.set_ip_lucid = value

    def get_ip_lucid(self):
        return self.set_ip_lucid

    def setter_serial_lucid_device(self, value):
        '''Здесь прописывается серийный номер камеры типа LUCID (см мак-адресс на торцевой части камеры или же в специальной API ArenaView'''

        self.serial_lucid = value

    def get_mac_lucid(self):
        return self.serial_lucid

    def set_queue(self, queue):

        '''Здесь прописывается очередь, в которую будут помещаться изображения'''

        self.q_input = queue

    def run_method_ip_cam(self):
        '''Здесь производится старт процесса для ip или usb камеры'''
        process_ip_cam = mp.Process(target=self.get_frame_usb_or_ip_cam, args=(self.q_input, self.ip_cam_adress,))
        process_ip_cam.daemon = True
        process_ip_cam.start()

    def check_work_lucid_process(self, name_lucid_poc):
        while True:
            if name_lucid_poc.is_alive():
                print("process_lucid_cam_works:", name_lucid_poc.is_alive())

            else:
                name_lucid_poc.join(3)
                name_lucid_poc.terminate()
                name_lucid_poc.close()
                time.sleep(2)
                name_lucid_poc.start()

    def run_method_lucid_cam(self):
        '''Здесь производится старт процесса LUCID камеры'''
        process_lucid = mp.Process(target=self.get_frame_lucid_cam, args=(self.q_input, self.set_ip_lucid, self.serial_lucid,))
        #prcoess_lucid_check = threading.Thread(target=self.check_work_lucid_process, args=(process_lucid,))
        process_lucid.daemon = True
        # prcoess_lucid_check.daemon = True
        process_lucid.start()
        # prcoess_lucid_check.start()





    def get_frame_usb_or_ip_cam(self, q_input, ip_cam_adress):
        cam = cv.VideoCapture(ip_cam_adress)


        c = 0
        while True:
            try:
                frame_valid, image = cam.read()
                image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
                h, w = image.shape
                c = 0
                #print("shape_ip", image.shape)
                if q_input.empty():
                    q_input.put(image)
                    time.sleep(0.019)

                if self.show:
                    cv.imshow("Original", image)
                    cv.waitKey(self.time_delay)
                #print(f"Frame_delay_original:{time.time() - t1}")
            except Exception as err_cam:
                print(f"Камера отвалилась, ошибка типа:{err_cam.args}")
                c += 1
                cam.release()
                time.sleep(1.5)
                cam = cv.VideoCapture(ip_cam_adress)
                if c == 25:
                    print(f"Попытка переподключения к камере_провалилась, пытаюсь перезапустить процесс.............")
                    break


    def get_frame_lucid_cam(self, q_input, set_ip_lucid, serial_lucid):
        c = 0
        while True:
            try:

                device = create_single_dev_by_serial(serial_lucid, set_ip_lucid)
                print(f'Device used in the example:\n\t{device}')
                # config_device_link_limit(device, enable_limit=True, throughput=100000000)
                config_device_soft_trig(device)

                config_exposure(device, True, 1000)
                # device.nodemap['TargetBrightness'].value = 105
                device.nodemap['TargetBrightness'].value = 105
                with device.start_stream(1):
                    #device.nodemap['TriggerSoftware'].execute()
                    print('Start')
                    while True:

                        if device.nodemap['TriggerArmed'].value:
                            frame_valid, images = get_triggered_image(device)
                            print("shape_lucid", images.shape)

                            if frame_valid:
                                c = 0
                                if q_input.empty():
                                    q_input.put(images)
                                else:
                                    print("Prolems_QUQUE")
                            else:
                                print("problem_IMAGE")

            except Exception as e:
                c += 1
                print(f"Камера_LUCID_{serial_lucid}_{set_ip_lucid}_перестала отвечать, ошибка типа: {e.args}, попытка переподключения {c}")

                system.destroy_device(None)
                time.sleep(6)
                if c == 25:
                    print(f"Попытка переподключения к камере_LUCID_{serial_lucid}_{set_ip_lucid}_провалилась, пытаюсь перезапустить процесс.............")
                    break



