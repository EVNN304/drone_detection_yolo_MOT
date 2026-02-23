import random
import threading

from arena_api.system import system
import time
import numpy as np
import multiprocessing as mp
import cv2 as cv
from math import ceil
from random import randint
from toolset import Mp_dev_interface, Command
import bin_image_proc_clear as bip
from geometry_lib import Image_meta
import os
from toolset import Timer



class Lucid_soft_triggered_meta:
    '''
    Класс для работы с камерой Lucid через интерфейс
    '''
    def __init__(self):

        self.interface = Mp_dev_interface()




    def listen_n_reply(self):
        print('connecting to lucid')
        device = create_devices_with_tries()[0]
        config_device_soft_trig(device)
        run = True
        with device.start_stream(1):
            device.nodemap['TriggerSoftware'].execute()
            buffers = device.get_buffer(1)
            device.requeue_buffer(buffers)
            # cv2.namedWindow("frame", 0)
            print('Start')
            while run:
                got,cmd = self.interface.get_cmd_to_dev()
                if got:
                    if cmd.name == 'get_image':
                        meta = cmd.val
                        self.interface.push_rep_from_dev('image',[meta,np.zeros((100,100))])
                else:
                    time.sleep(0.02)

    def run(self):

        w_process = mp.Process(target=self.listen_n_reply, args=())
        w_process.daemon = True

        w_process.start()
        w_process.join()




def get_triggered_image(device):
    '''
    Получение кадра с помощью программного триггера (должен быть заранее настроен)
    '''
    if device.nodemap['TriggerArmed']:
        device.nodemap['TriggerSoftware'].execute()
        # print(f'Stream started with 1 buffer')
        # print('\tGetting one buffer')
        # 'Device.get_buffer()' with no arguments returns only one buffer

        buffer = device.get_buffer(1)

        image = np.ctypeslib.as_array(
            buffer.pdata,
            (buffer.height, buffer.width))
        device.requeue_buffer(buffer)
        return True, image
    else:
        print('Триггер не сконфигурирован')
        return False, None

def find_devices_and_set_ips(subnet_mask,start_ip):
    # Discover devices --------------------------------------------------------
    print('Discover devices on network')
    device_infos = system.device_infos
    print(f'{len(device_infos)} devices found')

    if not device_infos:
        raise BaseException('No device is found!')
    current_ip = start_ip
    # Choose the first device for this example
    for device_info in device_infos:

        # Forcing the IP address requires a device's MAC address to specify the
        # device. This example grabs the IP address, subnet mask, and default
        # gateway as well to display changes and return the device to its
        # original IP address.
        print('Device info: ')
        print(device_info)

        # Create new IP -----------------------------------------------------------

        print('Current IP = ', device_info['ip'])

        # The original device_info can be used however system.force_ip
        # will used on 'mac' ,'ip' ,'subnetmask' , and 'defaultgateway'.
        # This new dictionary to show what is needed by 'System.force_ip()'
        device_info_new = {
            'mac': device_info['mac'],
            'ip': current_ip,
            'subnetmask': subnet_mask,
            'defaultgateway': device_info['defaultgateway']
        }
        print('New IP     = ', device_info_new['ip'])

        # Force IP ----------------------------------------------------------------

        # Note: The force_ip function can also take a list of device infos to
        # force new IP addesses for multiple devices.
        print('New IP is being forced')
        system.force_ip(device_info_new)
        print('New IP was forced successfully')
        current_ip = add_one_to_ip(current_ip)

def create_devices_with_tries():

    """
    This function will let users know that a device is needed and
    gives them a chance to connect a device instead of raising an exception
    """
    tries = 0
    tries_max = 6
    sleep_time_secs = 10

    while tries < tries_max:  # Wait for device for 60 seconds
        devices = system.create_device()
        if not devices:
            print(
                f'Try {tries+1} of {tries_max}: waiting for {sleep_time_secs}'
                f'secs for a device to be connected!')
            for sec_count in range(sleep_time_secs):
                time.sleep(1)
                print(f'{sec_count + 1 } seconds passed ',
                      '.' * sec_count, end='\r')
            tries += 1
        else:
            print(f'Created {len(devices)} device(s)\n')
            return devices
    else:
        raise Exception(f'No device found! Please connect a device and run '
                        f'the example again.')

def set_autonegotiation(device):
    """
    Use max supported packet size. We use transfer control to ensure that
    only one camera is transmitting at a time.
    """
    device.tl_stream_nodemap['StreamAutoNegotiatePacketSize'].value = True

def set_frame_rate(device,frame_rate:float):
    '''

    :param device:
    :return:
    '''
    device.nodemap['AcquisitionFrameRateEnable'].value = True
    device.nodemap['AcquisitionFrameRate'].value = frame_rate
def config_soft_trigger(device):
    # Настройка триггера
    device.nodemap['TriggerSelector'].value = 'FrameStart'
    device.nodemap['TriggerMode'].value = 'On'
    device.nodemap['TriggerSource'].value = 'Software'

def config_exposure(device,auto=True,value = -1):
    if auto:
        device.nodemap['ExposureAuto'].value = 'Continuous'
    else:
        device.nodemap['ExposureAuto'].value = 'Off'
        if value>0:
            device.nodemap['ExposureTime'].value = float(value)

def config_gain(device,auto = True,value = -1):
    if auto:
        device.nodemap['GainAuto'].value = 'Continuous'
    else:
        device.nodemap['GainAuto'].value = 'Off'
        if value>=0:
            device.nodemap['Gain'].value = float(value)

def config_device_soft_trig(device):
    '''
    Типовая
    :param device:
    :return:
    '''
    set_autonegotiation(device)
    # Enable stream packet resend
    device.tl_stream_nodemap['StreamPacketResendEnable'].value = True
    #Переворот изображения контроллером камеры
    device.nodemap['ReverseY'].value = False
    set_frame_rate(device,10.0)
    device.nodemap['GainAuto'].value = 'Continuous'
    # device.nodemap['GainAuto'].value = 'Off'
    # device.nodemap['TargetBrightness'].value = 128
    config_soft_trigger(device)
    print(device.nodemap['TriggerArmed'])

def config_device_link_limit(device,enable_limit = True,throughput = 300000000):

    device.nodemap['DeviceLinkThroughputLimitMode'].value = 'On' if enable_limit else 'Off'
    if enable_limit:
        min_limit = 156250000
        max_limit = 625000000
        throughput_valid = max(min_limit,throughput)
        throughput_valid = min(max_limit,throughput_valid)
        device.nodemap['DeviceLinkThroughputLimit'].value = throughput_valid
    else:
        pass

def image_stream(out_q:mp.Queue):
    # Create a device
    # devices = create_devices_with_tries()
    # device = devices[0]
    dev_info = system.device_infos
    # print(dev_info)
    # devices = system.create_device(dev_info)
    # device = devices[0]
    # system.destroy_device(devices[1])
    device = system.create_single_device(dev_info[0])
    # device = ('1c:0f:af:00:6b:19', 'ATX204S-M', '', '192.168.233.41')
    print(f'Device used in the example:\n\t{device}')
    config_device_soft_trig(device)
    config_exposure(device,True,1000)
    device.nodemap['TargetBrightness'].value = 50

    with device.start_stream(1):
        device.nodemap['TriggerSoftware'].execute()
        buffers = device.get_buffer(1)
        device.requeue_buffer(buffers)
        # cv2.namedWindow("frame", 0)
        print('Start')

        while True:
            if ((out_q.qsize() <= 2)and device.nodemap['TriggerArmed'].value):
                # device.nodemap['TargetBrightness'].value = randint(0,255)
                # config_exposure(device,False,7.0+randint(0,50000)/10)
                # device.nodemap['TriggerSoftware'].execute()
                # print(f'Stream started with 1 buffer')
                # print('\tGetting one buffer')
                # # 'Device.get_buffer()' with no arguments returns only one buffer
                #
                # buffer = device.get_buffer(1)
                #
                #
                # image = np.ctypeslib.as_array(
                #             buffer.pdata,
                #             (buffer.height, buffer.width))

                # device.requeue_buffer(buffer)
                flag, image = get_triggered_image(device)
                # config_exposure(device,False,random.randint(200,600))
                # config_gain(device, True, random.randint(0, 48))
                out_q.put(image)
            else:
                print('ожидание обработки')
                time.sleep(0.002)

def show_images_from_queue(q_in:mp.Queue):
    # window = cv.namedWindow('window',0)
    win_num = random.randint(0,100)
    res = np.zeros((1000,1000),dtype=np.uint8)
    prev = time.time()
    font = cv.FONT_HERSHEY_SIMPLEX
    print('Запущен процесс обработки/показа')
    while True:
        if not q_in.empty():
            frame = q_in.get()
            now = time.time()
            timer = now-prev
            prev = now
            fps = 1/timer
            print(q_in.qsize())
            # draw_grid(frame,(1200,1200),200)
            cv.resize(frame,dsize =(1000,1000),dst=res)
            # cv.putText(res,str(fps))
            cv.putText(res, f'fps: {str(round(fps,1))}', (700, 50), font, 1, (0, 0, 0), 2, cv.LINE_AA)
            cv.putText(res, f'queue size: {q_in.qsize()}', (700, 100), font, 1, (0, 0, 0), 2,
                       cv.LINE_AA)
            cv.imshow(f'window_{win_num}',cv.resize(res,(750,750)))
            cv.waitKey(1)


def standart_proc(dif):
    kernel = np.ones((4, 4), np.uint8)
    thresh = 20
    cv.threshold(dif, thresh, 1, cv.THRESH_BINARY, dif)
    '''Эрозия'''
    cv.erode(dif, kernel, dif, iterations=1)
    '''Дилатация'''
    cv.dilate(dif, kernel, dif, iterations=1)

def show_processed_images_from_queue(q_in:mp.Queue):
    # window = cv.namedWindow('window',0)
    res = np.zeros((1600,1600),dtype=np.uint8)
    prev = time.time()
    font = cv.FONT_HERSHEY_SIMPLEX
    print('Запущен процесс обработки/показа')
    prev_frame = np.zeros((4504,4504),dtype = np.uint8)
    counter = 0
    # folder_path = 'diff_images'
    folder_path = 'raw_images'
    t = Timer()
    while True:
        if not q_in.empty():
            frame = q_in.get()
            # if prev_frame:
            t.start()
            dif_frame = cv.absdiff(frame,prev_frame)
            bip.standart_proc(dif_frame)
            elapsed = t.stop()
            print(f'elapsed for standart proc {elapsed}')
            t.start()
            try:

                areas = bip.find_areas_on_binary(dif_frame)
                elapsed = t.stop()
                print(f'elapsed for areas [{len(areas)}] search {elapsed}')
                # for a in areas:
                #     print(a.to_string())
                #     area_p1 = (a.bbox[0], a.bbox[1])
                #     area_p2 = (a.bbox[2], a.bbox[3])
                #     cv.rectangle(frame, area_p1, area_p2, 155, 2)
            except Exception:
                print('ex')

            file_name = f'{counter}.jpg'
            # cv.imwrite(os.path.join(folder_path,file_name),frame)
            counter+=1
            # standart_proc(dif_frame)
            # cv.imshow('window_dif', cv.resize(dif_frame,(1600,1600)))
            # cv.imshow('window_dif', cv.resize(dif_frame*200, (1600, 1600)))

            prev_frame = frame



            now = time.time()
            timer = now-prev
            print(f'<Timer> {timer}')
            prev = now
            fps = 1/timer
            print(q_in.qsize())
            # draw_grid(frame,(1200,1200),200)
            cv.resize(frame,dsize =(1600,1600),dst=res)
            # cv.putText(res,str(fps))
            cv.putText(res, f'fps: {str(round(fps,1))}', (700, 50), font, 1, (0, 0, 0), 2, cv.LINE_AA)
            cv.putText(res, f'queue size: {q_in.qsize()}', (700, 100), font, 1, (0, 0, 0), 2,
                       cv.LINE_AA)
            cv.imshow('window_',cv.resize(res,(1200,1200)))
            cv.waitKey(30)







# def show_3frdif_images_from_queue(q_in:mp.Queue):
#     # window = cv.namedWindow('window',0)
#     res = np.zeros((1600,1600),dtype=np.uint8)
#     prev = time.time()
#     font = cv.FONT_HERSHEY_SIMPLEX
#     print('Запущен процесс обработки/показа')
#     prev_frame = np.zeros((4504,4504),dtype = np.uint8)
#     counter = 0
#     while True:
#         if not q_in.empty():
#             frame = q_in.get()
#             # if prev_frame:
#             dif_frame = cv.subtract(frame,prev_frame)
#             cv.imshow('window_dif', cv.resize(dif_frame,(1600,1600)))
#
#             prev_frame = frame
#             I1 = (cv.absdiff(gray3, gray2)).astype(np.int16)
#             I2 = (cv.absdiff(gray2, gray1)).astype(np.int16)
#             I_d = I1 - I2
#             I_d = I_d * (I_d > 0)
#             I_d_f = I_d.astype(np.uint8)
#
#
#             now = time.time()
#             timer = now-prev
#             prev = now
#             fps = 1/timer
#             print(q_in.qsize())
#             # draw_grid(frame,(1200,1200),200)
#             cv.resize(frame,dsize =(1600,1600),dst=res)
#             # cv.putText(res,str(fps))
#             cv.putText(res, f'fps: {str(round(fps,1))}', (700, 50), font, 1, (0, 0, 0), 2, cv.LINE_AA)
#             cv.putText(res, f'queue size: {q_in.qsize()}', (700, 100), font, 1, (0, 0, 0), 2,
#                        cv.LINE_AA)
#             cv.imshow('window',res)
#             cv.waitKey(1)

def draw_grid(frame, patch_shape, overlay):
    shades = [25,50,75,100,125,150,175,200,225,250]
    # grid_pict = np.zeros(np.shape(frame), dtype=np.uint8)
    h,w = np.shape(frame)
    h_,w_ = patch_shape

    #количество фрагментов по x,y
    n_x = ceil(w/(w_-overlay))
    n_y = ceil(h/(h_-overlay))

    over_x = round((w_*n_x - w)/n_x)
    over_y = round((h_*n_x-h)/n_y)

    # print(f'Количество шагов по x: {n_x}\n'
    #       f'Количество шагов по y: {n_y}\n'
    #       f'Реальное перекрытие по x: {over_x}\n'
    #       f'Реальное перекрытие по y: {over_y}')

    for j in range(n_y):
        if j==((n_y)-1):
            y2 = h-1
            y1 = y2-h_
        else:
            y1 = (h_-over_y)*j
            y2 = y1+h_
        for i in range(n_x):
            if i == ((n_x) - 1):
                x2 = w - 1
                x1 = x2 - w_
            else:
                x1 = (w_-over_x) * i
                x2 = x1 + w_

            # print(f'{j},{i}: ({x1},{y1}),({x2},{y2})')
            shade = shades[randint(0,9)]
            font = cv.FONT_HERSHEY_SIMPLEX
            cv.rectangle(frame, (x1, y1), (x2, y2), shade, 4,)
            cv.putText(frame, f'({i},{j})', (int(x1+w_/2), int(y1+h_/2)), font, 2, shade, 4, cv.LINE_AA)


def search_devices_and_force_ip(start_ip = '192.168.1.51',subnet_mask = '255.255.255.0',gateway = '192.168.1.1'):
    next_ip = start_ip
    device_infos = system.device_infos

    print(f'Найдены устройства: {device_infos}')
    for device_info in device_infos:
        new_ip = next_ip
        device_info_new = {
            'mac': device_info['mac'],
            'ip': new_ip,
            'subnetmask': subnet_mask,
            'defaultgateway': gateway
        }
        system.force_ip(device_info_new)
        next_ip = add_one_to_ip(next_ip)
        print(f'Назначен адрес {new_ip} (mac: {device_info["mac"]})')
    return new_ip

def add_one_to_ip(ip):

    bit0, bit1, bit2, bit3 = ip.split('.')
    if bit3 == '254':  # Avoid 255
        bit3 = '1'
    else:
        bit3 = str(int(bit3) + 1)
    return f'{bit0}.{bit1}.{bit2}.{bit3}'

def run_dev():
    dev = Lucid_soft_triggered_meta()
    dev.listen_n_reply()

if __name__ == '__main__':
    # search_devices_and_force_ip()
    # devices = create_devices_with_tries()

    # for device in devices:
    #     print(device)
    #     # nodes = device.nodemap.feature_names
    #     # for node in nodes:
    #     #     print(node)
    #     # print(device.nodemap.get_node(nodes))

    q_images = mp.Queue()
    aq_process = mp.Process(target=image_stream, args=(q_images,))
    aq_process.start()
    show_proc = mp.Process(target=show_processed_images_from_queue, args=(q_images,))
    # show_proc = mp.Process(target=show_images_from_queue, args=(q_images,))
    show_proc.start()


    # campr = Lucid_soft_triggered_meta()
    # pr = mp.Process(target=run_dev())
    # pr.start()
    # campr.listen_n_reply()
    # grid_pict = np.ones((4504,4504), dtype=np.uint8)


    # grid_pict = draw_grid(grid_pict,(1000,1000),250)
    # res = np.zeros((1000, 1000), dtype=np.uint8)
    # grid_pict = np.ones((4504, 4504), dtype=np.uint8)
    # # grid_pict = grid_pict+100
    # draw_grid(grid_pict,(1000,1000),200)
    # cv.resize(grid_pict,(1000,1000),dst = res)
    # cv.imshow('w',res)
    # cv.waitKey()
    # cv.destroyAllWindows()