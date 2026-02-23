import time

from lucid_additional_lib import *
import multiprocessing as mp

def run_lucid_grabber(serial:str,new_ip:str='', q_out:mp.Queue = None):
    device = create_single_dev_by_serial(serial,new_ip)
    config_device_soft_trig(device)
    config_exposure(device, True, 1000)
    device.nodemap['TargetBrightness'].value = 100

    with device.start_stream(1):
        device.nodemap['TriggerSoftware'].execute()
        buffers = device.get_buffer(1)
        device.requeue_buffer(buffers)
        # cv2.namedWindow("frame", 0)
        print('Start')

        while True:
            if ((q_out.qsize() <= 2) and device.nodemap['TriggerArmed'].value):
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
                q_out.put(image)
                time.sleep(0.1)
            else:
                print('ожидание обработки')
                time.sleep(0.1)




if __name__ == '__main__':
    q = mp.Queue(3)
    # serial = '213201880'
    # set_ip = '192.168.101.42'
    serial = '213201882'     ##### 213201882    '213201881'
    set_ip = '192.168.101.45'
    grab_proc = mp.Process(target=run_lucid_grabber, args = (serial,set_ip,q))
    view_proc = mp.Process(target = show_images_from_queue,args = (q,))
    grab_proc.start()
    view_proc.start()


