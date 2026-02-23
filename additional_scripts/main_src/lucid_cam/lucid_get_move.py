import copy
import time

from lucid_standart_lib import *


def standart_proc(dif):
    kernel = np.ones((3, 3), np.uint8)
    kernel2 = np.ones((16, 16), np.uint8)
    thresh = 10
    cv.threshold(dif, thresh, 1, cv.THRESH_BINARY, dif)
    '''Эрозия'''
    cv.erode(dif, kernel, dif, iterations=1)
    '''Дилатация'''
    cv.dilate(dif, kernel2, dif, iterations=2)


# Create a device
devices = create_devices_with_tries()
device = devices[0]
print(f'Device used in the example:\n\t{device}')
config_device_soft_trig(device)
config_exposure(device, True, 5000)
# kernel = np.ones((3, 3))


with device.start_stream(1):
    device.nodemap['TriggerSoftware'].execute()
    buffers = device.get_buffer(1)
    device.requeue_buffer(buffers)
    # cv2.namedWindow("frame", 0)
    print('Start')

    while True:
        got = False
        while not(got):
            if (device.nodemap['TriggerArmed'].value):
                got, image = get_triggered_image(device)
                gray1 = copy.deepcopy(image)
        print(got)
        time.sleep(0.03)
        got = False
        while not(got):
            if (device.nodemap['TriggerArmed'].value):
                got, image = get_triggered_image(device)
                gray2 = copy.deepcopy(image)
        print(got)
        time.sleep(0.03)
        got = False
        while not(got):
            if (device.nodemap['TriggerArmed'].value):
                got, image = get_triggered_image(device)
                gray3 = copy.deepcopy(image)
        print(got)

        t_start = time.time()

        I1 = cv.absdiff(gray2, gray1)
        I2 = cv.absdiff(gray3, gray2)
        I_d_f = cv.subtract(I2, I1)
        #
        standart_proc(I_d_f)
        t_middle = time.time()

        сontours, _ = cv.findContours(I_d_f, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

            # Detection_Cordinate = []
        for i, contour in enumerate(сontours):
            (x, y, w, h) = cv.boundingRect(contour)

            # Detection_Cordinate.append([x, y, x + w, y + h, w, h])
            cv.rectangle(gray3, (x, y), (x + w, y + h), (0, 0, 0), 6)

        t_finish = time.time()

        print(f'Middle tout = {t_middle - t_start}')
        print(f'Finish tout = {t_finish - t_start}')
        cv.imshow('frame', cv.resize(gray3, (1400, 1400)))
        # cv.imshow('fr3', cv.resize(gray3, (1400, 1400)))
        cv.imshow('extracted', cv.resize(I_d_f * 200, (1400, 1400)))
        cv.waitKey(1)

