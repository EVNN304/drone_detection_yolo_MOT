from lucid_standart_lib import *

def get_connected_dev_list():
    dev_info = system.device_infos
    for dev in dev_info:
        print(dev)
    return dev_info

def get_dev_info_by_serial(serial:str):
    found_device = None
    for d in system.device_infos:
        if d['serial'] in serial:
            found_device = d
            break
    return found_device

def get_dev_info_by_mac(mac:str):
    found_device = None
    for d in system.device_infos:
        if d['mac'] in mac:
            found_device = d
            break
    return found_device

def create_single_dev_by_serial(serial_num:str,set_ip:str = ''):
    dev_info  = get_dev_info_by_serial(serial_num)
    device = None
    if dev_info:
        if len(set_ip)>1:
            print(f'device_found: {dev_info}')
            device_info_new = {
                'mac': dev_info['mac'],
                'ip': set_ip,
                'subnetmask': dev_info['subnetmask'],
                'defaultgateway': dev_info['defaultgateway']
            }
            # print(f'dev:\n{found_device}\n')
            system.force_ip(device_info_new)
            print(f'ip changed to {set_ip}')
            dev_info = get_dev_info_by_serial(serial_num)
        print(f'try to create: {dev_info}')
        device = system.create_single_device(dev_info)
    return device





if __name__ == '__main__':
    # found_device = None
    # devs = get_connected_dev_list()
    # for d in devs:
    #     if d['serial'] in '213201880':
    #         print('!!', d['ip'])
    #         # d['ip'] = '192.168.101.55'
    #         found_device = d
    # # found_device['ip'] = '192.168.101.61'
    #
    # # dev = system.create_single_device(found_device)
    #
    # device_info_new = {
    #     'mac': found_device['mac'],
    #     'ip': '192.168.101.37',
    #     'subnetmask': found_device['subnetmask'],
    #     'defaultgateway': found_device['defaultgateway']
    # }
    # # print(f'dev:\n{found_device}\n')
    # system.force_ip(device_info_new)
    # # system.destroy_device(dev)
    #
    # #
    # print('changed ip')

    # dev_info = get_dev_info_by_serial(ser)
    # print(f'serial {ser} -> {dev_info}')
    # mac = '1c:0f:af:00:6b:19'
    # dev_info = get_dev_info_by_mac(mac)
    # print(f'mac {ser} -> {dev_info}')
    # # get_connected_dev_list()

    ser = '232100009'
    new_ip = '192.168.101.42'
    device = create_single_dev_by_serial(ser,new_ip)
    device.start_stream(1)
    device.stop_stream()
    # system.destroy_device(device)     1C:0F:AF:07:EC:1C

