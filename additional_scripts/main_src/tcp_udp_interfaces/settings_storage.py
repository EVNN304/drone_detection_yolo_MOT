from xml.etree.ElementTree import *

# core.set_video_master_self_address('192.168.0.20', 5005)
# core.set_video_master_serv_address('192.168.0.21', 5005)
# core.set_video_slave_self_address('192.168.0.20', 5006)
# core.set_video_slave_serv_address('192.168.0.2', 5005)
#
# # Настройка адресов UDP для работы с распознаваниями
# core.set_recg_master_self_address('192.168.0.20', 20041)
# core.set_recg_master_serv_address('192.168.0.21', 20041)
# core.set_recg_slave_self_address('192.168.0.20', 20042)
# core.set_recg_slave_serv_address('192.168.0.2', 20041)
#
# # Настройка адресов UDP для работы с траекториями
# core.set_tracker_master_self_address('192.168.0.20', 20051)
# core.set_tracker_master_serv_address('192.168.0.21', 20051)
# core.set_tracker_slave_self_address('192.168.0.20', 20052)
# core.set_tracker_slave_serv_address('192.168.0.2', 20051)
def pack_address_to_xml_el(address,header = 'address'):
    el = Element(header)
    ip_el = SubElement(el,'ip')
    ip_el.text = str(address[0])
    port_el = SubElement(el,'port')
    port_el.text = str(address[1])
    return el

def pack_opu_settings_to_xml_el(address,zero_az_el):
    el = Element('OPU_settings')
    el.append(pack_address_to_xml_el(address))
    zero_el = Element('Zero_position')
    az0_el = SubElement(zero_el,'az0')
    az0_el.text = str(zero_az_el[0])
    el0_el = SubElement(zero_el,'el0')
    el0_el.text = str(zero_az_el[1])
    el.append(zero_el)
    return el

def unpack_opu_settings_from_xml_el(el:Element):
    addr_rec = el.find('address')
    address = unpack_address_from_xml_el(addr_rec)
    zero_rec = el.find('Zero_position')
    az_0 = float(zero_rec.find('az0').text)
    el_0 = float(zero_rec.find('el0').text)
    return address,[az_0,el_0]

def unpack_address_from_xml_el(el:Element):
    ip_rec = el.find('ip')
    ip = ip_rec.text
    port_rec = el.find('port')
    port = int(port_rec.text)
    return (ip,port)

def pack_camera_settings_to_xml_el(shift,frame_size,view_angles):
    el = Element('Camera')
    shift_el = SubElement(el,'Shift_angles')
    h_shift = SubElement(shift_el,'horizontal')
    h_shift.text = str(shift[0])
    v_shift = SubElement(shift_el,'vertical')
    v_shift.text = str(shift[1])
    frame_size_el = SubElement(el,'Frame_size')
    h_size = SubElement(frame_size_el,'horizontal')
    h_size.text = str(frame_size[0])
    v_size = SubElement(frame_size_el, 'vertical')
    v_size.text = str(frame_size[1])
    view_el = SubElement(el, 'View_angles')
    h_view = SubElement(view_el, 'horizontal')
    h_view.text = str(view_angles[0])
    v_view = SubElement(view_el, 'vertical')
    v_view.text = str(view_angles[1])
    return el
def unpack_camera_settings_from_xml_el(el:Element):
    shift_rec = el.find('Shift_angles')
    shift_h = float(shift_rec.find('horizontal').text)
    shift_v = float(shift_rec.find('vertical').text)
    fr_size_rec = el.find('Frame_size')
    fr_w = int(fr_size_rec.find('horizontal').text)
    fr_h = int(fr_size_rec.find('vertical').text)
    view_rec = el.find('View_angles')
    view_w = float(view_rec.find('horizontal').text)
    view_h = float(view_rec.find('vertical').text)
    return [shift_h,shift_v],[fr_w,fr_h],[view_w,view_h]

class Server_settings_pack:
    def __init__(self):
        self.role = 'master'
        self.video_client_address = ('192.168.0.20',5005)
        self.video_server_address = ('192.168.0.21',5005)

        self.recg_client_address = ('192.168.0.20',20041)
        self.recg_server_address = ('192.168.0.21', 20041)

        self.tracker_client_address = ('192.168.0.20', 20051)
        self.tracker_server_address = ('192.168.0.21', 20051)

        self.control_client_address = ('192.168.0.20', 20031)
        self.control_server_address = ('192.168.0.21', 20031)

        self.opu_address = ('192.168.0.94',6000)
        self.opu_zero_az_el = [0,0]

        self.cam_shift = [0,0]
        self.cam_frame_size = [4504,4504]
        self.cam_view_angles = [42,42]

    def print(self):
        print('<---Settings pack--->')
        print(f'-->Server role:\t\t{self.role}\n')
        print(f'-->Control stream server: {self.control_server_address}')
        print(f'\t\t\t(client: {self.control_client_address})\n')
        print(f'-->Video stream server: {self.video_server_address}')
        print(f'\t\t\t(client: {self.video_client_address})\n')
        print(f'-->Recognition server: {self.recg_server_address}')
        print(f'\t\t\t(client: {self.recg_client_address})\n')
        print(f'-->Tracker server: {self.tracker_server_address}')
        print(f'\t\t\t(client: {self.tracker_client_address})\n')
        print(f'-->Opu address: {self.opu_address}')
        print(f'\t\t\tzero: {self.opu_zero_az_el}\n')
        print(f'-->Camera shift, frame size, view angles: {self.cam_shift,self.cam_frame_size,self.cam_view_angles}\n')

    def save_to_file(self,path):
        tree = ElementTree('root')
        settings_root = Element('Settings_pack')
        tree._setroot(settings_root)
        role_el = SubElement(settings_root,'role')
        role_el.text = str(self.role)
        #Записываем настройки для видеопотока
        video_stream_settings = SubElement(settings_root,'Video_stream')
        video_stream_settings.append(pack_address_to_xml_el(self.video_server_address,'server'))
        video_stream_settings.append(pack_address_to_xml_el(self.video_client_address, 'client'))
        # Записываем настройки для распознаваний
        recg_stream_settings = SubElement(settings_root, 'Recognition_stream')
        recg_stream_settings.append(pack_address_to_xml_el(self.recg_server_address, 'server'))
        recg_stream_settings.append(pack_address_to_xml_el(self.recg_client_address, 'client'))
        # Записываем настройки для траекторий
        tracker_stream_settings = SubElement(settings_root, 'Tracker_stream')
        tracker_stream_settings.append(pack_address_to_xml_el(self.recg_server_address, 'server'))
        tracker_stream_settings.append(pack_address_to_xml_el(self.recg_client_address, 'client'))
        # Записываем настройки для управления
        control_stream_settings = SubElement(settings_root, 'Control_stream')
        control_stream_settings.append(pack_address_to_xml_el(self.control_server_address, 'server'))
        control_stream_settings.append(pack_address_to_xml_el(self.control_client_address, 'client'))
        # Записываем настройки для работы с опу
        opu_el = pack_opu_settings_to_xml_el(self.opu_address,self.opu_zero_az_el)
        settings_root.append(opu_el)
        # Записываем настройки камеры
        settings_root.append(pack_camera_settings_to_xml_el(self.cam_shift,self.cam_frame_size,self.cam_view_angles))
        tree.write(path)



    def load_from_file(self,path):
        settings_tree = ElementTree('Settings_pack', path)
        vid_settings = settings_tree.find('Video_stream')
        self.video_server_address = unpack_address_from_xml_el(vid_settings.find('server'))
        self.video_client_address = unpack_address_from_xml_el(vid_settings.find('client'))
        recg_settings = settings_tree.find('Recognition_stream')
        self.recg_server_address = unpack_address_from_xml_el(recg_settings.find('server'))
        self.recg_client_address = unpack_address_from_xml_el(recg_settings.find('client'))
        control_settings = settings_tree.find('Control_stream')

        tracker_settings = settings_tree.find('Tracker_stream')
        self.tracker_server_address = unpack_address_from_xml_el(tracker_settings.find('server'))
        self.tracker_client_address = unpack_address_from_xml_el(tracker_settings.find('client'))

        control_settings = settings_tree.find('Control_stream')
        self.control_server_address = unpack_address_from_xml_el(control_settings.find('server'))
        self.control_client_address = unpack_address_from_xml_el(control_settings.find('client'))
        addr,shift = unpack_opu_settings_from_xml_el(settings_tree.find('OPU_settings'))
        self.opu_address = addr
        self.opu_zero_az_el = shift
        shift,fr_size, angle_s = unpack_camera_settings_from_xml_el(settings_tree.find('Camera'))
        self.cam_shift = shift
        self.cam_frame_size = fr_size
        self.cam_view_angles = angle_s

        print('ww',self.video_server_address)





if __name__=='__main__':
    # tree = ElementTree('root')
    # root = Element('root')
    # tree._setroot(root)
    # el = SubElement(root,'Server_settings')
    # test_el_1 = Element('test1')
    # test_el_1.text = '192.168.0.21'
    # el.append(test_el_1)
    # test_el_2 = Element('test2')
    # test_el_2.text = str(20031)
    #
    # el.append(test_el_2)
    # test_el3 =pack_address_to_xml_el(('192.168.0.21',20051),'dest')
    # print(unpack_address_from_xml_el(test_el3))
    # el.append(test_el3)
    # tree.write('test.xml')
    # print(el)
    settigs_pack = Server_settings_pack()
    # settigs_pack.save_to_file('settings.xml')
    settigs_pack.load_from_file('settings.xml')
    # opu_el = pack_opu_settings_to_xml_el(('192.168.0.94',6000), [68,0])
    # print(opu_el)
    settigs_pack.print()

