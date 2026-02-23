from socket import *
import time
import multiprocessing as mp

class Param_pair():
    def __init__(self,name = None,val = None):
        self.name = name
        self.value = val

    def print(self):
        print(f'cmd name = [{self.name}], value = [{self.value}]')

class Ptzf_ex():
    '''Расширенный Ptzf для обработчика видео'''
    def __init__(self):
        self.pan = 0.0
        self.tilt = 0.0
        self.zoom = 0.0
        self.focus = 0.0
        self.dpan = 0.0
        self.dtilt = 0.0

class Pelco_de_device():
    'Класс для работы с ОПУ Бик-Информ'
    def __init__(self,ip,port):
        # self.mode = mode
        self.az0 = 0
        self.ip = ip
        self.udp_port = port
        self.udp_soc = socket(AF_INET, SOCK_DGRAM)
        self.udp_soc.settimeout(0.5)
        self.pelco_addr = 1
        self.ptzf = Ptzf_ex()
        self.ptzf_geo =Ptzf_ex()
        self.max_pan = 0
        self.pan_k = 1
        self.max_tilt = 0
        self.tilt_k = 1
        self.max_zoom = 0
        self.zoom_k = 1
        self.max_focus = 0
        self.focus_k = 1
        self.temperature = 0
        self.voltage = 0
        self.base_pelco = Pelco_base_constructor(self.pelco_addr)

    def update_geo_pan_tilt(self):
        self.ptzf_geo.tilt = self.ptzf.tilt-90
        self.ptzf_geo.pan = (self.ptzf.pan - self.max_pan * self.pan_k/2 +self.az0)%360

    def convert_pan_tilt_geo_to_native(self,pan_geo,tilt_geo):
        nat_pan = (pan_geo+self.max_pan * self.pan_k/2-self.az0)%360
        nat_tilt = tilt_geo+90

        return nat_pan,nat_tilt

    def print(self):
        print(f'<<--UDP PELCO-DE device-->>\nip: {self.ip}\nudp-port: {self.udp_port}')
        print(f'<--LIMITS-->\npan: {self.max_pan}\npan step: {self.pan_k}\ntilt: {self.max_tilt}\ntilt step: {self.tilt_k}')
        print(f'<--CURRENT STATE-->\npan = {self.ptzf.pan}\ntilt = {self.ptzf.tilt}\ntemperature = {self.temperature}\nvoltage = {self.voltage}')

    def set_limits(self,max_pan = None,max_tilt = None,max_zoom = None,max_focus = None):
        if max_pan != None:
            self.max_pan = max_pan
            # print(f'max_pan = {self.max_pan}')
            # self.pan_k = 360/self.max_pan
            self.pan_k = 360 /58319
        if max_tilt != None:
            self.max_tilt = max_tilt
            # print(f'max_tilt = {self.max_tilt}')
            # self.tilt_k = 180/self.max_tilt
            self.tilt_k = 180 /29159
        if max_zoom != None:
            self.max_zoom = max_zoom
            # self.zoom_k = 100/self.max_zoom
        if max_focus != None:
            self.max_focus = max_focus
            # self.focus_k = 100/self.max_focus
    def get_limits_req(self):
        '''Делает запрос максимальных значений параметров
        и считывает ответ
        '''
        req = [0]*7
        req[0] = 255
        req[1] = self.pelco_addr
        req[3] = 123
        req[6] = pelco_check_sum_calc(req)
        # print(bytes(req))
        done = False
        try:
            self.udp_soc.sendto(bytes(req),(self.ip,self.udp_port))
            for i in range(4):
                data, remote_addr = self.udp_soc.recvfrom(1024)
                # pelco_de_parse(data).print()
                report = pelco_de_parse(data)
                if report.name == 'max_pan':
                    self.set_limits(max_pan=report.value)
                elif report.name == 'max_tilt':
                    self.set_limits(max_tilt=report.value)
                elif report.name == 'max_zoom':
                    self.set_limits(max_zoom=report.value)
                elif report.name == 'max_focus':
                    self.set_limits(max_focus=report.value)
                done = True
        except Exception as e:
            print(e.args)
        return done
    def get_temp_req(self):
        '''Запрос текущей температуры'''
        req = [0] * 7
        req[0] = 255
        req[1] = self.pelco_addr
        req[3] = 145 #/x91
        req[6] = pelco_check_sum_calc(req)
        done = False
        try:
            self.udp_soc.sendto(bytes(req),(self.ip,self.udp_port))
            data, remote_addr = self.udp_soc.recvfrom(1024)
            # pelco_de_parse(data).print()
            report = pelco_de_parse(data)
            if report.name =='temp':
                self.temperature = report.value
                done = True
        except Exception as e:
            print(e.args)
        return done
    def get_volt_req(self):
        '''Запрос напряжения питания'''
        req = [0] * 7
        req[0] = 255
        req[1] = self.pelco_addr
        req[3] = 155 #/x9B
        req[6] = pelco_check_sum_calc(req)
        done = False
        try:
            self.udp_soc.sendto(bytes(req),(self.ip,self.udp_port))
            data, remote_addr = self.udp_soc.recvfrom(1024)
            # pelco_de_parse(data).print()
            report = pelco_de_parse(data)
            if report.name =='volt':
                self.voltage = report.value
                done = True
        except Exception as e:
            print(e.args)
        return done
    def get_all_coords_req(self):
        '''Делает запрос всех текущих координат
        и считывает ответ
        '''
        req = [0]*7
        req[0] = 255
        req[1] = self.pelco_addr
        req[3] = 121 #\x79
        req[6] = pelco_check_sum_calc(req)
        # print(bytes(req))
        done = False
        changed = False
        try:
            self.udp_soc.sendto(bytes(req),(self.ip,self.udp_port))
            for i in range(4):
                data, remote_addr = self.udp_soc.recvfrom(1024)
                # pelco_de_parse(data).print()
                report = pelco_de_parse(data)
                if report.name == 'raw_pan':
                    new_pan = report.value*self.pan_k
                    if self.ptzf.pan != new_pan:
                        changed = True
                    self.ptzf.pan = new_pan
                elif report.name == 'raw_tilt':
                    new_tilt = report.value*self.tilt_k
                    if self.ptzf.tilt !=new_tilt:
                        changed = True
                    self.ptzf.tilt = new_tilt
                elif report.name == 'raw_zoom':
                    self.ptzf.zoom = report.value*self.zoom_k
                elif report.name == 'raw_focus':
                    self.ptzf.focus = report.value*self.focus_k
                done = True
            self.update_geo_pan_tilt()
        except Exception as e:
            print(e.args)
        return done, changed

    def get_pan_req(self):
        '''Делает запрос всех текущих координат
        и считывает ответ
        '''
        req = [0]*7
        req[0] = 255
        req[1] = self.pelco_addr
        req[3] = 81 #\x51
        req[6] = pelco_check_sum_calc(req)
        # print(bytes(req))
        done = False
        try:
            self.udp_soc.sendto(bytes(req),(self.ip,self.udp_port))
            data, remote_addr = self.udp_soc.recvfrom(1024)
            # pelco_de_parse(data).print()
            report = pelco_de_parse(data)
            if report.name == 'raw_pan':
                self.ptzf.pan = report.value*self.pan_k
                done = True
                self.update_geo_pan_tilt()
        except Exception as e:
            print(e.args, ', get pan req')
        return done
    def get_tilt_req(self):
        '''Делает запрос всех текущих координат
        и считывает ответ
        '''
        req = [0]*7
        req[0] = 255
        req[1] = self.pelco_addr
        req[3] = 83 #\x53
        req[6] = pelco_check_sum_calc(req)
        # print(bytes(req))
        done = False
        try:
            self.udp_soc.sendto(bytes(req),(self.ip,self.udp_port))
            data, remote_addr = self.udp_soc.recvfrom(1024)
            # pelco_de_parse(data).print()
            report = pelco_de_parse(data)
            if report.name == 'raw_tilt':
                self.ptzf.tilt = report.value*self.tilt_k
                done = True
                self.update_geo_pan_tilt()
        except Exception as e:
            print(e.args, ', get tilt req')
        return done

    def cont_move(self,pan_sp = 0, tilt_sp = 0):

        try:
            self.udp_soc.sendto(self.base_pelco.build_move_cmd(pan_sp,tilt_sp),(self.ip, self.udp_port))
        except Exception as e:
            print(e.args)

    def goto_pan_req(self,pan):
        raw_pan = int(pan/self.pan_k)
        b_pan = raw_pan.to_bytes(2,'big')
        req = [0]*7
        req[0] = 255
        req[1] = self.pelco_addr
        req[3] = 0x71
        req[4] = b_pan[0]
        req[5] = b_pan[1]
        req[6] = pelco_check_sum_calc(req)
        # print(bytes(req))
        done = False
        try:
            self.udp_soc.sendto(bytes(req),(self.ip,self.udp_port))
            # data, remote_addr = self.udp_soc.recvfrom(1024)
            # print(data)
            # pelco_de_parse(data).print()
            # report = pelco_de_parse(data)
            # if report.name == 'raw_tilt':
            #     self.ptzf.tilt = report.value*self.tilt_k
            #     done = True
        except Exception as e:
            print(e.args, ', set pan req')
        return done
    def goto_tilt_req(self,tilt):
        raw_tilt = int(tilt/self.tilt_k)
        b_tilt = raw_tilt.to_bytes(2,'big')
        req = [0]*7
        req[0] = 255
        req[1] = self.pelco_addr
        req[3] = 0x73
        req[4] = b_tilt[0]
        req[5] = b_tilt[1]
        req[6] = pelco_check_sum_calc(req)
        # print(bytes(req))
        done = False
        try:
            self.udp_soc.sendto(bytes(req),(self.ip,self.udp_port))
            data, remote_addr = self.udp_soc.recvfrom(1024)
            # print(data)
            # pelco_de_parse(data).print()
            # report = pelco_de_parse(data)
            # if report.name == 'raw_tilt':
            #     self.ptzf.tilt = report.value*self.tilt_k
            #     done = True
        except Exception as e:
            print(e.args, ', goto tilt req')
        return done
class Pelco_base_constructor():
    '''Конструктор базовой команды оригинального протокола Pelco-D'''
    def __init__(self,addr = 1):

        self.address = addr
        self.sense = False
        self.sense_mask = 0b10000000
        self.reserv1 = False
        self.reserv1_mask = 0b01000000
        self.reserv2 = False
        self.reserv2_mask = 0b00100000
        self.auto_man_scan = False
        self.auto_man_scan_mask = 0b00010000
        self.camera_on_off = False
        self.camera_on_off_mask = 0b00001000
        self.iris_close = False
        self.iris_close_mask = 0b00000100
        self.iris_open = False
        self.iris_open_mask = 0b00000010
        self.focus_near = False
        self.focus_near_mask = 0b00000001
        self.focus_far = False
        self.focus_far_mask = 0b10000000
        self.zoom_wide = False
        self.zoom_wide_mask = 0b01000000
        self.zoom_tele = False
        self.zoom_tele_mask = 0b00100000
        self.down = False
        self.down_mask = 0b00010000
        self.up = False
        self.up_mask = 0b00001000
        self.left = False
        self.left_mask = 0b00000100
        self.right = False
        self.right_mask = 0b00000010
        self.pan_speed = 0
        self.tilt_speed = 0
        self.max_speed = 0x3f

        # self.command1_list = [self.sense, self.reserv1,self.reserv2,self.auto_man_scan,self.camera_on_off,self.iris_close,self.iris_open,self.focus_near]
        self.command2_list = [self.focus_far,self.zoom_wide, self.zoom_tele,self.down,self.up,self.left,self.right]
        self.masks = [0b10000000,0b01000000,0b00100000,0b00010000,0b00001000,0b00000100,0b00000010,0b00000001]

    def build_move_cmd(self,pan_sp, tilt_sp):
        if pan_sp>0:
            self.left = False
            self.right = True
        elif pan_sp<0:
            self.left = True
            self.right = False
        else:
            self.left = False
            self.right = False
        if tilt_sp>0:
            self.up = True
            self.down = False
        elif tilt_sp <0:
            self.up = False
            self.down = True
        else:
            self.up = False
            self.down = False
        self.pan_speed = int((abs(pan_sp)/100)*self.max_speed)
        self.tilt_speed = int((abs(tilt_sp) / 100) * self.max_speed)
        return self.build_cmd()


    def build_cmd(self):
        # print(f'focus near {self.focus_near}')
        command1_list = [self.sense, self.reserv1, self.reserv2, self.auto_man_scan,
                           self.camera_on_off, self.iris_close, self.iris_open, self.focus_near]
        # self.command1_list[5] = True
        # print(command1_list)
        self.command2_list = [self.focus_far, self.zoom_wide, self.zoom_tele, self.down,
                           self.up, self.left, self.right]
        cmd = [0]*7
        cmd[0] = 0xff
        cmd[1] = self.address
        for i, parameter in enumerate(command1_list):
            # print(f'parameter № {i} is {parameter}')
            if parameter:
                cmd[2] = cmd[2] | self.masks[i]
            else:
                cmd[2] = cmd[2] & (~self.masks[i])
            # print(hex(cmd[2]))
        for i, parameter in enumerate(self.command2_list):
            if parameter:
                cmd[3] = cmd[3] | self.masks[i]
            else:
                cmd[3] = cmd[3] & (~self.masks[i])
            # print(hex(cmd[3]))
        cmd[4] = self.pan_speed
        cmd[5] = self.tilt_speed
        cmd[6] = pelco_check_sum_calc(cmd)
        return bytes(cmd)


def pelco_check_sum_calc(message):
    '''Функция для вычисления контрольной суммы PELCO'''
    ch_s = 0
    for i in range(1,6):
        ch_s+=message[i]
        # print(message[i])
    ch_s%=256
    return ch_s

def pelco_check(message):
    '''Проверка формата команды и контрольной суммы'''
    valid = False
    m_begin = message.find(b'\xff')
    mes = message[m_begin:m_begin+7]
    if len(mes) == 7:
        if pelco_check_sum_calc(mes) == mes[6]:
            valid = True

    # print(mes, ',', len(mes), ', ', valid)
    return valid, message

def pelco_de_parse(message):
    '''Разбор дополнительных команд'''
    valid, mes = pelco_check(message)
    # print(mes[4:6])
    param = Param_pair()
    if valid:
        if mes[2:4]==b'\x00\x61':
            param.name = 'raw_pan'
            param.value = int.from_bytes(mes[4:6],'big',signed=False)
        elif mes[2:4]==b'\x00\x63':
            param.name = 'raw_tilt'
            param.value = int.from_bytes(mes[4:6],'big',signed=False)
        elif mes[2:4]==b'\x00\x65':
            param.name = 'max_pan'
            param.value = int.from_bytes(mes[4:6],'big',signed=False)
        elif mes[2:4]==b'\x00\x67':
            param.name = 'max_tilt'
            param.value = int.from_bytes(mes[4:6], 'big', signed=False)
        elif mes[2:4]==b'\x00\x69':
            param.name = 'raw_zoom'
            param.value = int.from_bytes(mes[4:6], 'big', signed=False)
        elif mes[2:4]==b'\x00\x6b':
            param.name = 'raw_focus'
            param.value = int.from_bytes(mes[4:6], 'big', signed=False)
        elif mes[2:4]==b'\x00\x6d':
            param.name = 'max_zoom'
            param.value = int.from_bytes(mes[4:6],'big',signed=False)
        elif mes[2:4]==b'\x00\x6F':
            param.name = 'max_focus'
            param.value = int.from_bytes(mes[4:6],'big',signed=False)
        elif mes[2:4]==b'\x00\xA1':
            param.name = 'temp'
            param.value = int.from_bytes(mes[5:6],'big',signed=True)
        elif mes[2:4]==b'\x00\xAB':
            param.name = 'volt'
            param.value = int.from_bytes(mes[4:6],'big',signed=False)/100
    return param



class Mp_dev_interface:
    def __init__(self):
        self.q_in = mp.Queue(5)
        self.q_out = mp.Queue(5)
    def push_cmd_to_dev(self, cmd):
        self.q_in.put(cmd)
    def push_rep_from_dev(self,cmd):
        self.q_out.put(cmd)

    def get_rep_from_dev(self):
        got_cmd = False
        if not(self.q_out.empty()):
            got_cmd = True
            cmd = self.q_out.get()
        else:
            cmd = None
        return got_cmd,cmd

    def get_cmd_to_dev(self):
        got_cmd = False
        if not (self.q_in.empty()):
            got_cmd = True
            cmd = self.q_in.get()
        else:
            cmd = None
        return got_cmd, cmd

# class YOLO_processor:
#     def __init__(self):
#         self.interface = Mp_dev_interface()
#         self.yolo_net = YOLO()
#
#     def work(self):
#         while True:
#             got, img = self.interface.get_cmd_to_dev()
#             if got:
#                 detections = self.yolo_net.detect(img)
#                 self.interface.push_rep_from_dev()

# processors = []
# for i in range(5):
#     processors.append(YOLO_processor)



class Pelco_device_controller:
    def __init__(self,ip = '192.168.0.93',port = 6000):
        self.device_ip = ip
        self.device_port = port
        self.q_interface = Mp_dev_interface()
        self.telemetry_timeout = 0.1

    def device_work(self):
        device = Pelco_de_device(self.device_ip,self.device_port)
        run = True
        move = True
        move_timeout = 0.3
        move_last = time.time()
        last_telemetry_report = 0
        last_not_operative_update = 0.0
        not_operative_update_timeout = 5.0
        while run:
            now = time.time()
            got, cmd = self.q_interface.get_cmd_to_dev()
            if got:
                print(time.time())
                cmd.print()
                if cmd.name == 'pan_tilt_ctl':
                    device.cont_move(cmd.value[0],cmd.value[1])
                    move = True
                    move_last = time.time()
                elif cmd.name == 'pan_tilt_goto':
                    nat_pan,nat_tilt = device.convert_pan_tilt_geo_to_native(cmd.value[0],cmd.value[1])
                    device.goto_pan_req(nat_pan)
                    device.goto_tilt_req(nat_tilt)
                    move = True
                    move_last = time.time()
            if (now - last_not_operative_update)>not_operative_update_timeout:
                device.get_limits_req()
                device.get_temp_req()
                device.get_volt_req()
                last_not_operative_update = now
                self.q_interface.push_rep_from_dev(Command('temp_volt',[device.temperature,device.voltage]))
            if (now - last_telemetry_report)>self.telemetry_timeout:
                got, changed = device.get_all_coords_req()
                ready_to_shot = False
                if (got)and(not changed):
                    ready_to_shot = True
                if got:
                    if changed:
                        ready_to_shot = False

                    # print(time.time())
                    # print(f'got telemetry: {got}, Movement = {changed}')
                    last_telemetry_report = now
                    self.q_interface.push_rep_from_dev(Command('pan_tilt',[device.ptzf_geo.pan,device.ptzf_geo.tilt,ready_to_shot]))
                # device.print()
    def run_separately(self):
        proc = mp.Process(target=self.device_work,args=())
        proc.start()

class Command:
    def __init__(self,name,val):
        self.name = name
        self.value = val
    def print(self):
        print(f'CMD: {self.name}, Value: {self.value}')


if __name__ == '__main__':
    # ch_s = pelco_check_sum_calc(b'\xFF\x01\x00\xA1\x00\x39\xDB')
    # print(hex(ch_s))
    # # pelco_check(b'\xFF\x01\x00\xA1\x00\x39\xDB')
    #
    # pan_mes = b'\xff\x01\x00\x61\x71\xe8\xbb'
    #
    # pelco_de_parse(pan_mes).print()
    #
    # temp_mes = b'\xff\x01\x00\xA1\x00\x39\xdb'
    #
    # pelco_de_parse(temp_mes).print()
    bic_opu = Pelco_de_device('192.168.0.93',6000)
    bic_opu.get_limits_req()

    bic_opu.get_temp_req()
    bic_opu.get_volt_req()
    bic_opu.get_all_coords_req()
    bic_opu.print()
    print(bic_opu.pan_k)
    print(bic_opu.tilt_k)


    # bic_opu.goto_pan_req(180)
    # bic_opu.goto_tilt_req(180)
    # time.sleep(3)
    bic_opu.get_all_coords_req()
    bic_opu.print()

    cons = Pelco_base_constructor()

    time.sleep(1)


    # while True:
    #     bic_opu.get_pan_req()
    #     bic_opu.get_tilt_req()
    #     bic_opu.print()
    #     time.sleep(1)

    print('_________________________________')

    con = Pelco_base_constructor()
    con.focus_near = True
    print(f'reread = {con.focus_near}')

    cmd = con.build_cmd()
    # print(const.command1_list)
    print(cmd)
    # mask = 0b10000000
    # for i in range(8):
    #     mask = mask >> 1
    #     print(mask)
    #     print(mask.to_bytes(1,'big'))


    # dir = 1
    # speed = 30
    # for i in range(10):
    #     bic_opu.cont_move(speed*dir,0)
    #     time.sleep(5)
    #     bic_opu.cont_move(0,0)
    #     time.sleep(0.5)
    #     dir*=(-1)
    # bic_opu.cont_move(0, 0)

    opuer = Pelco_device_controller()
    opuer.run_separately()
    while True:
        # print(opuer.q_interface.q_in.qsize())
        # print(opuer.q_interface.q_out.qsize())
        got,cmd = opuer.q_interface.get_rep_from_dev()
        # print(got)
        if got:
            print(time.time())
            cmd.print()
        time.sleep(0.1)
