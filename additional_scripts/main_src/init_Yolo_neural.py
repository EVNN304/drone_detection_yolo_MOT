from additional_scripts.main_src.yolov8_lib_2 import collect_single_step_dets

from additional_scripts.save_modules.Module_save_detection import *






class Yolo_inits:
    def __init__(self, q_control: mp.Queue, q_from_neuro: mp.Queue, self_addres=None, dest_address=None, saved_mode=None, names_files=None, name_folder="TEST_SAVE_SERVER_IR_PRIOZERSK"):
        self.pix_x, self.pix_y = 30, 30
        self.name_window = "w"
        self.name_folder = name_folder
        self.self_addres, self.dest_address = self_addres, dest_address
        self.q_control, self.q_from_neuro = q_control, q_from_neuro
        self.saved_mode = saved_mode
        self.udp_caster = None
        self.names_files = names_files


        if self.saved_mode != None:
            self.saved_mode = mp.JoinableQueue(1)
            self.run_saver()

    def set_name_window(self, val):
        self.name_window += val

    def set_name_save(self, val:str):
        self.names_files = val

    def set_name_folder(self, val):
        self.name_folder = val

    def set_self_addres(self, addr):
        self.self_addres = addr

    def set_dest_addres(self, addr):
        self.dest_address = addr


    def set_pix_x(self, val):
        self.pix_x = val


    def set_pix(self, val):
        self.pix_y = val



    def run_saver(self):
        process_save_det = Save_detect(self.saved_mode, )
        process_save_det.set_names_file(self.names_files)
        process_save_det.set_path_save(self.name_folder)
        process_save_det.main_start_process()


    def run_nets(self):
        neuro_data_collector = mp.Process(target=self.collect_n_cast_neuro, args=(self.q_control,  self.q_from_neuro, self.saved_mode))
        neuro_data_collector.start()



    def collect_n_cast_neuro(self, q_control: mp.Queue, q_dets: mp.Queue,  saved_mode):


        timer = Timer()
        while True:
            all_detections = []
            args_control = q_control.get()
            timer.start()
            frame_id = args_control[0]
            n_subframes = args_control[1]
            meta = args_control[2]
            frame = args_control[3]
            x_az, y_el, size_obj, id_object = args_control[4], args_control[5], args_control[6], args_control[7]
            fr = copy.deepcopy(frame)
            filtered_dets = collect_single_step_dets(q_dets, frame_id, n_subframes)
            h, w = fr.shape

            dist, filter_detect = [], []
            if filtered_dets:

                for num, obj_tr in enumerate(id_object):
                    id_track_obj, cord_x_y_w_h = obj_tr[0], obj_tr[1]
                    for det_n, det in enumerate(filtered_dets):
                        x, y = det.get_int_center()
                        if (abs(x - cord_x_y_w_h[0]) <= self.pix_x) and (abs(y - cord_x_y_w_h[1]) <= self.pix_y):
                            det.id = id_track_obj
                            filter_detect.append(det)
                            det.draw(frame)
                            cord_calc = cover_pt_by_area((x, y), area_w_h=[288, 288], limit_box=[0, 0, w, h])
                            cv.imshow("fragmet", frame[cord_calc[1]:cord_calc[3], cord_calc[0]:cord_calc[2]])
                            cv.waitKey(1)
                            break


            for i, k in enumerate(id_object):
                cv.circle(frame, (k[1][0], k[1][1]), 170, (0, 0, 0), 8)
                cv.putText(frame, f'{k[0]}', (k[1][0] - (k[1][2] // 2) - 30, k[1][1] - (k[1][3] // 2)), 3, 1.3, (0, 0, 0), 2)

            if saved_mode != None:
                if saved_mode.empty():
                    if filter_detect != []:
                        saved_mode.put([fr, filter_detect, True])
                    if filter_detect == [] and id_object != []:
                        saved_mode.put([fr, id_object, False])




            cv.imshow(self.name_window, cv.resize(frame, (700, 700)))
            cv.waitKey(1)