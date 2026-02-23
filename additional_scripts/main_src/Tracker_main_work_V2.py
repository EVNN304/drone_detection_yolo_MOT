from additional_scripts.main_src.tcp_udp_interfaces.v_net_control import Simple_udp_dialog
from additional_scripts.main_src.prep_images.image_prepare_lib import Contour_finder
from additional_scripts.main_src.tracking_lib_22_v0 import *
import multiprocessing as mp
import copy
from additional_scripts.main_src.tcp_udp_interfaces.v_net_control import run_tcp_v_server, CMD_list
from additional_scripts.main_src.geometry_and_math.geometry_lib import cover_pt_by_area
from additional_scripts.main_src.protocols.toolset import Command







class Tracker_main:

    def __init__(self, size_img, q_in: mp.Queue, q_control: mp.Queue, q_to_neuro: mp.Queue):
        self.seize_img, self.q_in,  self.q_control, self.q_to_neuro = size_img, q_in, q_control, q_to_neuro
        self.size_w_nn, self.size_h_nn, self.resize_w, self.resize_h, self.p_w, self.p_h = 288, 288, 500, 500, 0.8, 0.8
        self.name_window = "tracker"

    def set_name_window(self, val):
        self.name_window += val

    def set_size_w_nn(self, val):
        self.size_w_nn = val

    def set_size_h_nn(self, val):
        self.size_h_nn = val

    def set_resize_w(self, val):
        self.resize_w = val

    def set_resize_h(self, val):
        self.resize_h = val

    def set_p_w(self, val):
        self.p_w = val

    def set_p_h(self, val):
        self.p_h = val

    def start_tracking(self):
        q_from_move_to_track = mp.Queue(1)

        pr_movement = mp.Process(target=self.find_contours_from_stream,
                                 args=(self.seize_img, self.q_in, q_from_move_to_track))  # q_from_move_to_track))
        pr_movement.daemon = True

        pr_tracking = mp.Process(target=self.track_objects_from_stream,
                                 args=(q_from_move_to_track, self.q_control, self.q_to_neuro))
        # pr_tracking.daemon = True
        pr_movement.start()
        pr_tracking.start()
        print("RUNNER_PROCESS_TRACK")




    def find_contours_from_stream(self, image_size, q_in: mp.Queue, q_out: mp.Queue):
        movement_fps_timer = Timer()
        movement_fps_timer.start()
        contour_finder = Contour_finder(image_size[1], image_size[0])
        run = True
        while run:

            image, meta, frame_count, cmd_list = q_in.get()
            h, w = image.shape
            image_copy = copy.deepcopy(image)


            #valid, contour_boxes = contour_finder.process_step_directed(image, [meta.az, meta.el])
            valid, contour_boxes = contour_finder.process_step_directed(image, [meta.az, meta.el])
            elapsed = movement_fps_timer.stop()
            movement_fps_timer.start()
            print(f'movements: {len(contour_boxes)}, fps = {1 / elapsed if elapsed != 0 else 0}')
                #if q_out:
            #if q_out.empty():
            q_out.put([valid, meta, contour_boxes, image_copy, frame_count, 0, h, w])

    # def track_from_stream()
    def track_objects_from_stream(self, q_in, q_control, q_to_neural):

        tracker_fps_timer = Timer()
        tracker_fps_timer.start()
        search_radius = 250  # Радиус для поиска
        size_mismatch = 0.4  # Допустимая ошибка по размеру (в долях)
        tracker = Tracker_m(BOX_STYLE_CVT)
        tracker.initial_search_radius = search_radius
        tracker.size_mismatch = size_mismatch
        tracker.set_alg(1)
        tracker.angle_thresh = 35
        tracker.flush_timeout = 5
        *obj, h, w = q_in.get()
        run = True
        test_image = np.zeros((h, w), dtype=np.uint8)
        last_cnn = 0
        kernel = np.ones((4, 4), dtype=np.float32) / 17
        while run:
            lst_track = []
            # cv.multiply(test_image, 0, test_image)
            valid, meta, contour_boxes, neuro_image, frame_count, cnn, h, w = q_in.get()
            h, w = neuro_image.shape
            if last_cnn != cnn:
                last_cnn = cnn
                test_image = np.zeros((h, w), dtype=np.uint8)
            cv.multiply(test_image, 0, test_image)
            if valid:
                tracker.update_points_frame_pos(meta)
                stamp = meta.timestamp
                tracker.update_n_process(contour_boxes, stamp, meta)
                elapsed = tracker_fps_timer.stop()
                tracker_fps_timer.start()

                lst_track = [[k, tracker.points_db.get_det_by_index(
                    tracker.tracks_db.get_track_by_index(k).get_last_point_id()).centered_box] for i, k in
                             enumerate(tracker.get_fresh_track_ids(5))]

                tracker.draw_fresh_tracks(test_image, 4, True)

                cv.imshow(self.name_window, cv.resize(test_image, (1000, 1000)))
                cv.waitKey(1)
                print(f'tracker_fps: {1 / elapsed if elapsed != 0 else 0}')

            print(f"list_track: {lst_track} len:{len(lst_track)}")

            if ((q_to_neural.qsize() == 0) & (q_control.qsize() == 0)):

                for i, k in enumerate(lst_track):
                    if ((k[1][2] / self.size_w_nn) < self.p_w) and ((k[1][3] / self.size_h_nn) < self.p_h):
                        obl_detection = cover_pt_by_area((k[1][0], k[1][1]), area_w_h=[self.size_w_nn, self.size_h_nn], limit_box=[0, 0, w, h])  # Получаем координаты нарезаемой области
                        print(f"shapes_{self.size_w_nn}x{self.size_h_nn}", neuro_image[obl_detection[1]:obl_detection[3], obl_detection[0]:obl_detection[2]].shape)
                    else:
                        obl_detection = cover_pt_by_area((k[1][0], k[1][1]), area_w_h=[self.resize_w, self.resize_h], limit_box=[0, 0, w, h])  # Получаем координаты нарезаемой области
                        print(f"shapes_{self.resize_w}x{self.resize_h}", neuro_image[obl_detection[1]:obl_detection[3], obl_detection[0]:obl_detection[2]].shape)
                    q_control.put([frame_count, 1, meta, neuro_image, k[1][0], k[1][1], k, lst_track])
                    #q_to_neural.put((cv.cvtColor(cv.filter2D(neuro_image[obl_detection[1]:obl_detection[3], obl_detection[0]:obl_detection[2]], -1, kernel), cv.COLOR_GRAY2RGB), [obl_detection[0], obl_detection[1], obl_detection[2], obl_detection[3]], [frame_count, len([[obl_detection[0], obl_detection[1], obl_detection[2], obl_detection[3]]]), 1, cnn]))
                    q_to_neural.put((cv.cvtColor(neuro_image[obl_detection[1]:obl_detection[3], obl_detection[0]:obl_detection[2]], cv.COLOR_GRAY2RGB), [obl_detection[0], obl_detection[1], obl_detection[2], obl_detection[3]], [frame_count, len([[obl_detection[0], obl_detection[1], obl_detection[2], obl_detection[3]]]), 1, cnn]))
