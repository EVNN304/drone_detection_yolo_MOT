import numpy as np
import time
import cv2 as cv
import additional_scripts.main_src.geometry_and_math.geometry_lib as glib
from additional_scripts.main_src.protocols.toolset import Timer, Codec_mini

BOX_STYLE_CENTERED = 0
BOX_STYLE_CVT = 1
BOX_STYLE_P1P2 = 2
BOX_STYLE_CVT_CONTOUR = 3





def box_cvt_cent2corners(box):
    pt_lt = (int(box[0]-box[2]/2),int(box[1]-box[3]/2))
    pt_rb = (int(box[0] + box[2] / 2), int(box[1] + box[3] / 2))
    return (pt_lt[0],pt_lt[1],pt_rb[0],pt_rb[1])

def box_cvt_2corners(box):
    pt_lt = (int(box[0]),int(box[1]))
    pt_rb = (int(box[0] + box[2]), int(box[1] + box[3]))
    return (pt_lt[0],pt_lt[1],pt_rb[0],pt_rb[1])

def check_boxes_sizes(box1,box2,mismatch = 0.3,box_style = BOX_STYLE_CENTERED):
    '''
    Функция проверяет совпадение размеров с точностью до заданной в аргументе mismatch
    '''
    if box_style == BOX_STYLE_CENTERED:
        w1 = box1[2]
        w2 = box2[2]
        h1 = box1[3]
        h2 = box2[3]
    else:
        w1 = box1[2]-box1[0]
        w2 = box2[2]-box2[0]
        h1 = box1[3]-box1[1]
        h2 = box2[3]-box2[1]

    x_check = ((w1-w1*mismatch)<=w2)&((w1+w1*mismatch)>=w2)
    y_check = ((h1 - h1 * mismatch) <= h2) & ((h1 + h1 * mismatch) >= h2)
    return x_check&y_check




class Deep_storage:
    '''
    Класс для удобного обращения к текущим и предыдущим индексам точек
    '''
    def __init__(self, deep = 3):
        self.deep = deep
        self.storage = []
        for i in range(deep):
            self.storage.append([])
        self.current_index = -1

    def put_current_data(self,new_data):
        self.current_index+=1
        self.current_index%=self.deep
        self.storage[self.current_index] = new_data

    def get_data(self,step_from_current):
        '''
        функция возвращает элемент истории на заданное количество шагов от текущей записи
        '''
        req_id = (self.current_index+step_from_current)%self.deep
        return self.storage[req_id]
class Point_2d:
    def __init__(self):
        self.xy


class Tracker_m:
    '''
    Класс для построения траекторий
    '''
    def __init__(self, detections_style = BOX_STYLE_CVT):
        self.default_detection_style = detections_style
        self.points_db = Detections_m_list(10000)
        self.tracks_db = Tracks_list(1000,20)
        self.points_history = Deep_storage(3) #Инициализируем хранилище индексов точек для завязки по трем точкам
        self.flush_timeout = 3.0
        self.initial_search_radius = 400
        self.size_mismatch = 0.4
        self.current_stamp = 0.0
        self.prev_vector = glib.Vect()
        self.test_vector = glib.Vect()
        self.angle_thresh = 30
        self.algorithm = 0
        self.search_radius_k = 0.8

        self.active_single_track = -1
        self.color_fresh = (200,0,0)
        self.color_old = (100,100,100)
        self.color_checked = (0,200,0)
        self.meta_converter = None
        self.draw_converted = False
        self.codec = Codec_mini()


    def get_active_tracks_ids(self):
        return self.tracks_db.active_tracks_ids
    def get_track_by_id(self,idx):
        return self.tracks_db.get_track_by_index(idx)

    def get_detection_by_idx(self,idx):
        return self.points_db.get_det_by_index(idx)

    def set_converter(self,converter:glib.Meta2meta_converter):
        self.meta_converter = converter
        self.meta_converter.set_aliases('self','export_to')
    def set_draw_converted(self, conv_flag):
        self.draw_converted = conv_flag

    def set_alg(self,alg):
        self.algorithm = alg

    def set_active_track(self,id):
        if id in self.tracks_db.active_tracks_ids:
            self.active_single_track = id
            return True
        else:
            self.active_single_track = -1
            return False

    def set_detection_style(self,new_style):
        self.default_detection_style = new_style

    def update_step(self,detections,stamp,image_meta:glib.Image_meta = None):
        current_step_detections = []
        for detection in detections:
            if self.default_detection_style == BOX_STYLE_CVT_CONTOUR:
                current_step_detections.append(self.points_db.insert_new(cv.boundingRect(detection),stamp,BOX_STYLE_CVT,image_meta))
            else:
                current_step_detections.append(self.points_db.insert_new(detection,stamp,self.default_detection_style,image_meta))
        self.points_history.put_current_data(current_step_detections)
        self.current_stamp = stamp

    def update_points_frame_pos(self,new_meta:glib.Image_meta):
        self.points_db.update_all_px_pos_to_meta(new_meta)
    def process_current_step(self):
        check_single = False
        if (self.active_single_track >=0) & (self.active_single_track in self.tracks_db.active_tracks_ids):
            # algorithm =2
            algorithm = self.algorithm
            check_single = True
            # check_single = False
        else:
            algorithm = self.algorithm

        if algorithm == 0:
            test_point_to_predicted_ranges = []  # Расстояния между прогнозной и первой областью (Инициализация)
            triples = []    #тройки
            check_r = self.initial_search_radius
            timer = Timer()
            timer.start()
            step_count = 0

            for det_idx_0 in self.points_history.get_data(0):
                print(f'step{step_count},{timer.stop()}')

                step_count += 1
                # if step_count>15:
                #     break
                current_det = self.points_db.get_det_by_index(det_idx_0)
                for det_idx_m1 in self.points_history.get_data(-1):
                    prev_det = self.points_db.get_det_by_index(det_idx_m1)
                    # Создаем проверочный критерий и далее делаем ряд проверок
                    criterion = True
                    # criterion&=((((step_count%160)**2+(step_count%160-30)**2)**0.5 + ((step_count%160-40)**2+(step_count%180-40)**2)**0.5)>500)
                    # criterion&=(glib.pt2pt_2d_range([(step_count%160),(step_count%160-30)],[(step_count%130+60.0),(step_count%160-30.2)])<=1000)
                    p2p_range = glib.pt2pt_2d_range(current_det.get_center(),
                                                      prev_det.get_center())
                    criterion &= ( p2p_range< check_r)  # проверка расстояния
                    criterion &= check_boxes_sizes(current_det.centered_box, prev_det.centered_box,self.size_mismatch)  # проверка размеров
                    if criterion:
                        shift_vector = [(current_det.centered_box[0] - prev_det.centered_box[0])/(current_det.stamp-prev_det.stamp),
                                        (current_det.centered_box[1] - prev_det.centered_box[1])/(current_det.stamp-prev_det.stamp)]
                        # v = p2p_range/(current_det.stamp-prev_det.stamp)
                        # shift_vector[0] = shift_vector[0]*v
                        # shift_vector[1] = shift_vector[1]*v
                        predicted = [0, 0, prev_det.centered_box[2], prev_det.centered_box[3]]  # Инициализируем и заполняем предполагаемую область начала

                        # t_pred = prev_det.stamp - prev_prev_det.stamp
                        # predicted[0] = prev_det.centered_box[0] - t_pred*shift_vector[0]
                        # predicted[1] = prev_det.centered_box[1] - t_pred*shift_vector[1]
                        b1_idxs = []  # Индексы прошедших проверку точек (Инициализация)

                        for det_idx_m2 in self.points_history.get_data(-2):
                            prev_prev_det = self.points_db.get_det_by_index(det_idx_m2)
                            t_pred = prev_det.stamp - prev_prev_det.stamp
                            predicted[0] = prev_det.centered_box[0] - t_pred * shift_vector[0]
                            predicted[1] = prev_det.centered_box[1] - t_pred * shift_vector[1]
                            p2 = prev_det.get_center()
                            p1 = prev_prev_det.get_center()
                            p_t = current_det.get_center()
                            self.prev_vector.get_vect_by_pts(p1,p2)
                            self.test_vector.get_vect_by_pts(p2,p_t)


                            glib.get_v_angle(self.prev_vector,self.test_vector)
                            # Создаем проверочный критерий и далее делаем ряд проверок
                            criterion = True
                            criterion &= (glib.pt2pt_2d_range(prev_prev_det.get_center(),
                                                              (predicted[0],
                                                               predicted[1])) < self.initial_search_radius)  # проверка расстояния
                            criterion &= check_boxes_sizes(prev_prev_det.centered_box, prev_det.centered_box,self.size_mismatch)  # проверка размеров
                            criterion &= (prev_det.tr_id == prev_prev_det.tr_id) #проверка принадлежности к одной траектории
                            criterion &= (abs(glib.get_v_angle(self.prev_vector,self.test_vector)) < self.angle_thresh)


                            if criterion:


                                # cv.circle(image_color, (predicted[0], predicted[1]), (50), (0, 128, 255), 5)
                                range = glib.pt2pt_2d_range((predicted[0], predicted[1]), prev_prev_det.get_center())
                                triples.append([det_idx_0,det_idx_m1,det_idx_m2])
                                test_point_to_predicted_ranges.append(range)
            elapsed = timer.stop()
            # print(f'<<<<<<<<<<<<Внутри обработки: первичная провека троек, {elapsed}')

            # print('Gotcha!',len(test_point_to_predicted_ranges))
            # for i,triple in enumerate(triples):
            #     print(triple, test_point_to_predicted_ranges[i])

            # проходим по массиву троек и обновляем траектории жадным алгоритмом

            # def get_last_active_pos():
            #     self.ge
            new_tracks = []
            timer.start()
            for r in test_point_to_predicted_ranges:

                min_range = min(test_point_to_predicted_ranges)
                min_idx = test_point_to_predicted_ranges.index(min_range)
                triple_on_test = triples[min_idx]
                prev_tr_id = self.points_db.get_det_by_index(triple_on_test[1]).tr_id
                prev_prev_tr_id = self.points_db.get_det_by_index(triple_on_test[1]).tr_id
                if prev_tr_id == prev_prev_tr_id:   #Проверяем, что точки не были заняты в процессе обработки для другой траектории
                    vx, vy = self.calc_p2p_velocity(triple_on_test[1], triple_on_test[0])
                    if prev_tr_id == -1:
                        # new_track_id = self.tracks_db.get_empty_id()
                        new_track_id = self.tracks_db.activate_new_track()
                        new_tracks.append(new_track_id)
                        self.points_db.bind_to_track_by_index(triple_on_test[2],new_track_id)
                        self.tracks_db.update_track_by_index(new_track_id,triple_on_test[2],self.current_stamp)
                        self.tracks_db.update_track_by_index(new_track_id, triple_on_test[1],self.current_stamp)

                        self.tracks_db.update_track_by_index(new_track_id, triple_on_test[0],self.current_stamp,vx,vy)
                        for p_index in triple_on_test:
                            self.points_db.bind_to_track_by_index(p_index,new_track_id)
                        # track = self.tracks_db.get_track_by_index(new_track_id)
                        # print(f'Новая траектория: {new_track_id}')

                    elif self.tracks_db.get_track_by_index(prev_tr_id).last_update<self.current_stamp:
                        self.points_db.bind_to_track_by_index(triple_on_test[0],prev_tr_id) #Привязываем последнюю точку тройки к траектории
                        self.tracks_db.update_track_by_index(prev_tr_id,triple_on_test[0], self.current_stamp,vx,vy)
                        # print(f'Продолжение траектории: {prev_tr_id}')
                    # print(f'V = {vx},{vy}')

                # test_point_to_predicted_ranges[min_idx] = 10000
                # print(f'min: {min_range}, index: {min_idx}, triple: {triples[min_idx]}, track id: {tr_id}')

            #Проверяем, могут ли быть новые тройки продолжением имеющейся траектории
            for tr_id in new_tracks:
                new_track = self.tracks_db.get_track_by_index(tr_id)
                new_tr_first_p = self.points_db.get_det_by_index(new_track.tail[0])
                predict_stamp = self.points_db.get_det_by_index(new_track.tail[0]).stamp
                for act_track in self.tracks_db.active_tracks_ids:
                    track_on_test = self.tracks_db.get_track_by_index(act_track)
                    on_test_last_p = self.points_db.get_det_by_index(track_on_test.get_last_point_id())
                    criterion = False
                    if on_test_last_p.stamp<new_tr_first_p.stamp:
                        # dt = new_tr_first_p.stamp-on_test_last_p.stamp
                        # print(f'dt = {dt}, st1 = {new_tr_first_p.stamp}, st2 = {on_test_last_p.stamp}')
                        # test_p_x, test_p_y = on_test_last_p.get_center()
                        # print(f'test_p: [{test_p[0]},{test_p[1]}], v = {track_on_test.v_vect}')

                        try:
                            x, y = on_test_last_p.get_center()
                            dt = predict_stamp - on_test_last_p.stamp
                            v = track_on_test.v_vect
                            # print(f'dt = {dt}, st1 = {new_tr_first_p.stamp}, st2 = {on_test_last_p.stamp}')
                            # print(f'v: {v}')
                            px = x + v[0] * dt
                            py = y + v[1] * dt

                            # print(f'px,py: {px},{py}')
                            predicted = [px,py]

                            criterion = True
                            criterion&= glib.pt2pt_2d_range(predicted,new_tr_first_p.get_center())<self.initial_search_radius
                            # print(f'Vs: {track_on_test.v_vect}, {new_track.v_vect}')
                            # criterion&=glib.get_v_angle(track_on_test.v_vect,new_track.v_vect)<45
                            criterion&=check_boxes_sizes(on_test_last_p.centered_box,new_tr_first_p.centered_box,self.size_mismatch)
                            # criterion&=
                        except:
                            print('Ошибка в проверке продолжений')
                            criterion = False

                    if criterion:
                        track_on_test.update(new_track.tail[0],self.current_stamp,new_track.v_vect[0],new_track.v_vect[1])
                        track_on_test.update(new_track.tail[1],self.current_stamp,new_track.v_vect[0],new_track.v_vect[1])
                        track_on_test.update(new_track.tail[2],self.current_stamp,new_track.v_vect[0],new_track.v_vect[1])
                        self.points_db.get_det_by_index(new_track.tail[0]).tr_id = act_track
                        self.points_db.get_det_by_index(new_track.tail[1]).tr_id = act_track
                        self.points_db.get_det_by_index(new_track.tail[2]).tr_id = act_track

                        new_track.flush()
                        break


            elapsed = timer.stop()
            # print(f'<<<<<<<<<<<<Внутри обработки: окончательная обработка, {elapsed}')
            # print('angle_thresh', self.angle_thresh)

        elif algorithm == 1:
            test_point_to_predicted_ranges = []  # Расстояния между прогнозной и первой областью (Инициализация)
            triples = []  # тройки
            check_r = self.initial_search_radius
            timer = Timer()
            timer.start()
            step_count = 0

            if check_single:
                single_track = self.tracks_db.get_track_by_index(self.active_single_track)
                if self.current_stamp -  single_track.last_update < 0.3:
                    prev_check_list = [single_track.get_last_point_id()]
                    prev_prev_check_list = self.points_history.get_data(-2)
                else:
                    prev_check_list = self.points_history.get_data(-1)
                    prev_prev_check_list = self.points_history.get_data(-2)
                # pass
            else:
                prev_check_list = self.points_history.get_data(-1)
                prev_prev_check_list = self.points_history.get_data(-2)

            for det_idx_0 in self.points_history.get_data(0):
                # print(f'step{step_count},{timer.stop()}')
                step_count += 1
                # if step_count>15:
                #     break
                current_det = self.points_db.get_det_by_index(det_idx_0)
                for det_idx_m1 in prev_check_list:
                    prev_det = self.points_db.get_det_by_index(det_idx_m1)
                    # Создаем проверочный критерий и далее делаем ряд проверок
                    criterion = True
                    p2p_range = glib.pt2pt_2d_range(current_det.get_center(),
                                                    prev_det.get_center())
                    criterion &= (p2p_range < self.initial_search_radius)  # проверка расстояния
                    criterion &= check_boxes_sizes(current_det.centered_box, prev_det.centered_box,
                                                   self.size_mismatch)  # проверка размеров
                    if criterion:
                        shift_vector = [(current_det.centered_box[0] - prev_det.centered_box[0]) / (
                                    current_det.stamp - prev_det.stamp),
                                        (current_det.centered_box[1] - prev_det.centered_box[1]) / (
                                                    current_det.stamp - prev_det.stamp)]
                        predicted = [0, 0, prev_det.centered_box[2], prev_det.centered_box[
                            3]]  # Инициализируем и заполняем предполагаемую область начала

                        b1_idxs = []  # Индексы прошедших проверку точек (Инициализация)
                        check_r = max(p2p_range*self.search_radius_k,20)
                        # print(f'CHHHHHHHHHHHHHHHHHHHHEEECK {check_r}')
                        for det_idx_m2 in prev_prev_check_list:
                            prev_prev_det = self.points_db.get_det_by_index(det_idx_m2)
                            t_pred = prev_det.stamp - prev_prev_det.stamp
                            predicted[0] = prev_det.centered_box[0] - t_pred * shift_vector[0]
                            predicted[1] = prev_det.centered_box[1] - t_pred * shift_vector[1]
                            p2 = prev_det.get_center()
                            p1 = prev_prev_det.get_center()
                            p_t = current_det.get_center()
                            self.prev_vector.get_vect_by_pts(p1, p2)
                            self.test_vector.get_vect_by_pts(p2, p_t)

                            glib.get_v_angle(self.prev_vector, self.test_vector)
                            # Создаем проверочный критерий и далее делаем ряд проверок
                            criterion = True
                            criterion &= (glib.pt2pt_2d_range(prev_prev_det.get_center(),
                                                              (predicted[0],
                                                               predicted[
                                                                   1])) < check_r)  # проверка расстояния
                            criterion &= check_boxes_sizes(prev_prev_det.centered_box, prev_det.centered_box,
                                                           self.size_mismatch)  # проверка размеров
                            criterion &= (
                                        prev_det.tr_id == prev_prev_det.tr_id)  # проверка принадлежности к одной траектории
                            criterion &= (abs(glib.get_v_angle(self.prev_vector, self.test_vector)) < self.angle_thresh)
                            if criterion:
                                # print(f'angle = {abs(glib.get_v_angle(self.prev_vector, self.test_vector))}\n'
                                #       f'{p1},{p2},{p_t}')
                                # cv.circle(image_color, (predicted[0], predicted[1]), (50), (0, 128, 255), 5)
                                range = glib.pt2pt_2d_range((predicted[0], predicted[1]), prev_prev_det.get_center())
                                triples.append([det_idx_0, det_idx_m1, det_idx_m2])
                                test_point_to_predicted_ranges.append(range)
            elapsed = timer.stop()
            # print(f'<<<<<<<<<<<<Внутри обработки: первичная провека троек, {elapsed}')
            # print(f'angle_thresh = {self.angle_thresh}')

            # print('Gotcha!',len(test_point_to_predicted_ranges))
            # for i,triple in enumerate(triples):
            #     print(triple, test_point_to_predicted_ranges[i])

            # проходим по массиву троек и обновляем траектории жадным алгоритмом

            # def get_last_active_pos():
            #     self.ge
            new_tracks = []
            timer.start()
            for r in test_point_to_predicted_ranges:

                min_range = min(test_point_to_predicted_ranges)
                min_idx = test_point_to_predicted_ranges.index(min_range)
                triple_on_test = triples[min_idx]
                prev_tr_id = self.points_db.get_det_by_index(triple_on_test[1]).tr_id
                prev_prev_tr_id = self.points_db.get_det_by_index(triple_on_test[1]).tr_id
                if prev_tr_id == prev_prev_tr_id:  # Проверяем, что точки не были заняты в процессе обработки для другой траектории
                    vx, vy = self.calc_p2p_velocity(triple_on_test[1], triple_on_test[0])
                    if prev_tr_id == -1:
                        # new_track_id = self.tracks_db.get_empty_id()
                        new_track_id = self.tracks_db.activate_new_track()
                        new_tracks.append(new_track_id)
                        self.points_db.bind_to_track_by_index(triple_on_test[2], new_track_id)
                        self.tracks_db.update_track_by_index(new_track_id, triple_on_test[2], self.current_stamp)
                        self.tracks_db.update_track_by_index(new_track_id, triple_on_test[1], self.current_stamp)

                        self.tracks_db.update_track_by_index(new_track_id, triple_on_test[0], self.current_stamp, vx,
                                                             vy)
                        for p_index in triple_on_test:
                            self.points_db.bind_to_track_by_index(p_index, new_track_id)
                        # track = self.tracks_db.get_track_by_index(new_track_id)
                        # print(f'Новая траектория: {new_track_id}')

                    elif self.tracks_db.get_track_by_index(prev_tr_id).last_update < self.current_stamp:
                        self.points_db.bind_to_track_by_index(triple_on_test[0],
                                                              prev_tr_id)  # Привязываем последнюю точку тройки к траектории
                        self.tracks_db.update_track_by_index(prev_tr_id, triple_on_test[0], self.current_stamp, vx, vy)
                        # print(f'Продолжение траектории: {prev_tr_id}')
                    # print(f'V = {vx},{vy}')

                test_point_to_predicted_ranges[min_idx] = 10000
                # print(f'min: {min_range}, index: {min_idx}, triple: {triples[min_idx]}, track id: {tr_id}')

            # Проверяем, могут ли быть новые тройки продолжением имеющейся траектории
            p2p_ranges = []
            p2p_ids = []
            if (self.active_single_track >= 0) & (self.active_single_track in self.tracks_db.active_tracks_ids):
                for tr_id in new_tracks:
                    new_track = self.tracks_db.get_track_by_index(tr_id)
                    new_tr_first_p = self.points_db.get_det_by_index(new_track.tail[0])
                    predict_stamp = self.points_db.get_det_by_index(new_track.tail[0]).stamp

                    act_track = self.active_single_track
                    # self.calc_prediction(act_track,)
                    track_on_test = self.tracks_db.get_track_by_index(act_track)
                    on_test_last_p = self.points_db.get_det_by_index(track_on_test.get_last_point_id())
                    criterion = False
                    if on_test_last_p.stamp < new_tr_first_p.stamp:
                        # dt = new_tr_first_p.stamp-on_test_last_p.stamp
                        # print(f'dt = {dt}, st1 = {new_tr_first_p.stamp}, st2 = {on_test_last_p.stamp}')
                        # test_p_x, test_p_y = on_test_last_p.get_center()
                        # print(f'test_p: [{test_p[0]},{test_p[1]}], v = {track_on_test.v_vect}')

                        try:
                            # x, y = on_test_last_p.get_center()
                            # dt = predict_stamp - on_test_last_p.stamp
                            # v = track_on_test.v_vect
                            # print(f'dt = {dt}, st1 = {new_tr_first_p.stamp}, st2 = {on_test_last_p.stamp}')
                            # print(f'v: {v}')
                            # px = x + v[0] * dt
                            # py = y + v[1] * dt
                            px,py = self.calc_prediction(act_track,predict_stamp)
                            # print(f'px,py: {px},{py},,<=>{new_tr_first_p.get_center()}')
                            p2p_range = glib.pt2pt_2d_range([px,py],
                                                             new_tr_first_p.get_center())

                            predicted = [px, py]

                            criterion = True
                            criterion &= p2p_range < self.initial_search_radius
                            # print(f'Vs: {track_on_test.v_vect}, {new_track.v_vect}')
                            # criterion&=glib.get_v_angle(track_on_test.v_vect,new_track.v_vect)<45
                            criterion &= check_boxes_sizes(on_test_last_p.centered_box, new_tr_first_p.centered_box,
                                                           self.size_mismatch)
                            # criterion&=
                        except Exception as e:
                            print('Ошибка в проверке продолжений')
                            print(e.args)
                            criterion = False

                    if criterion:
                        p2p_ranges.append(p2p_range)
                        p2p_ids.append(tr_id)

                        # track_on_test.update(new_track.tail[0], self.current_stamp, new_track.v_vect[0],
                        #                      new_track.v_vect[1])
                        # track_on_test.update(new_track.tail[1], self.current_stamp, new_track.v_vect[0],
                        #                      new_track.v_vect[1])
                        # track_on_test.update(new_track.tail[2], self.current_stamp, new_track.v_vect[0],
                        #                      new_track.v_vect[1])
                        # self.points_db.get_det_by_index(new_track.tail[0]).tr_id = act_track
                        # self.points_db.get_det_by_index(new_track.tail[1]).tr_id = act_track
                        # self.points_db.get_det_by_index(new_track.tail[2]).tr_id = act_track
                        #
                        # new_track.flush()
                        # break

                if len(p2p_ranges)>0:

                    print('Найдены Подходящие варианты')
                    print(p2p_ranges)
                    min_range = min(p2p_ranges)
                    p_index = p2p_ranges.index(min_range)
                    tr_id = p2p_ids[p_index]
                    new_track = self.tracks_db.get_track_by_index(tr_id)
                    v_vect = new_track.v_vect
                    p0 = new_track.tail[0]
                    self.tracks_db.update_track_by_index(self.active_single_track, p0, self.current_stamp,v_vect[0],v_vect[1])
                    self.points_db.bind_to_track_by_index(p0,self.active_single_track)
                    p1 = new_track.tail[1]
                    self.tracks_db.update_track_by_index(self.active_single_track, p1,self.current_stamp,v_vect[0],v_vect[1])
                    self.points_db.bind_to_track_by_index(p1, self.active_single_track)
                    p2 = new_track.tail[2]
                    self.tracks_db.update_track_by_index(self.active_single_track, p2, self.current_stamp,v_vect[0],v_vect[1])
                    self.points_db.bind_to_track_by_index(p2, self.active_single_track)
                    new_track.flush()
                    print(f'!_!_!__#@($()*$@(&@^Траектория продолжена, {min_range}')

            elapsed = timer.stop()
            # print(f'<<<<<<<<<<<<Внутри обработки: окончательная обработка, {elapsed}')
        elif algorithm == 2:    #Сопровождение одной цели:
            track = self.tracks_db.get_track_by_index(self.active_single_track)
            last_point = self.points_db.get_det_by_index(track.get_last_point_id())
            p_x,p_y = self.calc_prediction(self.active_single_track,self.current_stamp)
            v = ((track.v_vect[0])**2 +(track.v_vect[1])**2)**0.5
            new_p_ids = []
            new_p_ranges = []
            for det_idx_0 in self.points_history.get_data(0):
                criterion = True
                current_det = self.points_db.get_det_by_index(det_idx_0)
                dt = current_det.stamp - last_point.stamp
                search_r = dt*v*self.search_radius_k
                p2p_range = glib.pt2pt_2d_range([p_x,p_y],
                                                 current_det.get_center())
                criterion &= check_boxes_sizes(last_point.centered_box,current_det.centered_box,self.size_mismatch)
                criterion &=  p2p_range < search_r

                if criterion:
                    new_p_ids.append(det_idx_0)
                    new_p_ranges.append(p2p_range)
                    # self.points_db.bind_to_track_by_index(det_idx_0,
                    #                                       self.active_single_track)  # Привязываем последнюю точку тройки к траектории
                    # self.tracks_db.update_track_by_index(self.active_single_track, det_idx_0, self.current_stamp, 0, 0)
                    # print('!!!!!!!!!!!!!!!!!!!!!!!!!!!SINGLE TRACK UPDATED')




    def calc_prediction(self,id,predict_stamp):
        track = self.tracks_db.get_track_by_index(id)
        last_p_id = track.get_last_point_id()
        if last_p_id >= 0:
            last_point = self.points_db.get_det_by_index(last_p_id)
            x, y = last_point.get_center()
            dt = predict_stamp - last_point.stamp
            v = track.v_vect
            px = x + v[0] * dt
            py = y + v[1] * dt
        return px,py
    def calc_p2p_velocity(self,p1_id,p2_id):
        p1 = self.points_db.get_det_by_index(p1_id)
        p2 = self.points_db.get_det_by_index(p2_id)
        p1_center = p1.get_center()
        p2_center = p2.get_center()
        dt = p2.stamp - p1.stamp
        vx = (p2_center[0]-p1_center[0])/dt
        vy = (p2_center[1] - p1_center[1])/dt
        # return np.array([vx,vy],dtype=np.float)
        return vx,vy


    def draw_det_with_id(self,image,id, color = (0,0,150)):
        track = self.tracks_db.get_track_by_index(id)

        last_p_id = track.get_last_point_id()
        if last_p_id>=0:
            last_point = self.points_db.get_det_by_index(last_p_id)
            centered_box = last_point.centered_box
            b_to_draw = box_cvt_cent2corners(centered_box)  # преобразуем вид бокса на текущем кадре для удобства отрисовки
            if self.draw_converted:
                # p1 = (b_to_draw[0], b_to_draw[1])
                # p2 = (b_to_draw[2], b_to_draw[3])
                x1,y1 =  self.meta_converter.translate_pt(b_to_draw[0],b_to_draw[1],'self','export_to')
                x2, y2 = self.meta_converter.translate_pt(b_to_draw[2], b_to_draw[3], 'self','export_to')
                b_to_draw = (x1,y1,x2,y2)
                # b_to_draw[0] = int(x1)
                # b_to_draw[1] = int(y1)
                # b_to_draw[2] = int(x2)
                # b_to_draw[3] = int(y2)
            # def translate_pt(self, pt_x, pt_y, from_alias, to_alias):

            cv.rectangle(image, (b_to_draw[0], b_to_draw[1]), (b_to_draw[2], b_to_draw[3]), color, 7)
            font = cv.FONT_HERSHEY_SIMPLEX
            # cv.putText(image,f'{id}({track.points_count}, {track.v_vect}px/s)',(b_to_draw[2],b_to_draw[1]),font,1.3,color,2,cv.LINE_AA)
            cv.putText(image, f'{id}({track.points_count}', (b_to_draw[2], b_to_draw[1]), font,
                       1.3, color, 2, cv.LINE_AA)
    def draw_circle_by_id(self,image,id, color = (0,150,150)):
        track = self.tracks_db.get_track_by_index(id)

        last_p_id = track.get_last_point_id()
        if last_p_id >= 0:
            last_point = self.points_db.get_det_by_index(last_p_id)
            x,y = last_point.get_center()
            cv.circle(image,(x,y),50,color,4)
    def draw_track_tail(self,image,id,color):
        track = self.tracks_db.get_track_by_index(id)

        last_p_id = track.get_last_point_id()
        if last_p_id >= 0:
            for i in range(track.tail_length-1):
                p1_idx = track.tail[(track.last_p_insert_idx-i)%track.tail_length]
                p2_idx = track.tail[(track.last_p_insert_idx-i-1)%track.tail_length]
                if (p2_idx>=0)&(p1_idx>=0):

                    p1_c = self.points_db.get_det_by_index(p1_idx).get_center()
                    p2_c = self.points_db.get_det_by_index(p2_idx).get_center()
                    if self.draw_converted:
                        x1,y1 = self.meta_converter.translate_pt(p1_c[0],p1_c[1],'self','export_to')
                        x2,y2 = self.meta_converter.translate_pt(p2_c[0],p2_c[1],'self','export_to')
                        cv.line(image, (x1, y1), (x2, y2), color, 2)
                    else:
                        cv.line(image, (p1_c[0], p1_c[1]), (p2_c[0], p2_c[1]), color, 2)
                    # print(f'central::::  {p1_c}, {p2_c}')

    def track_to_bytes_by_id(self,id,tail_l):
        track = self.tracks_db.get_track_by_index(id)
        length_to_encode = min(track.tail_length,tail_l,track.points_count)
        last_point = self.points_db.get_det_by_index(track.get_last_point_id())
        last_point.get_center()
        w,h = last_point.get_size()
        buf = self.codec.encode_track_header(id,[w,h],length_to_encode)
        for i in range(length_to_encode):
            p_idx = track.tail[(track.last_p_insert_idx - i) % track.tail_length]
            pt = self.points_db.get_det_by_index(p_idx)
            p1_c = pt.get_center()
            buf+=self.codec.encode_point(p1_c,pt.stamp)
        return buf
    def get_track_stamp(self,id):
        return self.tracks_db.get_track_by_index(id).last_update

    def draw_track_prediction(self,image,id,color, predict_stamp):
        track = self.tracks_db.get_track_by_index(id)
        last_p_id = track.get_last_point_id()
        if last_p_id >= 0:
            last_point = self.points_db.get_det_by_index(last_p_id)
            x, y = last_point.get_center()
            dt = predict_stamp - last_point.stamp
            v = track.v_vect
            px = x + v[0]*dt
            py = y + v[1] * dt
            cv.circle(image, (int(px), int(py)), 15, color, 4)
            cv.line(image,(x,y),(int(px),int(py)),color,1,cv.LINE_4)

    def get_fresh_track_ids(self,min_points = 5):
        ids_list = []
        for track_idx in self.tracks_db.active_tracks_ids:
            track = self.tracks_db.get_track_by_index(track_idx)
            if (track.points_count >= min_points):
                if (track.last_update == self.current_stamp):
                    ids_list.append(track_idx)
        return ids_list



    def draw_fresh_tracks(self,image,min_points = 5,need_tails = False):
        for track_idx in self.tracks_db.active_tracks_ids:
            track = self.tracks_db.get_track_by_index(track_idx)
            if (track.points_count>=min_points):
                if (track.last_update == self.current_stamp):
                    if track.obj_type>0:
                        color = (0,0,150)
                    elif track.checked:
                        color = (0, 150, 0)
                    else:
                        color = self.color_fresh

                else:
                    color = self.color_old
                self.draw_det_with_id(image, track_idx,color)
                if need_tails:
                    self.draw_track_tail(image, track_idx,color)

    def get_last_track_az_el(self,track_id):
        track = self.tracks_db.get_track_by_index(track_id)
        point_idx = track.get_last_point_id()
        last_point = self.points_db.get_det_by_index(point_idx)
        return last_point.az_el_enabled, last_point.az,last_point.el

    def get_last_track_xy(self, track_id):
        track = self.tracks_db.get_track_by_index(track_id)
        point_idx = track.get_last_point_id()
        last_point = self.points_db.get_det_by_index(point_idx)
        return last_point.get_center()

    def get_track_len_by_id(self,track_id):
        return self.tracks_db.get_track_by_index(track_id).points_count

    def get_check_stat_by_id(self,track_id):
        track = self.tracks_db.get_track_by_index(track_id)
        return track.checked,track.obj_type,track.obj_type_last
    def update_n_process(self,detections,stamp,image_meta: glib.Image_meta = None):
        self.update_step(detections,stamp,image_meta)
        self.process_current_step()
        self.tracks_db.flush_by_stamp(stamp,self.flush_timeout)
        if self.active_single_track >=0:
            single_stamp = self.get_track_stamp(self.active_single_track)
            if abs(stamp-single_stamp) >self.flush_timeout:
                self.active_single_track = -1
        # print(f'<<< Tracker processed, {len(self.get_active_tracks_ids())} >>>')

    def update_object_type_by_id(self,id,type,stamp):
        track = self.tracks_db.get_track_by_index(id)
        if track.alive:
            track.update_obj_type(type,stamp)
class Detection_m:
    def __init__(self):
        self.centered_box = [np.int32(0),np.int32(0),np.int32(0),np.int32(0)]
        self.stamp = np.float64(0)
        self.tr_id = np.int32(-1)
        self.az_el_enabled = False
        self.az = np.float64(0)
        self.el = np.float64(0)

    def get_center(self):
        return [self.centered_box[0],self.centered_box[1]]
    def set_center(self,x_c,y_c):
        self.centered_box[0] = x_c
        self.centered_box[1] = y_c
    def get_az_el(self):
        return self.az,self.el
    def get_size(self):
        return [self.centered_box[2],self.centered_box[3]]
    def draw(self,image,color = (0,0,0),thickness = 2):
        pt1,pt2 = glib.box_cvt_cent2corners_pts(self.centered_box)
        cv.rectangle(image,pt1,pt2,color,thickness )
        cv.putText(image,f'{self.object_class}[{round(self.p,3)}]',(pt1[0],pt1[1]-10),3,1.3,(0,0,0),2)

    def print(self):
        print(f'detecton: {self.centered_box}')

class Detection_recognition(Detection_m):
    '''
    Расширенный класс для хранения записи о распознанном объекте
    '''
    def __init__(self):
        super().__init__()
        self.object_class = 0

class Detections_m_list:
    """
    класс для хранения списка обнаружений
    """

    def __init__(self,list_size):
        print('Инициализация списка областей')
        self.default_box_style = BOX_STYLE_CVT
        self.list_size = list_size
        self.list = []
        for i in range(list_size):
            self.list.append(Detection_m())
        self.last_insert = -1

    def insert_new(self, box, stamp = 0.0,b_style = None, image_meta:glib.Image_meta = None):
        if not b_style:
            b_style = self.default_box_style
        self.last_insert +=1
        self.last_insert%=self.list_size
        if b_style == BOX_STYLE_CENTERED:
            self.list[self.last_insert].centered_box[0] = box[0]
            self.list[self.last_insert].centered_box[1] = box[1]
            self.list[self.last_insert].centered_box[2] = box[2]
            self.list[self.last_insert].centered_box[3] = box[3]
        elif b_style == BOX_STYLE_CVT:
            self.list[self.last_insert].centered_box[0] = box[0]+int(box[2]/2)
            self.list[self.last_insert].centered_box[1] = box[1]+int(box[3]/2)
            self.list[self.last_insert].centered_box[2] = box[2]
            self.list[self.last_insert].centered_box[3] = box[3]
        elif b_style == BOX_STYLE_P1P2:
            w = box[2]-box[0]
            h = box[3]-box[1]
            self.list[self.last_insert].centered_box[0] = box[0] + int(w / 2)
            self.list[self.last_insert].centered_box[1] = box[1] + int(h / 2)
            self.list[self.last_insert].centered_box[2] = w
            self.list[self.last_insert].centered_box[3] = h
        if not(image_meta ==None):
            xc,yc = self.list[self.last_insert].get_center()
            az,el = image_meta.get_abs_p_pos(xc,yc)
            self.list[self.last_insert].az_el_enabled = True
            self.list[self.last_insert].az = az
            self.list[self.last_insert].el = el
        self.list[self.last_insert].stamp = stamp
        self.list[self.last_insert].tr_id = -1
        # print(f'[[[[[[[[[[[[[[[[[last inserted id in p db: {self.last_insert}, {(xc,yc)} - >{(az,el)}')
        return self.last_insert

    def get_det_by_index(self,index):
        return self.list[index]

    def bind_to_track_by_index(self,p_index,track_id):
        self.list[p_index].tr_id = track_id

    def update_all_px_pos_to_meta(self,new_meta:glib.Image_meta):
        for pt in self.list:
            # print(f'converting {pt.az,pt.el}')
            new_x, new_y = new_meta.put_abs_p_pos(pt.az,pt.el)
            pt.set_center(new_x,new_y)

class Recognition_m_list(Detections_m_list):
    '''Расширенный класс для работы с разпознанными обнаружениями'''
    def __init__(self,list_size):
        print('Инициализация списка распознаваний')
        self.default_box_style = BOX_STYLE_P1P2
        self.list_size = list_size
        self.list = []
        for i in range(list_size):
            self.list.append(Detection_recognition())
        self.last_insert = -1

    def insert_new_recognition(self,box,object_class, stamp = 0.0,b_style = None, image_meta:glib.Image_meta = None,p= 0.0):
        last_insert = self.insert_new(box,stamp,b_style,image_meta)
        self.list[last_insert].object_class = object_class
        self.list[last_insert].tr_id = last_insert
        self.list[last_insert].p = p
        return last_insert

    def get_recogn_history(self,start_stamp, stop_stamp = None,classes = None):
        if not stop_stamp:
            stop_stamp = self.list[self.last_insert].stamp
        idxs = []
        for i in range(self.list_size):
            current_idx = (self.last_insert-i)%self.list_size
            rec_on_check = self.get_det_by_index(current_idx)
            if rec_on_check.stamp>=start_stamp:
                if rec_on_check.stamp<=stop_stamp:
                    if not classes:
                        idxs.append(current_idx)
                    elif rec_on_check.object_class in classes:
                        idxs.append(current_idx)
            else:
                break
        return idxs


class Track:
    '''
    Класс для описания траектории
    '''
    def __init__(self, id, tail_length = 10):
        self.tail_length = tail_length
        self.alive = False
        self.id = np.int32(id)
        self.tail = np.zeros(self.tail_length, dtype = np.int32)-1
        self.last_p_insert_idx = -1
        self.last_update = np.float64(0.0)
        self.points_count = np.int32(0)
        self.obj_type = np.int32(0) #тип/класс объекта
        self.obj_type_last = np.float64(0.0)
        self.checked = False
        self.v_vect = np.array([0.0, 0.0], dtype=np.float32)


    def update_obj_type(self,obj_type,stamp):
        self.obj_type = obj_type
        self.stamp = stamp
        self.checked = True

    def get_last_point_id(self):
        if self.last_p_insert_idx>=0:
            last_p = self.tail[self.last_p_insert_idx]
        else:
            last_p = -1
        return last_p

    def get_sorted_tail(self):
        sorted_t = [-1]*self.tail_length
        for i, el in enumerate(self.tail):
            sorted_t[i] = self.tail[(self.last_p_insert_idx - i) % self.tail_length]
        return sorted_t


    def flush(self):
        # print(f'Траектория {self.id} стерта')
        self.alive = False
        self.tail*=0
        self.tail-=1
        self.last_update*=0
        self.last_p_insert_idx = -1
        self.points_count*=0
        self.obj_type*=0
        self.obj_type_last*=0
        self.checked = False
        self.v_vect[0] = 0.0
        self.v_vect[1] = 0.0

    def update(self,new_pt_idx,stamp,vx = 0,vy = 0):
        self.last_p_insert_idx +=1
        self.last_p_insert_idx%=self.tail_length
        self.tail[self.last_p_insert_idx] = new_pt_idx
        self.last_update = stamp
        self.alive = True
        self.points_count+=1
        self.v_vect[0] = vx
        self.v_vect[1] = vy

    def print(self):
        print(f'Object id: {self.id}, last point index = {self.tail[self.last_p_insert_idx]}, last update: {self.last_update}')



class Tracks_list:
    '''
    Класс для хранения и обработки траекторий
    '''
    def __init__(self,list_size,tail_length = 10):
        self.list_size = list_size
        self.tracks_list = []
        self.tail_length = tail_length
        self.active_tracks_ids = []
        self.last_insert = -1 #Для отслеживания выдачи идентификаторов
        for i in range(list_size):
            self.tracks_list.append(Track(i,self.tail_length))




    def get_empty_id(self):
        for i in range(1,self.list_size):
            check = (self.last_insert+i)%self.list_size
            if not (check in self.active_tracks_ids):
                break
        return check

    def activate_new_track(self):
        new_id = self.get_empty_id()
        self.last_insert = new_id
        self.tracks_list[new_id].alive = True
        return new_id

    def update_track_by_index(self,track_index,new_point,stamp,vx = 0.0,vy = 0.0):
        self.tracks_list[track_index].update(new_point,stamp,vx,vy)
        if not(track_index in self.active_tracks_ids):
            # print('new!')
            self.active_tracks_ids.append(track_index)

    def get_track_by_index(self,idx):
        return self.tracks_list[idx]


    def flush_by_stamp(self,stamp,tout):
        active_buffer = []
        # print(self.active_tracks_ids)
        for i in self.active_tracks_ids:
            # print(f'analizing id {i}')
            current_tr = self.tracks_list[i]
            # current_tr.print()
            if (stamp - current_tr.last_update)>tout:
                current_tr.flush()
                # print('after_flush')
                # current_tr.print()
                # self.tracks_list[i].flush()
            else:
                active_buffer.append(i)
        self.active_tracks_ids = active_buffer




class Alias_list:

    def __init__(self,size):
        # type 1 - Для narrow, type 2 - для Wide
        self.aliases_type1 = [-1]*size
        self.aliases_type2 = [-1]*size
        self.busy = [False]*size

    def flush_all(self):
        self.aliases_type1*=0
        self.aliases_type1-=1
        self.aliases_type2 *= 0
        self.aliases_type2 -= 1
        self.busy &= False

    def to_string(self):
        s = ''
        for i,flag in enumerate(self.busy):
            if flag:
                s+=f'type 1 <=> type 2: {self.aliases_type1[i]} <=> {self.aliases_type2[i]}\n'
        return s


    def put_or_update_pair(self,id_type_1,id_type_2):
        if (id_type_1 in self.aliases_type1):
            idx = self.aliases_type1.index(id_type_1)
        elif (id_type_2 in self.aliases_type2):
            idx = self.aliases_type2.index(id_type_2)
        else:
            for i,flag in enumerate(self.busy):
                if not (flag):
                    idx = i
                    break
                else:
                    idx = -1
        if idx >=0:
            self.busy[idx] = True
            self.aliases_type1[idx] = id_type_1
            self.aliases_type2[idx] = id_type_2

    def search_alias(self,id,type:int):
        idx = -1
        if type ==1:
            if id in self.aliases_type1:
                idx = self.aliases_type1.index(id)
        else:
            if id in self.aliases_type2:
                idx = self.aliases_type2.index(id)
        if idx>=0:
            return self.aliases_type1[idx],self.aliases_type2[idx]
        else:
            return -1,-1

    def pair_to_string(self,id,type:int):
        pair = self.search_alias(id,type)
        return f'{pair[0]} <=>{pair[1]}'

    def delete_pair(self,id,type):
        idx = -1
        if type == 1:
            if id in self.aliases_type1:
                idx = self.aliases_type1.index(id)
        else:
            if id in self.aliases_type2:
                idx = self.aliases_type2.index(id)
        if idx >= 0:
            self.busy[idx] = False
            self.aliases_type1[idx] = -1
            self.aliases_type2[idx] = -1
    def validate_with_trackers(self,tracker1:Tracker_m,tracker2:Tracker_m):
        for i,b in enumerate(self.busy):
            if b:
                alive = False
                alive|=(self.aliases_type1[i] in tracker1.get_active_tracks_ids())
                alive|=(self.aliases_type2[i] in tracker2.get_active_tracks_ids())
                if alive:
                    pass
                else:
                    self.aliases_type1[i] = -1
                    self.aliases_type2[i] = -1
                    self.busy[i] = False

def check_pairs(tracker_n:Tracker_m,tracker_w:Tracker_m, aliases:Alias_list, pts_converter:glib.Meta2meta_converter):
    fresh_ids_n = tracker_n.get_fresh_track_ids(5)
    fresh_ids_w = tracker_w.get_fresh_track_ids(5)

    pairing_search_r = 75
    for fr_id_n in fresh_ids_n:
        x, y = tracker_n.get_last_track_xy(fr_id_n)
        x_n2w, y_n2w = pts_converter.translate_pt(x, y, 'narrow', 'wide')
        # cv.circle(test_image_w, (int(x_n2w), int(y_n2w)), pairing_search_r, (100, 100, 100), 2)

        # Проходим по траекториям узкой камеры и проверяем на соответствие траекториям широкой камеры
        for fr_id_w in fresh_ids_w:

            x_w, y_w = tracker_w.get_last_track_xy(fr_id_w)


            p2p_range = glib.pt2pt_2d_range([x_w, y_w], [x_n2w, y_n2w])
            if p2p_range < pairing_search_r:
                # Если поблизости найдена траектория широкоугольной камеры, требуется провести дополнительный анализ

                w_points_tail_ids = tracker_w.get_track_by_id(fr_id_w).get_sorted_tail()
                n_points_tail_ids = tracker_n.get_track_by_id(fr_id_n).get_sorted_tail()

                pt_n = tracker_n.get_detection_by_idx(n_points_tail_ids[0])

                min_time_mismatch = 999
                for tail_deepness_w, pt_id_w in enumerate(w_points_tail_ids):
                    if pt_id_w >= 0:
                        pt_w = tracker_w.get_detection_by_idx(pt_id_w)
                        # print('id = ',pt_id_w)
                        # # print(tracker_w.points_db.get_det_by_index(pt_id_w).stamp)
                        # print(pt_w)
                        # print(pt_n.stamp)
                        if abs(pt_w.stamp - pt_n.stamp) < min_time_mismatch:
                            min_time_mismatch = abs(pt_w.stamp - pt_n.stamp)
                            x_n_, y_n_ = pt_n.get_center()

                            x_w_, y_w_ = pt_w.get_center()
                            w_tail_nearest = tail_deepness_w
                n2w_direction_thresh = 7
                if min_time_mismatch < 10:
                    x_n2w_, y_n2w_ = pts_converter.translate_pt(x_n_, y_n_, 'narrow', 'wide')
                    # cv.line(test_image_w,(int(x_n2w_),int(y_n2w_)),(int(x_w_),int(y_w_)),(0,0,255),5)
                    checker = 0
                    for i in range(3):
                        # Осуществляем валидацию точек, которые будут использоваться для вычисления векторов скорости
                        validate_points = True
                        validate_points &= (w_points_tail_ids[i + w_tail_nearest] >= 0)
                        validate_points &= (w_points_tail_ids[i + w_tail_nearest + 1] >= 0)
                        validate_points &= (n_points_tail_ids[i + 1] >= 0)
                        validate_points &= (n_points_tail_ids[i + 1] >= 0)
                        # Если с точками все в порядке, проверяем вектора скорости
                        if validate_points:
                            v_w = tracker_w.calc_p2p_velocity(w_points_tail_ids[i + w_tail_nearest + 1],
                                                              w_points_tail_ids[i + w_tail_nearest])
                            v_n = tracker_n.calc_p2p_velocity(n_points_tail_ids[i + 1], n_points_tail_ids[i])
                            v_ratio = glib.pt2pt_2d_range([0, 0], v_w) / glib.pt2pt_2d_range([0, 0], v_n)
                            # print(f'wide_V = {v_w}, \nnarrow_V = ')
                            w2n_angle = glib.get_v_angle(glib.Vect(v_w), glib.Vect(v_n))
                            # print('Wide to narrow velocity angle: ',w2n_angle)
                            # print(f'velocity ratio: {v_ratio}')
                            if (w2n_angle < n2w_direction_thresh) & (v_ratio > 0.1) & (v_ratio < 0.3):
                                checker += 1
                    # Если все вектора скорости прошли проверку, делаем вывод о тождественности траектрий

                    if checker == 3:
                        aliases.put_or_update_pair(fr_id_n, fr_id_w)

                        # print(f'narrow_id {fr_id_n} == wide_id {fr_id_w}')
                        # x_n2w_, y_n2w_ = pts_converter.translate_pt(x_n_, y_n_, 'narrow', 'wide')
                        # cv.line(test_image_w, (int(x_n2w_), int(y_n2w_)), (int(x_w_), int(y_w_)), (0, 0, 255), 10)
                        #
                        # # cv.putText(image,f'{id}({track.points_count}, {track.v_vect}px/s)',(b_to_draw[2],b_to_draw[1]),font,1.3,color,2,cv.LINE_AA)
                        # cv.putText(test_image_w, f'{id}({fr_id_n}/{fr_id_w}', (x_n2w_, y_n2w_ - 50), font,
                        #            1.3, (0, 0, 230), 5, cv.LINE_AA)

if __name__ == '__main__':
    det_list = Detections_m_list(100000)
    st = det_list.list[0].stamp
    st+=time.time()
    print(type(st))

    track_db = Tracks_list(1000)
    empty_id = track_db.get_empty_id()
    print(f'empty_id: {empty_id}')
    track_db.update_track_by_index(empty_id,3,10)
    track_db.tracks_list[empty_id].print()
    track_db.flush_by_stamp(30,5)
    track_db.tracks_list[empty_id].print()

    recogn_list = Recognition_m_list(100)
    last_time = 0.0
    for i in range(40):
        last_time =time.time()
        insertion_id = recogn_list.insert_new_recognition([10,10,20,20],1,time.time())
        print(insertion_id)
        time.sleep(0.1)

    list_r = recogn_list.get_recogn_history(last_time-1, classes=[0])
    print(list_r)
