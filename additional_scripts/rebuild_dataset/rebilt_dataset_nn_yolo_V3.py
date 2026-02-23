import copy
import random
import os
from geometry_lib import *

class Rebild_Data:
    def __init__(self, path_file, path_file_rebild, path_no_sort, h_rebilt, w_rebild):
        self.path_file, self.path_file_rebild, self.path_no_sort = path_file, path_file_rebild, path_no_sort
        self.h_rebild, self.w_rebild = h_rebilt, w_rebild
        self.__x_rand_left, self.__x_rand_right, self.__y_rand_left, self.__y_rand_right = 25, 25, 25, 25
        self.__pix_x_indent, self.__pix_y_indent = 5, 5
        self.__time_delay = 1000

    def set_x_rand_left(self, x_rand_left):
        self.__x_rand_left = x_rand_left

    def get_x_rand_left(self):
        return self.__x_rand_left

    def set_x_rand_right(self, x_rand_right):
        self.__x_rand_right = x_rand_right

    def get_x_rand_right(self):
        return self.__x_rand_right

    def set_y_rand_left(self, y_rand_left):
        self.__y_rand_left = y_rand_left

    def get_y_rand_left(self):
        return self.__y_rand_left

    def set_y_rand_right(self, y_rand_right):
        self.__y_rand_right = y_rand_right

    def get_y_rand_right(self):
        return self.__y_rand_right

    def set_pix_x_indent(self, pix_x_indent):
        self.__pix_x_indent = pix_x_indent

    def get_pix_x_indent(self):
        return self.__pix_x_indent

    def set_pix_y_indent(self, pix_y_indent):
        self.__pix_y_indent = pix_y_indent

    def get_pix_y_indent(self):
        return self.__pix_y_indent

    def set_time_delay(self, time_delay):
        self.__time_delay = time_delay

    def get_time_delay(self):
        return self.__time_delay

    def rebild_cordinate(self, lst_det, size_image_W=800, size_image_H=800, h_n=192, w_n=192):

        delta_x, delta_y = random.randint(-self.__x_rand_left, self.__x_rand_right), random.randint(-self.__y_rand_left, self.__y_rand_right)  # Получение смещения
        x_cnt, y_cnt, w, h = lst_det[1] * size_image_W, lst_det[2] * size_image_H, lst_det[3] * size_image_W, lst_det[4] * size_image_H
        x_left, y_left, x_right, y_right = int(x_cnt - (w / 2)), int(y_cnt - (h / 2)), int(x_cnt + (w / 2)), int(y_cnt + (h / 2))
        converture = cover_pt_by_area([x_cnt + delta_x, y_cnt + delta_y], area_w_h=[w_n, h_n], limit_box=[0, 0, size_image_W, size_image_H])
        x_left_reb, y_left_reb, x_right_reb, y_right_reb = x_left - converture[0], y_left - converture[1], x_right - converture[0], y_right - converture[1]

        return [x_left_reb, y_left_reb, x_right_reb, y_right_reb], converture

    def rebild_cordinate_with_increasing_resolution(self, lst_det, size_image_W=800, size_image_H=800, h_n=192, w_n=192):

        coef_W, coef_H = w_n / size_image_W, h_n / size_image_H
        x_cnt, y_cnt, w, h = lst_det[1] * size_image_W * coef_W, lst_det[2] * size_image_H * coef_H, lst_det[3] * size_image_W * coef_W, lst_det[4] * size_image_H * coef_H
        x_left, y_left, x_right, y_right = int(x_cnt - (w / 2)), int(y_cnt - (h / 2)), int(x_cnt + (w / 2)), int(y_cnt + (h / 2))


        return [x_left, y_left, x_right, y_right]



    def cord_normal(self, rebilt_lst, w_n=192, h_n=192):
        x_left_reb, y_left_reb, x_right_reb, y_right_reb = rebilt_lst[0], rebilt_lst[1], rebilt_lst[2], rebilt_lst[3]
        w_box, h_box = x_right_reb - x_left_reb, y_right_reb - y_left_reb
        x_cnt_reb_norm, y_cnt_reb_norm, w_norm, h_norm = (x_left_reb + (w_box / 2)) / w_n, (y_left_reb + (h_box / 2)) / h_n, w_box / w_n, h_box / h_n
        return [x_cnt_reb_norm, y_cnt_reb_norm, w_norm, h_norm]

    def read_detect(self, path_f, names, format_open="r"):
        list_detection = []
        with open(f"{path_f}/{names}.txt", format_open) as file_1:
            for j, b in enumerate(file_1):
                list_detection.append(list(map(float, b.split())))
        return list_detection

    def function_img_show(self, frame, time_delay):
        cv.imshow("frame", frame)
        cv.waitKey(time_delay)

    def write_file(self, names, detect_norm, class_id, recording_mode="a"):
        with open(names, recording_mode) as file:  # str(int(class_id))
            file.write(str(int(class_id)) + " " + str(detect_norm[0]) + " " + str(detect_norm[1]) + " " + str(detect_norm[2]) + " " + str(detect_norm[3]) + "\n")

    def check_detect(self, read_det, img, img_copy, H, W, names, count_image):
        if self.w_rebild <= W and self.h_rebild <= H:
            if len(read_det) > 1:
                for i in range(len(read_det)):
                    rebilt_lst, converture = self.rebild_cordinate(read_det[i], size_image_W=W, size_image_H=H, w_n=self.w_rebild, h_n=self.h_rebild)
                    cv.imwrite(f"{self.path_file_rebild}/{names}_{str(self.w_rebild)}x{str(self.h_rebild)}_{str(i)}.jpg", img[converture[1]:converture[3], converture[0]:converture[2]])
                    detect_norm = self.cord_normal(rebilt_lst, w_n=self.w_rebild, h_n=self.h_rebild)
                    cv.rectangle(img_copy[converture[1]:converture[3], converture[0]:converture[2]], (rebilt_lst[0], rebilt_lst[1]), (rebilt_lst[2], rebilt_lst[3]), (0, 0, 255), 1)
                    self.write_file(f"{self.path_file_rebild}/{names}_{str(self.w_rebild)}x{str(self.h_rebild)}_{str(i)}.txt", detect_norm, read_det[i][0], recording_mode="a")

                    for j in range(len(read_det)):
                        x_cnt, y_cnt, w, h = read_det[j][1] * W, read_det[j][2] * H, read_det[j][3] * W, read_det[j][4] * H
                        if (converture[0] + self.__pix_x_indent < int(x_cnt) < converture[2] - self.__pix_x_indent) and (converture[1] + self.__pix_y_indent < int(y_cnt) < converture[3] - self.__pix_y_indent) and (read_det[i] != read_det[j]):
                            x_left, y_left, x_right, y_right = int(x_cnt - (w / 2)), int(y_cnt - (h / 2)), int(x_cnt + (w / 2)), int(y_cnt + (h / 2))
                            x_left_reb, y_left_reb, x_right_reb, y_right_reb = x_left - converture[0], y_left - converture[1], x_right - converture[0], y_right - converture[1]
                            det_norm = self.cord_normal([x_left_reb, y_left_reb, x_right_reb, y_right_reb], w_n=self.w_rebild, h_n=self.h_rebild)
                            cv.rectangle(img_copy[converture[1]:converture[3], converture[0]:converture[2]], (x_left_reb, y_left_reb), (x_right_reb, y_right_reb), (0, 0, 255), 1)
                            self.write_file(f"{self.path_file_rebild}/{names}_{str(self.w_rebild)}x{str(self.h_rebild)}_{str(i)}.txt", det_norm, read_det[j][0], recording_mode="a")

                    self.function_img_show(img_copy[converture[1]:converture[3], converture[0]:converture[2]], self.__time_delay)
            else:
                for i in range(len(read_det)):
                    rebilt_lst, converture = self.rebild_cordinate(read_det[i], size_image_W=W, size_image_H=H, w_n=self.w_rebild, h_n=self.h_rebild)
                    cv.imwrite(f"{self.path_file_rebild}/{names}_{str(self.w_rebild)}x{str(self.h_rebild)}_{str(i)}.jpg", img[converture[1]:converture[3], converture[0]:converture[2]])
                    detect_norm = self.cord_normal(rebilt_lst, w_n=self.w_rebild, h_n=self.h_rebild)

                    cv.rectangle(img_copy[converture[1]:converture[3], converture[0]:converture[2]], (rebilt_lst[0], rebilt_lst[1]), (rebilt_lst[2], rebilt_lst[3]), (0, 0, 255), 1)
                    self.write_file(f"{self.path_file_rebild}/{names}_{str(self.w_rebild)}x{str(self.h_rebild)}_{str(i)}.txt", detect_norm, read_det[i][0], recording_mode="a")
                    self.function_img_show(img_copy[converture[1]:converture[3], converture[0]:converture[2]], self.__time_delay)
        else:
            img = cv.resize(img, (self.w_rebild, self.h_rebild))
            cv.imwrite(f"{self.path_file_rebild}/{names}_{str(self.w_rebild)}x{str(self.h_rebild)}_{str(count_image)}.jpg", img)
            for i in range(len(read_det)):
                rebilt_lst = self.rebild_cordinate_with_increasing_resolution(read_det[i], size_image_W=W, size_image_H=H, w_n=self.w_rebild, h_n=self.h_rebild)
                detect_norm = self.cord_normal(rebilt_lst, w_n=self.w_rebild, h_n=self.h_rebild)
                cv.rectangle(img, (rebilt_lst[0], rebilt_lst[1]), (rebilt_lst[2], rebilt_lst[3]), (0, 0, 255), 1)
                self.write_file(f"{self.path_file_rebild}/{names}_{str(self.w_rebild)}x{str(self.h_rebild)}_{str(count_image)}.txt", detect_norm, read_det[i][0], recording_mode="a")

            self.function_img_show(img, self.__time_delay)

    def walk_files(self):
        lst_file = os.listdir(self.path_file)#[:50000]
        count_image = 0
        for i, k in enumerate(lst_file):
            print(k[:-4])
            name, *formats = k.split(".")
            if formats[-1] != "txt":
                count_image += 1
                image = cv.cvtColor(cv.imread(f"{self.path_file}/{k}"), cv.COLOR_BGR2GRAY)
                H, W = image.shape
                image_copy = copy.deepcopy(image)
                try:
                    #lst_det = self.read_detect(self.path_file, name, format_open="r")
                    lst_det = self.read_detect(self.path_file, k[:-4], format_open="r")
                    self.check_detect(lst_det, image, image_copy, H, W, k[:-4], count_image)

                    #self.check_detect(lst_det, image, image_copy, H, W, name, count_image)
                except:
                    print(f"Не найден соответсвующий файл txt с именем {name}.txt. Переношу изображение {k} из каталога {self.path_file} в каталог {self.path_no_sort}.....")
                    os.replace(f"{self.path_file}/" + k, f"{self.path_no_sort}/" + k)

cls = Rebild_Data(r"/home/usr/Музыка/saver_detect_60 (1)/saver_detect_60/heavy_vehicle/", r"/home/usr/Музыка/saver_detect_60 (1)/saver_detect_60/reb_data/", r"/home/usr/Изображения/no_sort/", 640, 640)
cls.set_pix_x_indent(5)
cls.set_pix_y_indent(5)
cls.set_time_delay(10)
cls.set_x_rand_left(-22)
cls.set_x_rand_right(22)
cls.set_y_rand_left(-22)
cls.set_y_rand_right(22)
cls.walk_files()
