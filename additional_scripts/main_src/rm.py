import os
import cv2 as cv


class Refactoring_dataset:
    def __init__(self):
        self.obj = None

    def get_description_write_img(self):
        print("Эта функция нарезает видео на фреймы. Необходимо указать путь до видеофайла, путь куда сохраняем, имя картинки и расширение ее (при необходимости).")


    def write_img(self, path_videos, path_save, title_images=f"BMP4_", expansion_images="jpg", img_show=True):
        c = 0
        try:
            os.makedirs(path_save, exist_ok=True)
            cap = cv.VideoCapture(path_videos)

            for _ in range(12000):
                img, flg = cap.read()
                print(f"Skipping_images:{_}")


            while cap.isOpened():

                flag, images = cap.read()
                if flag:
                    c += 1
                    cv.imwrite(path_save+f"{title_images}{c}.{expansion_images}", images)
                    if img_show:
                        cv.imshow(f"frame", images)
                        cv.waitKey(1)
                else:
                    break
        except Exception as e:
            print(f"ERRR_write_img: {e.args}")

    def get_description_read_lables_txt(self):
        print("Эта функция читает разметку в файле и отбирает только классы объектов, небходимо указать путь до файла.")

    def read_lables_txt(self, pth):
        lst_classes = []
        try:
            with open(pth, "r") as files:
                for file in files:
                    ls = file.split()
                    lst_classes.append(str(int(float(ls[0]))))
            return lst_classes
        except Exception as e:
            print(f"Errr_in_function_read_lables_txt: {e.args}")

    def get_description_edit_classes(self):
        print("Эта функция редактирует имена классов при условии если известно, что в папке лежат объекты только одного класса, необходимо указать путь и имя класса на который редактируем.")


    def edit_classes(self, path, classes=0):


        lst = [k for i, k in enumerate(os.listdir(path)) if k[-3:] == "txt"]
        for i, k in enumerate(lst):
            lst_obj = []
            with open(path+k, "r") as files:
                for file in files:
                    ls = file.split()
                    ls[0] = str(classes)
                    lst_obj.append(ls)

            with open(path+k, "w") as files2:
                for i, k in enumerate(lst_obj):
                    files2.write(" ".join(k) + "\n")

    def get_description_rename_file(self):
        print("Эта функция раскидывает данные в разные папки для треннировки (папку с картинками и в отдельную папку с лейблами), указываем основной путь к данным, путь к папке с катинками и путь к папке где будут храниться лейблы")

    def rename_file(self, path1, path_img, path_txt):
        os.makedirs(path_img, exist_ok=True)
        os.makedirs(path_txt, exist_ok=True)

        lst_files_images = [k for i, k in enumerate(os.listdir(path1)) if k[-3::] == "jpg" or k[-3::] == "png" or k[-3::] == "jpeg" or k[-3::] == "tif"]
        lst_files_txt = [k for i, k in enumerate(os.listdir(path1)) if k[-3::] == "txt"]

        for i, k in enumerate(lst_files_images):
            os.rename(path1 + k, path_img + k)

        for j, m in enumerate(lst_files_txt):
            os.rename(path1 + m, path_txt + m)

    def get_description_rm_empty_txt(self):
        print("Эта функция удаляет пустой txt файл и картинку к нему при флаге True просто переносит в другую папку на допроверку (указываем пути до основного набора данных и пути куда надо перенсти данные при условии если флаг True)")

    def rm_empty_txt(self, path, name_file, path_opt_txt=None, path_opt_img=None, flag=False):
        lst = []
        with open(path+name_file, "r") as file:
            for i in file:
                ls = i.split()
                lst.append(ls)

        if lst == []:
            print(path[:-3]+"jpg", path)
            if flag == False:
                print(path+name_file, path+name_file[:-3]+"jpg")
                os.remove(path+name_file)
                os.remove(path+name_file[:-3]+"jpg")
            else:
                os.makedirs(path_opt_img, exist_ok=True)
                os.makedirs(path_opt_txt, exist_ok=True)
                os.rename(path + name_file, path_opt_txt + name_file)
                os.rename(path + name_file[:-3] + "jpg", path_opt_img + name_file[:-3] + "jpg")

    def get_description_remove_txt_or_image_data(self):
        print("Эта функция удаляет txt файл при условии что под него нет картинки, указываем путь к изображениям и путь к лейблам")

    def remove_txt_or_image_data(self, path_image, path_labels):
        lst_images, lst_labels = sorted(os.listdir(path_image)), sorted(os.listdir(path_labels))
        lst_del_label = []
        for i, k in enumerate(lst_labels):
            if not (k[:-3]+"jpg" in lst_images or  k[:-3]+"jpeg" in lst_images or k[:-3]+"png" in lst_images or k[:-3]+"tif" in lst_images):
                lst_del_label.append(k)
                try:
                    os.remove(path_labels + k)
                    print(f"Dell txt file: {k} in path: {path_labels}")
                except Exception as e:
                    print(f"ERRRRR_in_function_remove_txt_or_image_data: {e.args}")

        print(len(lst_del_label))
        print(lst_del_label)



    def get_description_rm_r(self):
        print("Эта функция удаляет файл в зависмости есть ли txt и если нет jpg то убирает txt и наоборот (указать основной путь).")

    def rm_r(self, path):
        lst = os.listdir(path)

        for i, k in enumerate(lst):
            try:
                if (k[:-3]+"jpg" in lst) and (k[:-3]+"txt" in lst):
                    print(f"img it's OK: {k[:-3]+'jpg'}, lables it's OK: {k[:-3]+'txt'}")
                else:
                    if (k[:-3] + "jpg" in lst) and not(k[:-3] + "txt" in lst):
                        os.remove(path + k[:-3] + "jpg")
                        print(f"Remove_images: {k[:-3] + 'jpg'} in path: {path}")
                    if (k[:-3] + "txt" in lst) and not(k[:-3] + "jpg" in lst):
                        os.remove(path + k[:-3] + "txt")
                        print(f"Remove_lables: {k[:-3] + 'txt'} in path: {path}")
            except Exception as e:
                print(e.args)

    def count_detect(self, path_img, path_lbl, path_dest, class_interested=["0", "1", "2", "3", "4", "5"], class_choice="3", max_obj=1000):
        lst_img = os.listdir(path_img)
        lst_txt = os.listdir(path_lbl)
        dict_object = {int(k):0 for i, k in enumerate(class_interested)}
        #lst_img, lst_txt = [k for i, k in enumerate(lst_files) if k[-3::] == "jpg" or k[-3::] == "jpeg" or k[-3::] == "png"], [k for i, k in enumerate(lst_files) if k[-3::] == "txt"]
        cnt = 0
        for img, txt in zip(lst_img, lst_txt):
            try:
                obj_det = self.read_lables_txt(path_lbl+img[:-3]+"txt")

                for i, p in enumerate(obj_det):
                    dict_object[int(p)] += 1

                if len(obj_det) == obj_det.count(class_choice) and cnt < max_obj:
                    try:
                        os.rename(path_img + img, path_dest + img)
                        os.rename(path_lbl + img[:-3]+"txt", path_dest + img[:-3]+"txt")
                        cnt += len(obj_det)
                    except Exception as e:
                        print(f"Err file_in_function_count_detect: {e.args}")
            except Exception as e:
                print(f"Errrr_in_function_count_detect: {e.args}")
        print(f"Total dict obj: {dict_object}")


    def rename_empty(self, path, path_dest):
        lst_txt = [k for i, k in enumerate(os.listdir(path)) if k[-3::] == "txt"]
        for i, k in enumerate(lst_txt):
            ls = self.read_lables_txt(path + k)
            if ls == []:
                try:
                    os.makedirs(path_dest, exist_ok=True)
                    os.rename(path + k, path_dest + k)
                    os.rename(path + k[:-3]+"jpg", path_dest + k[:-3]+"jpg")
                except Exception as e:
                    print(f"Errrr_in_function_rename_empty:{e.args}")





# labels_directory = "/home/usr/Документы/TANKS_DATA_T72-90 ABRAMS MERK/ABRAMS/lbl/"
# images_directory = "/home/usr/Документы/TANKS_DATA_T72-90 ABRAMS MERK/ABRAMS/img/"
# batch_process_folder(labels_directory, iou_thresh=0.5)
#batch_fix_format("/home/usr/Документы/TANKS_DATA_T72-90 ABRAMS MERK/ABRAMS/merkava/F/")
#
cls = Refactoring_dataset()    ### car, truck, heavy_vehicle, moto, boat, solder
#cls.edit_classes(f"/home/usr/Документы/TANKS_DATA_T72-90 ABRAMS MERK/ABRAMS/test/labels/")

#lst_txt = [k for i, k in enumerate(os.listdir(f"/home/usr/Изображения/saver_detect_120/boat/")) if k[-3:] == "txt"]
lst = [k for i, k in enumerate(os.listdir(f"/home/usr/Рабочий стол/data_no_dupl/images/calibrate/test/")) if k[-3::] == "txt"]
#cls.edit_classes(f"/home/usr/Изображения/saver_detect_120/boat/")
for i, k in enumerate(lst):
    cls.rm_empty_txt(f"/home/usr/Рабочий стол/data_no_dupl/images/calibrate/test/", k)
#cls.write_img(f"/media/usr/sdcard/DCIM/100MEDIA/DJI_0571.MP4", f"/home/usr/Документы/sahi5/", title_images=f"RUSP11_1_")
#cls.rename_file(f"/home/usr/Рабочий стол/moto/", f"/home/usr/Рабочий стол/data_no_dupl/images/train/", f"/home/usr/Рабочий стол/data_no_dupl/labels/train/")
#cls.count_detect(f"/home/usr/Рабочий стол/dst_path/images/train/", f"/home/usr/Рабочий стол/dst_path/labels/train/", f"/home/usr/Рабочий стол/moto/")

#cls.rename_empty(f"/home/usr/Изображения/saver_detect_115/car/", f"/home/usr/Изображения/saver_detect_115/empty_lbl/")


# lst = [k for i, k in enumerate(os.listdir(f"/home/usr/Документы/TANKS_DATA_T72-90 ABRAMS MERK/ABRAMS/img/")) if k[-3::] == "txt"]
# for i, k in enumerate(lst):
#     cls.rm_empty_txt(f"/home/usr/Документы/TANKS_DATA_T72-90 ABRAMS MERK/ABRAMS/img/", k)

#cls.rename_file(f"/home/usr/Видео/Записи экрана/truck_data (1)/truck_data/dst/", f"/home/usr/Загрузки/dataset_ground_vehicle/filtering_dataset/dataset_augment/images/train/", f"/home/usr/Загрузки/dataset_ground_vehicle/filtering_dataset/dataset_augment/labels/train/")
#cls.rename_file(f"/home/usr/Видео/Записи экрана/truck_data (1)/truck_data/truck_augm/", f"/home/usr/Видео/Записи экрана/truck_data (1)/truck_data/tst_img/", f"/home/usr/Видео/Записи экрана/truck_data (1)/truck_data/tst_txt/")
#cls.count_detect(f"/home/usr/Видео/Записи экрана/truck_data (1)/truck_data/tst_img/", f"/home/usr/Видео/Записи экрана/truck_data (1)/truck_data/tst_txt/", f"/home/usr/Видео/Записи экрана/truck_data (1)/truck_data/dst/")
# cls.set_dirs_data_sort((f"/home/usr/Музыка/images_288x288_3_class_V4/images/train/", f"/home/usr/Музыка/img_heli/drone/", f"/home/usr/Музыка/img_heli/bird/", f"/home/usr/Музыка/img_heli/plane/", f"/home/usr/Музыка/img_heli/heli/", f"/home/usr/Музыка/img_heli/drone_bird/", f"/home/usr/Музыка/img_heli/drone_plane/", f"/home/usr/Музыка/img_heli/drone_heli/", f"/home/usr/Музыка/img_heli/heli_bird/", f"/home/usr/Музыка/img_heli/plane_heli/", f"/home/usr/Музыка/img_heli/backround/", f"/home/usr/Музыка/img_heli/plane_bird/"))
# cls.cls
#cls.count_detect(f"/home/usr/Загрузки/dataset_ground_vehicle/filtering_dataset/dataset_augment/images/train/", f"/home/usr/Загрузки/dataset_ground_vehicle/filtering_dataset/dataset_augment/labels/train/", f"/home/usr/Изображения/ships_unmanned/GloriousStar_0531.v2i.yolov11/train/reb_data/", class_choice="4")
#cls.rename_file(f"/home/usr/Изображения/ships_unmanned/GloriousStar_0531.v2i.yolov11/train/reb_img/", f"/home/usr/Загрузки/dataset_ground_vehicle/filtering_dataset/dataset_augment/images/train/", f"/home/usr/Загрузки/dataset_ground_vehicle/filtering_dataset/dataset_augment/labels/train/")
#cls.rename_file(f"/home/usr/Изображения/ships_unmanned/unmanned boat.v2i.yolov11/test/root_data/", f"/home/usr/Загрузки/dataset_ground_vehicle/filtering_dataset/dataset_augment/images/train/", f"/home/usr/Загрузки/dataset_ground_vehicle/filtering_dataset/dataset_augment/labels/train/")
# lst = [k for i, k in enumerate(os.listdir(f"/home/usr/Видео/Записи экрана/trucker_img/")) if k[-3:] == "txt"]
# for i, k in enumerate(lst):
#     cls.rm_empty_txt(f"/home/usr/Видео/Записи экрана/trucker_img/", k)
#cls.count_detect(f"/home/usr/Видео/data_moto/data/", f"/home/usr/Видео/data_moto/filt_moto/")
#cls.write_img(f"/home/usr/Загрузки/Telegram Desktop/IMG_0820.MP4", f"/home/usr/sahi3/", title_images="for_sahi_")

#cls.edit_classes(f"/home/usr/Изображения/ships_unmanned/GloriousStar_0524.v1i.yolov11/valid/labels/", classes=4)
#cls.rm_r(f"/home/usr/Музыка/auto_dataset_19_02_2024/auto_dataset/plane/")
#cls.rename_file(f"/home/usr/Музыка/Drone Detection.v1i.yolov11/train/fck_img/", f"/home/usr/Музыка/Drone Detection.v1i.yolov11/train/fck_img_2/", f"/home/usr/Музыка/Drone Detection.v1i.yolov11/train/fck_lbl_2/")
#cls.rm_r(f"/home/usr/Музыка/Drone Detection.v1i.yolov11/train/fck_img/")


#rename_file(f"/home/usr/Изображения/saver_detect_3/reb_288x288_no_det/", f"/home/usr/Музыка/dataset_4_classes_iter_1.3/images/train/")

# path = f"/home/usr/Изображения/saver_detect_4/drone_reb_288x288/"
# rm_r(path)

#path_main_data, path_drone, path_bird, path_plane, path_heil, path_drone_bird, path_drone_plane, path_drone_heli, path_heli_bird, path_heli_plane, path_background, path_plane_bird = f"/home/usr/Музыка/img_heli/images_288x288_3_class_V4/images/train/", f"/home/usr/Музыка/img_heli/drone/", f"/home/usr/Музыка/img_heli/bird/", f"/home/usr/Музыка/img_heli/plane/", f"/home/usr/Музыка/img_heli/heli/", f"/home/usr/Музыка/img_heli/drone_bird/", f"/home/usr/Музыка/img_heli/drone_plane/", f"/home/usr/Музыка/img_heli/drone_heli/", f"/home/usr/Музыка/img_heli/heli_bird/", f"/home/usr/Музыка/img_heli/plane_heli/", f"/home/usr/Музыка/img_heli/backround/", f"/home/usr/Музыка/img_heli/plane_bird/"

#sort_data_dirs(path_main_data, path_drone, path_bird, path_plane, path_heil, path_drone_bird, path_drone_plane, path_drone_heli, path_heli_bird, path_heli_plane, path_background, path_plane_bird)

#rm_txt_or_image_data(f"/home/usr/Документы/img_heli/images_288x288_3_class_V4/images/test/", f"/home/usr/Документы/img_heli/images_288x288_3_class_V4/labels/test/")

# rm_r(f"/home/usr/Видео/saver_detect/plane/")
#path = f"/home/usr/Видео/saver_detect/reb_data_288x288/"

#
# path, path_opt, path_opt_2 = f"/home/usr/Музыка/img_heli_iter_2/dataset_4_classes_iter_1.3/labels/train/", f"/home/usr/Музыка/img_heli_iter_2/dataset_4_classes_iter_1.3/labels/fuck/", f"/home/usr/Музыка/img_heli_iter_2/dataset_4_classes_iter_1.3/images/train/"
#
# lst_txt = [k for i, k in enumerate(os.listdir(path)) if k[-3::] == "txt"]
#
# for j, k in enumerate(lst_txt):
#     rm_empty_txt(path, k, path_opt=path_opt, path_opt_2=path_opt_2)

# #txt_rm(f"/home/usr/Загрузки/dataset_ground_vehicle/filtering_dataset/VISDRONE/VisDrone2019-DET-train/reb_augment/truck_only/", f"/home/usr/Загрузки/dataset_ground_vehicle/filtering_dataset/dataset_augment/images/train/", f"/home/usr/Загрузки/dataset_ground_vehicle/filtering_dataset/dataset_augment/labels/train/")
#
# path = f"/home/usr/Изображения/images_roboflow/path_5/valid/images/"
# #
# lst = [k for i, k in enumerate(os.listdir(path)) if k[-3::] == "txt"]
#
# for i, k in enumerate(lst):
#     edit_classes(path+k)

#write_img(f"/home/usr/my_video-2.mkv", f"/home/usr/Документы/marata_image/")

