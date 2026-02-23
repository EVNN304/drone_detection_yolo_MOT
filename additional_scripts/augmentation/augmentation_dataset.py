import os
import cv2 as cv
import numpy as np
import random
import copy
from ultralytics import YOLO, SAM

class Background_replace:
    def __init__(self, path_data, path_img_background):
        self.path_data, self.path_img_background = path_data, path_img_background
        self.random_pos_flag = True
        self.flag_use_yolo = False
        self.path_mod = f"yolo11x-seg.pt"
        self.output_dir_img = None
        self.output_dir_lbl = None
        self.flag_save = False
        self.check_lables = False
        self.time_delay = 1
        self.flag_show = True
        self.path_sam_model = "sam_l.pt"


    def set_flag_show(self, val:bool):
        self.flag_show = val



    def set_path_sam_model(self, val:str):
        self.path_sam_model = val

    def set_time_delay(self, val:int):
        self.time_delay = val


    def set_path_data(self, val:str):
        self.path_data = val

    def set_flag_show_lables(self, val:bool):
        self.check_lables = val

    def set_flag_save(self, val:bool):
        self.flag_save = val

    def set_out_dir_lbl(self, val:str):
        self.output_dir_lbl = val
        if self.flag_save:
            os.makedirs(self.output_dir_lbl, exist_ok=True)

    def set_out_dir_img(self, val:str):
        self.output_dir_img = val
        if self.flag_save:
            os.makedirs(self.output_dir_img, exist_ok=True)

    def set_flag_use_yolo(self, val:bool):
        self.flag_use_yolo = val

    def set_path_model_yolo(self, val:str):
        self.path_mod = val

    def load_model_yolo(self):
        try:
            return YOLO(self.path_mod)
        except Exception as e:
            print(f"Errr_load_model_yolo: {e.args}, check your path!")
            exit(0)

    def load_model_sam(self):
        try:
            return SAM(self.path_sam_model)
        except Exception as e:
            print(f"Errrr_load_model_sam: {e.args}, check your path!")
            exit(0)

    def check_intersection(self, boxA, boxB):
        xA1, yA1, xA2, yA2 = boxA
        xB1, yB1, xB2, yB2 = boxB
        return not (xA2 <= xB1 or xB2 <= xA1 or yA2 <= yB1 or yB2 <= yA1)

    def find_available_position(self, occupied_boxes, box_width, box_height, bg_width, bg_height):
        step = 10
        for y in range(0, bg_height - box_height, step):
            for x in range(0, bg_width - box_width, step):
                candidate = (x, y, x + box_width, y + box_height)
                if all(not self.check_intersection(candidate, occupied) for occupied in occupied_boxes):
                    return candidate
        return None

    def get_random_position(self, box_w, box_h, bg_w, bg_h, occupied_boxes, max_attempts=100):
        for _ in range(max_attempts):
            x1 = random.randint(0, bg_w - box_w)
            y1 = random.randint(0, bg_h - box_h)
            new_box = (x1, y1, x1 + box_w, y1 + box_h)
            if all(not self.check_intersection(new_box, occupied) for occupied in occupied_boxes):
                return new_box
        return None

    def xywhn_to_xyxy(self, lst, w, h):
        new_lst = []
        for i, k in enumerate(lst):
            obj, xn, yn, wn, hn = k

            x_lft = int((xn - wn / 2) * w)
            y_lft = int((yn - hn / 2) * h)
            x_rgh = int((xn + wn / 2) * w)
            y_rgh = int((yn + hn / 2) * h)

            new_lst.append([obj, x_lft, y_lft, x_rgh, y_rgh])
        return new_lst

    def writer_labels(self, output_ann_path, ann_lines):
        try:
            with open(output_ann_path, 'w') as f:
                for line in ann_lines:
                    f.write(line + "\n")
        except Exception as e:
            print(f"Errr_writer_labels: {e.args} !!")



    def writer_images(self, output_img_path, image_background):
        try:
            cv.imwrite(output_img_path, image_background)
        except Exception as e:
            print(f"Errr_writer_images: {e.args}")

    def grabcut_mask(self, roi):
        try:
            mask = np.zeros(roi.shape[:2], np.uint8)
            rect = (1, 1, roi.shape[1] - 2, roi.shape[0] - 2)
            bgdModel = np.zeros((1, 65), np.float64)
            fgdModel = np.zeros((1, 65), np.float64)
            cv.grabCut(roi, mask, rect, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_RECT)

            kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))

            sure_fg = cv.erode((mask == cv.GC_FGD).astype(np.uint8), kernel, iterations=2)
            sure_bg = cv.dilate((mask == cv.GC_BGD).astype(np.uint8), kernel, iterations=2)

            if np.count_nonzero(sure_fg) == 0 or np.count_nonzero(sure_bg) == 0:
                print("Warning: sure foreground or background empty, skipping mask refinement.")
                mask2 = np.where((mask == cv.GC_FGD) | (mask == cv.GC_PR_FGD), 255, 0).astype('uint8')
                return mask2

            mask[sure_fg == 1] = cv.GC_FGD
            mask[sure_bg == 1] = cv.GC_BGD

            cv.grabCut(roi, mask, None, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_MASK)

            mask2 = np.where((mask == cv.GC_FGD) | (mask == cv.GC_PR_FGD), 255, 0).astype('uint8')

            return mask2
        except Exception as e:
            print(f"Errrrr_grabcut_mask: {e.args}")
            return None


    def transfer_labels_to_new_position(self, original_labels_path, objects_positions, w_bg, h_bg, original_img_width, original_img_height):
        new_labels = []
        with open(original_labels_path, 'r') as f:
            lines = f.readlines()

        for line, (cls_id, x_off, y_off, w_roi, h_roi) in zip(lines, objects_positions):
            parts = line.strip().split()
            cls_orig = int(parts[0])
            x_center_norm = float(parts[1])
            y_center_norm = float(parts[2])
            width_norm = float(parts[3])
            height_norm = float(parts[4])
            x_center_px = x_center_norm * original_img_width
            y_center_px = y_center_norm * original_img_height
            width_px = width_norm * original_img_width
            height_px = height_norm * original_img_height
            x_center_px_new = x_off + (x_center_px - x_off)
            y_center_px_new = y_off + (y_center_px - y_off)

            new_x_center = (x_off + w_roi / 2) / w_bg
            new_y_center = (y_off + h_roi / 2) / h_bg
            new_width = w_roi / w_bg
            new_height = h_roi / h_bg

            new_labels.append(f"{cls_orig} {new_x_center:.6f} {new_y_center:.6f} {new_width:.6f} {new_height:.6f}")

        return new_labels

    def show_image_res(self, img, tag_name:str, time_delay=0):
        cv.imshow(tag_name, img)
        cv.waitKey(time_delay)

    def transfer_object(self):
        lst_images_obj = [k for k in os.listdir(self.path_data) if k[-3:] in ["jpg", "png", "jpeg"]]
        background_images = os.listdir(self.path_img_background)

        if not self.flag_use_yolo:
            for img_obj in lst_images_obj:
                for backgr in background_images:
                    try:
                        image_obj = cv.imread(os.path.join(self.path_data, img_obj))
                        image_background = cv.imread(os.path.join(self.path_img_background, backgr))
                        if image_obj is None or image_background is None:
                            if image_obj is None:
                                print(f"Ошибка загрузки {img_obj}!")
                            if image_background is None:
                                print(f"Ошибка загрузки {image_background}!")
                            continue

                        backgr_name = os.path.splitext(backgr)[0]
                        img_name = os.path.splitext(img_obj)[0]
                        out_img_name = f"{img_name}_on_{backgr_name}.jpg"
                        out_lbl_name = f"{img_name}_on_{backgr_name}.txt"

                        h, w, _ = image_obj.shape
                        bboxes = self.read_txt(img_obj[:-3] + "txt", w, h)
                        if bboxes is None:
                            print(f"BBOXES is None")
                            continue

                        occupied_boxes = []
                        objects_positions = []

                        for bbox in bboxes:
                            obj, x1, y1, x2, y2 = bbox
                            roi_w, roi_h = x2 - x1, y2 - y1
                            h_bg, w_bg, _ = image_background.shape

                            pos = self.get_random_position(roi_w, roi_h, w_bg, h_bg,occupied_boxes) if self.random_pos_flag else self.find_available_position(occupied_boxes, roi_w, roi_h, w_bg, h_bg)
                            if pos is None:
                                print("Нет места для объекта, пропускаем")
                                continue

                            x_off, y_off, _, _ = pos
                            occupied_boxes.append(pos)

                            roi = image_obj[y1:y2, x1:x2]

                            mask = self.grabcut_mask(roi)
                            mask_inv = cv.bitwise_not(mask)

                            bg_roi = image_background[y_off:y_off + roi.shape[0], x_off:x_off + roi.shape[1]]
                            bg_roi = cv.bitwise_and(bg_roi, bg_roi, mask=mask_inv)
                            fg_roi = cv.bitwise_and(roi, roi, mask=mask)

                            dst = cv.add(bg_roi, fg_roi)
                            image_background[y_off:y_off + roi.shape[0], x_off:x_off + roi.shape[1]] = dst

                            objects_positions.append((obj, x_off, y_off, roi_w, roi_h))
                        if self.flag_show:
                            self.show_image_res(image_background, "classic_segment", time_delay=self.time_delay)

                        if self.flag_save:
                            original_labels_path = os.path.splitext(os.path.join(self.path_data, img_obj))[0] + ".txt"
                            new_labels = self.transfer_labels_to_new_position(original_labels_path, objects_positions, w_bg, h_bg, w, h)
                            output_img_path = os.path.join(self.output_dir_img, out_img_name)
                            output_ann_path = os.path.join(self.output_dir_lbl, out_lbl_name)
                            self.writer_images(output_img_path, image_background)
                            self.writer_labels(output_ann_path, new_labels)
                    except Exception as e:
                        print(f"Errrr: {e.args}")
                        continue
        else:
            model_yolo = self.load_model_yolo()
            model_sam = self.load_model_sam()

            for img_obj in lst_images_obj:
                for backgr in background_images:
                    try:
                        image_obj = cv.imread(os.path.join(self.path_data, img_obj))
                        image_background = cv.imread(os.path.join(self.path_img_background, backgr))
                        if image_obj is None or image_background is None:
                            print(f"Ошибка загрузки {img_obj} или {backgr}")
                            continue

                        backgr_name = os.path.splitext(backgr)[0]
                        img_name = os.path.splitext(img_obj)[0]
                        out_img_name = f"{img_name}_on_{backgr_name}.jpg"
                        out_lbl_name = f"{img_name}_on_{backgr_name}.txt"

                        original_labels_path = os.path.splitext(os.path.join(self.path_data, img_obj))[0] + ".txt"
                        try:
                            with open(original_labels_path, 'r') as f:
                                lines = f.readlines()
                            class_ids_from_txt = [int(line.strip().split()[0]) for line in lines]
                        except Exception as e:
                            print(f"ERROR_transfer_object: {e.args}")
                            continue

                        h_bg, w_bg = image_background.shape[:2]
                        h_obj, w_obj = image_obj.shape[:2]

                        results_yolo = model_yolo(image_obj)[0]

                        boxes = results_yolo.boxes.xyxy.cpu().numpy().astype(int)
                        classes = results_yolo.boxes.cls.cpu().numpy().astype(int)
                        occupied_boxes = []

                        objects_positions = []

                        for idx, (cls_id, box) in enumerate(zip(classes, boxes)):
                            x1, y1, x2, y2 = box
                            x1 = max(0, x1)
                            y1 = max(0, y1)
                            x2 = min(w_obj - 1, x2)
                            y2 = min(h_obj - 1, y2)

                            if x2 <= x1 or y2 <= y1:
                                print("Некорректный bbox, пропускаем")
                                continue

                            sam_results = model_sam(image_obj, bboxes=[box.tolist()])[0]

                            sam_mask = sam_results.masks.data.cpu().numpy()[0]
                            mask_bin = (sam_mask > 0.5).astype(np.uint8)
                            kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
                            mask_clean = cv.morphologyEx(mask_bin, cv.MORPH_CLOSE, kernel, iterations=2)
                            mask_clean = cv.morphologyEx(mask_clean, cv.MORPH_OPEN, kernel, iterations=1)

                            roi_mask = mask_clean[y1:y2, x1:x2]
                            roi_obj = image_obj[y1:y2, x1:x2]

                            if roi_mask.shape != roi_obj.shape[:2]:
                                print("Размеры маски и объекта не совпадают, пропускаем")
                                continue

                            roi_h, roi_w = roi_mask.shape

                            pos = self.get_random_position(roi_w, roi_h, w_bg, h_bg,occupied_boxes) if self.random_pos_flag else self.find_available_position(occupied_boxes, roi_w, roi_h, w_bg, h_bg)
                            if pos is None:
                                print("Нет места для объекта, пропускаем")
                                continue

                            x_off, y_off, _, _ = pos
                            occupied_boxes.append(pos)

                            bg_roi = image_background[y_off:y_off + roi_h, x_off:x_off + roi_w]

                            if bg_roi.shape[:2] != roi_mask.shape:
                                print("Размеры ROI фона и маски не совпадают, пропускаем объект")
                                continue

                            mask_inv = cv.bitwise_not(roi_mask * 255)

                            bg_roi = cv.bitwise_and(bg_roi, bg_roi, mask=mask_inv)
                            fg_roi = cv.bitwise_and(roi_obj, roi_obj, mask=roi_mask)

                            dst = cv.add(bg_roi, fg_roi)
                            image_background[y_off:y_off + roi_h, x_off:x_off + roi_w] = dst

                            cls_id_txt = class_ids_from_txt[idx] if idx < len(class_ids_from_txt) else cls_id

                            objects_positions.append((cls_id_txt, x_off, y_off, roi_w, roi_h))

                        image_background_cp = copy.deepcopy(image_background)
                        if (not self.check_lables) and self.flag_show:
                            self.show_image_res(image_background, "sam_segmented", time_delay=self.time_delay)
                        else:
                            for i, k in enumerate(objects_positions):
                                cv.rectangle(image_background_cp, (k[1], k[2]), (k[1] + k[3], k[2] + k[4]), (0, 255, 0), 2)
                                cv.putText(image_background_cp, str(int(k[0])), (k[1], k[2] - 10), cv.FONT_HERSHEY_SIMPLEX,
                                           0.9, (0, 255, 0), 2)
                            if self.flag_show:
                                self.show_image_res(image_background_cp, "sam_segmented", time_delay=self.time_delay)

                        if self.flag_save:
                            new_labels = self.transfer_labels_to_new_position(original_labels_path, objects_positions, w_bg,
                                                                              h_bg, w_obj, h_obj)
                            output_img_path = os.path.join(self.output_dir_img, out_img_name)
                            output_ann_path = os.path.join(self.output_dir_lbl, out_lbl_name)
                            self.writer_images(output_img_path, image_background)
                            self.writer_labels(output_ann_path, new_labels)
                    except Exception as e:
                        print(f"Errror: {e.args}")
                        continue


    def read_txt(self, name, w, h):
        lst = []
        try:
            with open(self.path_data + name, "r") as file:
                for i, k in enumerate(file):
                    b = list(map(float, k.split()))
                    b[0] = int(b[0])
                    lst.append(b)
            return self.xywhn_to_xyxy(lst, w, h)
        except Exception as e:
            print(f"Errr_read_txt: {e.args}")
            return None

class Affine_augment:
    def __init__(self, path_data, path_new_folder):
        self.path_data = path_data
        self.path_new_folder = path_new_folder
        self.step_angle = 25
        self.time_delay = 1
        self.flag_imshow, self.flag_use_resize = True, True
        self.img_name = f"_AUGMENT_"
        self.resize_img = (640, 640)

    def set_time_delay(self, val:int):
        self.time_delay = val

    def set_img_name(self, val:str):
        self.img_name = val

    def set_resize_wh(self, val:tuple):
        self.resize_img = val

    def set_use_resize(self, val:bool):
        self.flag_use_resize = val

    def set_flag_imshow(self, val:bool):
        self.flag_imshow = val

    def set_step_angle(self, val:int):
        self.step_angle = val

    def set_path_new_floder(self, val:str):
        self.path_new_folder = val
        os.makedirs(self.path_new_folder, exist_ok=True)

    def write_new_image(self, folder, name_image, images):
        if self.flag_use_resize:
            cv.resize(images, self.resize_img)
        cv.imwrite(folder+name_image+'.jpg', images)

    def write_new_txt(self, folder, name_txt, bboxes):
        for i, k in enumerate(bboxes):
            with open(folder+name_txt+'.txt', "a") as file:
                st = " ".join(list(map(str, k))) +"\n"
                file.write(st)



    def rotate_bbox(self, lst_det, image_shape, image, angle, folder_new, name):

        center = (image_shape[1] // 2, image_shape[0] // 2)
        b = 0

        for angl in range(1, angle + 1, self.step_angle):
            b += 1
            ls = []
            M = cv.getRotationMatrix2D(center, angl, 1.0)
            rotated_img = cv.warpAffine(image, M, (image_shape[1], image_shape[0]))
            root_img = copy.deepcopy(rotated_img)
            for i, bbox in enumerate(lst_det):
                obj, x, y, w, h = bbox
                x_abs = x * image_shape[1]
                y_abs = y * image_shape[0]
                w_abs = w * image_shape[1]
                h_abs = h * image_shape[0]

                coords = np.array([
                    [x_abs - w_abs / 2, y_abs - h_abs / 2],
                    [x_abs + w_abs / 2, y_abs - h_abs / 2],
                    [x_abs - w_abs / 2, y_abs + h_abs / 2],
                    [x_abs + w_abs / 2, y_abs + h_abs / 2]
                ], dtype=np.float32)

                coords = np.hstack((coords, np.ones((4, 1), dtype=np.float32)))
                rotated_coords = np.dot(M, coords.T).T

                x1 = np.min(rotated_coords[:, 0])
                y1 = np.min(rotated_coords[:, 1])
                x2 = np.max(rotated_coords[:, 0])
                y2 = np.max(rotated_coords[:, 1])

                if (x1 < 0 or x2 > image_shape[1] - 1 or y1 < 0 or y2 > image_shape[0] - 1):
                    continue

                x_norm = (x1 + (x2 - x1) / 2) / image_shape[1]
                y_norm = (y1 + (y2 - y1) / 2) / image_shape[0]
                w_norm = (x2 - x1) / image_shape[1]
                h_norm = (y2 - y1) / image_shape[0]

                ls.append([obj, x_norm, y_norm, w_norm, h_norm])

                self.write_new_image(folder_new, f"{name}_{b}{self.img_name}", root_img)
                self.write_new_txt(folder_new, f"{name}_{b}{self.img_name}", ls)
                if self.flag_imshow:
                    cv.rectangle(rotated_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                    cv.imshow("root_image", cv.resize(rotated_img, (800, 800)))
                    cv.waitKey(self.time_delay)



    def read_dataset(self, path):
        list_dir = sorted(os.listdir(path))
        lst_detect = []
        for i, k in enumerate(list_dir):
            name, *rsh = k.split(".")
            name = name+"."+".".join(rsh[:-1])
            name = name if name[-1] == "." else name+"."
            if rsh[-1] == "txt":
                lst = []
                with open(path + name + "txt", "r") as file:
                    for i, k in enumerate(file):
                        b = list(map(float, k.split()))
                        b[0] = int(b[0])
                        lst.append(b)
                lst_detect[-1].append(lst)
            else:
                lst_detect.append([path + name + rsh[-1]])
        return lst_detect








flag_use_affine = False


#path = "sputnik/"
path = "/home/usr/Документы/TANKS_DATA_T72-90 ABRAMS MERK/T64-90/test/images/"
folder_new = "/home/usr/Документы/TANKS_DATA_T72-90 ABRAMS MERK/T64-90/test/reeb_root/"
if flag_use_affine:
    # path = "/home/usr/Изображения/ships_unmanned/GloriousStar_0524.v1i.yolov11/valid/images/"
    # folder_new = "/home/usr/Изображения/ships_unmanned/GloriousStar_0524.v1i.yolov11/valid/data_augment/"
    cls_affine = Affine_augment(path, folder_new)
    cls_affine.set_step_angle(60)
    cls_affine.set_path_new_floder(folder_new)
    lst_det = cls_affine.read_dataset(path)
    for i, k in enumerate(lst_det):
        try:
            name_image, list_detect = k[0], k[-1]
            *fld, names = name_image.split("/")
            names_split = names.split(".")
            image = cv.imread(name_image,  cv.IMREAD_UNCHANGED)
            image_shape = image.shape
            cord_rotate = cls_affine.rotate_bbox(list_detect, image_shape, image, 359, folder_new, names_split[0])
        except Exception as e:
            print(f"Error:{e.args}")

else:

    cls_repl = Background_replace(f"/home/usr/Изображения/ships_unmanned/GloriousStar_0524.v1i.yolov11/valid/images/", f"/home/usr/Видео/Записи экрана/object_detectionv2_val.v1i.yolov11/valid/images/")
    cls_repl.set_flag_use_yolo(True)
    cls_repl.set_time_delay(30)
    cls_repl.set_path_model_yolo(f"yolo11x-seg.pt") # sam_l.pt    yolo11x-seg.pt
    cls_repl.set_flag_save(True)
    cls_repl.set_flag_show_lables(True)
    cls_repl.set_out_dir_lbl(f"/home/usr/Видео/Записи экрана/lbl2/")
    cls_repl.set_out_dir_img(f"/home/usr/Видео/Записи экрана/img2/")
    cls_repl.transfer_object()