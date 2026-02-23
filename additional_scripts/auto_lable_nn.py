import copy
import multiprocessing as mp
import cv2 as cv
from ultralytics import YOLO



class Yolo_nn:
    def __init__(self, path_model, q_imgae:mp.Queue, q_to_save:mp.Queue):
        self.path_model, self.show, self.time_delay, self.save_res = path_model, True, 10, True
        self.q_image, self.q_to_save = q_imgae, q_to_save


    def set_show_pred(self, b:bool):
        self.show = b

    def set_save_res(self, b:bool):
        self.save_res = b

    def set_time_delay(self, val:int):
        self.time_delay = val

    def set_q_image(self, q:mp.Queue):
        self.q_image = q

    def set_q_to_save(self, q:mp.Queue):
        self.q_to_save = q

    def set_path_model(self, val:str):
        self.path_model = val

    def run_process(self, daemon=False):
        proc = mp.Process(target=self.predict, args=(), daemon=daemon)
        proc.start()

    def load_model(self):
        return YOLO(self.path_model)

    def show_res(self, res, img, dct):

        for bbox, clss, cnf in zip(res[0], res[1], res[2]):
            cv.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 4)
            cv.putText(img, f'{dct[int(clss)]}[{round(cnf, 2)}]', (int(bbox[0]), int(bbox[1]) - 10), 3, 1.3, (0, 0, 255), 2)
        cv.imshow("res_pred", img)
        cv.waitKey(self.time_delay)

    def predict(self):

        model = self.load_model()
        dct_name = model.names
        while True:
            if not self.q_image.empty():
                images = self.q_image.get()
                img_copy = copy.deepcopy(images)
                result = model(images)
                bboxes, cls, conf, bboxes_norm = result[0].boxes.xyxy.cpu().tolist(), result[0].boxes.cls.cpu().tolist(), result[0].boxes.conf.cpu().tolist(), result[0].boxes.xywhn.cpu().tolist()
                if self.show:
                    self.show_res([bboxes, cls, conf], images, dct_name)

                if self.save_res:
                    if self.q_to_save.empty():
                        self.q_to_save.put([img_copy, [cls, bboxes_norm]])


class Saver_label_image:
    def __init__(self, path_save_image, path_save_lable, q_save:mp.Queue):
        self.path_save_lable, self.path_save_image, self.q_save, self.name_file = path_save_lable, path_save_image,  q_save, "truck_3"


    def set_path_save_lable(self, pth:str):
        self.path_save_lable = pth

    def set_path_save_images(self, pth:str):
        self.path_save_image = pth

    def set_q_save(self, q:mp.Queue):
        self.q_save = q

    def set_name_file(self, val:str):
        self.name_file = val

    def run_process(self, daemon=False):
        proc = mp.Process(target=self.save_lable, args=(), daemon=daemon)
        proc.start()


    def writer_txt(self, path, cls, labels):

        with open(self.path_save_lable+path, "a") as file:
            for cl, bbox in zip(cls, labels):
                print(int(cl), bbox)
                cl = 2
                st = str(int(cl)) + " " + " ".join(list(map(str, bbox)))+"\n"
                file.write(st)

    def writer_images(self, pth, images):
        cv.imwrite(self.path_save_image+pth, images)

    def save_lable(self):
        c = 0
        while True:
            if not self.q_save.empty():
                images, labels = self.q_save.get()
                c += 1
                self.writer_txt(f"{self.name_file}_{c}.txt", labels[0], labels[1])
                self.writer_images(f"{self.name_file}_{c}.jpg", images)



class Video_runner:
    def __init__(self, path_to_video, q_images:mp.Queue):
        self.path_to_video, self.q_images = path_to_video, q_images


    def set_path_vide(self, vid:str):
        self.path_to_video = vid


    def set_q_image(self, val:mp.Queue):
        self.q_images = val

    def run_process(self, daemon=False):
        proc = mp.Process(target=self.run_video, args=(), daemon=daemon)
        proc.start()


    def run_video(self):

        cap = cv.VideoCapture(self.path_to_video)

        while cap.isOpened():
            flag, images = cap.read()
            #if self.q_images.empty():
            self.q_images.put(images)



if __name__ == '__main__':
    path_label, path_images, path_videos = f"/home/usr/Документы/TANKS_DATA_T72-90 ABRAMS MERK/ABRAMS/lbl/", f"/home/usr/Документы/TANKS_DATA_T72-90 ABRAMS MERK/ABRAMS/img/", f"/home/usr/Документы/TANKS_DATA_T72-90 ABRAMS MERK/MERK/stock-footage-abrams-power-the-most-iconic-tank-shootouts.webm"
    q_image, q_to_label = mp.Queue(maxsize=1), mp.Queue(maxsize=1)  # [f"stock-footage-t-or-vladimir-the-russian-medium-and-main-tank.webm", stock-footage-military-army-tank-driving-aerial-view.webm f"stock-footage-circa-u-s-army-nd-and-rd-brigade-combat-teams-conduct-armored-vehicle-gunnery-practice-at.webm", f"stock-footage-extra-wild-shot-military-tank-firing-concept-of-war-and-conflict.webm", f"stock-footage-fort-stewart-ga-u-s-army-soldiers-and-abrams-tanks-st-armored-brigade-combat-team.webm"

    # f"/home/usr/PycharmProjects/yolo_proj/ultralytics/runs/detect/train/weights/best.pt"/home/usr/PycharmProjects/yolo_proj/ultralytics/runs/detect/train15/weights/best.pt
    yolo_nn = Yolo_nn( f"/home/usr/Видео/train15_type_tank_large/weights/best.pt", q_image, q_to_label)   # f"bb.pt"
    saver = Saver_label_image(path_images, path_label, q_to_label)
    saver.set_name_file("mrk_12_")
    video = Video_runner(path_videos, q_image)

    yolo_nn.run_process()
    video.run_process()
    saver.run_process()