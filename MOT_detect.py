import cv2
import time
import multiprocessing as mp
import numpy as np
from pathlib import Path
from boxmot import BoostTrack, BotSort, StrongSort, DeepOcSort, ByteTrack, HybridSort, OcSort




class MOT:
    def __init__(self, q_to_mot:mp.Queue, method_mot=0):
        self.q_mot = q_to_mot
        self.mot_method = method_mot
        self.mot = None
        self.path_reid = 'osnet_x0_25_msmt17.pt'
        #self.path_reid = '/home/usr/Загрузки/resnet50_.onnx'
        self.flag_show_track = False
        self.half = True
        self.device = 0

    def set_mot_method(self, val:int):
        self.mot_method = val

    def set_path_weights_reid(self, val:str):
        self.path_reid = val
        print(f"Choice_weights_reid:{self.path_reid}")

    def set_flag_half(self, val:bool):
        self.half = val

    def set_device(self, val:int):
        self.device = val

    def set_flag_show_track(self, val:bool):
        self.flag_show_track = val


    def choice_mot(self):

        dict_mot = {0:(BoostTrack(reid_weights=Path(self.path_reid), device=self.device, half=self.half), "Choice method mot BoostTrack!"), 1:(BotSort(reid_weights=Path(self.path_reid), device=self.device, half=self.half), "Choice method mot BotSort!"), 2:(StrongSort(reid_weights=Path(self.path_reid), device=self.device, half=self.half), "Choice method mot StrongSort!"), 3:(DeepOcSort(reid_weights=Path(self.path_reid), device=self.device, half=self.half), "Choice method mot DeepOcSort!"), 4:(ByteTrack(reid_weights=Path(self.path_reid), device=self.device, half=self.half), "Choice method mot ByteTrack!"), 5:(HybridSort(reid_weights=Path(self.path_reid), device=self.device, half=self.half), "Choice method mot HybridSort!"), 6:(OcSort(reid_weights=Path(self.path_reid), device=self.device, half=self.half), "Choice method mot OcSort!")}
        try:
            self.mot, st = dict_mot[self.mot_method]
            print(st)
            return self.mot, True
        except Exception as e:
            print(f"Errrr_choice_mot:{e.args}")
            self.mot, st = dict_mot[5]
            print(f"Default_mot_choice:", st)
            return  self.mot, True


    def run_process(self, daemon=False):

        prc = mp.Process(target=self.main_worker, args=(), daemon=daemon)
        prc.start()

    def main_worker(self):

        tracker, flg = self.choice_mot()
        frame_count = 0
        start_time = time.time()
        while flg:
            if not self.q_mot.empty():
                try:
                    frame, detections = self.q_mot.get()
                    dets = []
                    for i, k in enumerate(detections):
                        x_lft, y_lft = k.left_top()
                        x_rgh, y_rgh = k.right_bottom()
                        conf, obj_cls = k.p, k.obj_class



                        dets.append([x_lft, y_lft, x_rgh, y_rgh, conf, obj_cls])

                    res = tracker.update(np.array(dets), frame)
                    tracker.plot_results(frame, show_trajectories=self.flag_show_track)



                    frame_count += 1
                    current_time = time.time()
                    fps = frame_count / (current_time - start_time)
                    cv2.putText(frame, f'FPS: {fps:.1f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.imshow('BoXMOT + YOLOv11x + SAHI', frame)
                    cv2.waitKey(1)
                except Exception as e:
                    print(f"Errrrrrrrrrrr_MOT_prc: {e.args}")



