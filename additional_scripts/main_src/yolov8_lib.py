import multiprocessing as mp
import threading as thr
from queue import Queue
from ultralytics import YOLO
from toolset import Detection_centered



class Yolov8_custom():
    def __init__(self, path_weights="best_11.pt"):
        self.conf = 0.6
        self.path_weights = path_weights
        self._q_proc_out = Queue(1)
        self._q_proc_in = Queue(1)
        self.net = YOLO(self.path_weights)




    def set_output_queue(self,out_q):
        self._q_proc_out = out_q







    def detect_from_q(self):
        darknet_im, original_area, args = self._q_proc_in.get()
        # print('______________________')
        # print('received area: ',original_area)
        # areas_test = glib.calc_scan_areas([0, 0, 4504, 4504], (800, 800), overlay=(0.05, 0.05))
        # idx = args[2]
        # print(f'should be: {areas_test[idx]}')
        detections = detect_image_custom(self.net, darknet_im, self.conf, original_area_box=original_area)

        return detections, args




    def detect_stream(self):
        run = True
        while run:
            # print('detect_alive')
            try:
                dets, args = self.detect_from_q()
                self._q_proc_out.put((dets, args))
            except Exception as e:
                print(e.args)

    def run_detect_stream(self):
        th = thr.Thread(target=self.detect_stream, args=(), daemon=True)
        th.start()



    def push_image(self, py_image, original_area, args=None):

        self._q_proc_in.put([py_image, original_area, args])






    def get_from_proc_stream(self):
        dets, args = self._q_proc_out.get()
        return dets, args

class Multi_net_process:
    def __init__(self, count_net, q_in, q_out):
        self.q_in, self.q_out, self.count_net = q_in, q_out, count_net
        self.net_qs, self.net_process = [mp.Queue(1) for _ in range(self.count_net)], []


    def config_run_nets(self):
        for i in range(self.count_net):
            process = mp.Process(target=self.init_and_run_single_net, args=(self.net_qs[i], self.q_out), daemon=True)
            process.start()

        self.run_net_commutator()


    
    def init_and_run_single_net(self,net_q,out_q):
        net = Yolov8_custom()
        net.set_output_queue(out_q)
        net.run_detect_stream()
        run = True
        count = 0
        while run:
            # print("!23")
            image,original_area,args = net_q.get()
            # print(f'Init an run single net {count}, {original_area}')
            count+=1
            # print(np.shape(image))
            net.push_image(image, original_area, args)
    
    

    def net_commutator(self):
        run = True
        last_net_idx = -1
        while run:
            last_net_idx+=1
            last_net_idx%=self.count_net
            subframe, area, args = self.q_in.get()
            self.net_qs[last_net_idx].put((subframe, area, args))

    def run_net_commutator(self):
        pr = thr.Thread(target=self.net_commutator, args=(), daemon=True)
        pr.start()



def detect_image_custom(network, image, conf=0.6, original_area_box=None):
    results = network(image, conf=conf)
    lst_det = [[cls, conf, k[0], k[1], k[2], k[3]] for cls, conf, k in zip(results[0].boxes.cls.cpu().tolist(), results[0].boxes.conf.cpu().tolist(), results[0].boxes.xyxy.cpu().tolist())]
    lst_global = []
    if original_area_box:
        for i, k in enumerate(lst_det):
            x1_global, y1_global, x2_global, y2_global = int(k[2]+original_area_box[0]), int(k[3]+original_area_box[1]), int(k[4]+original_area_box[0]), int(k[5]+original_area_box[1])
            lst_global.append([int(k[0]), k[1], x1_global, y1_global, x2_global, y2_global])
    else:
        return lst_det
    return lst_global


def collect_single_step_dets(q_dets,frame_id,n_subframes):
    count = 0
    search = True
    all_detections = []
    filtered_dets = []
    while search:
        dets, args = q_dets.get()
        #print(f'args:{args} {dets}')

        if args[0] == frame_id:
            count += 1
            if len(dets) > 0:
                all_detections.extend(dets)
            if count == n_subframes:
                search = False
        else:
            q_dets.put((dets, args))

    return all_detections