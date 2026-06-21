from additional_scripts.save_modules.Module_save_detection_v2 import *
from loggers import *
from render_utils import BlockRenderer
from shm_utils import SharedSimpleFrameReader

class Yolo_inits_batch:
    def __init__(self, q_from_neuro: mp.Queue, saved_mode=None, names_files=None,
                 name_folder="TEST_SAVE_SERVER_", render_cfg=None, classes_names={},
                 shm_neuro_name=None, frame_h=1080, frame_w=1920):  # ← ДОБАВИТЬ ПАРАМЕТРЫ
        self.pix_x, self.pix_y = 30, 30
        self.name_window = "w"
        self.name_folder = name_folder
        self.q_from_neuro = q_from_neuro
        self.saved_mode = saved_mode
        self.udp_caster = None
        self.size_cut_w = 288
        self.size_cut_h = 288
        self.min_conf = 0.65

        self.names_files = names_files
        self.name_classes = classes_names
        self.logger = logging.getLogger(__name__)
        self.renderer = BlockRenderer(render_cfg or {}, classes_names)

        # Shared Memory для чтения кадра от YOLO
        self.shm_neuro_reader = None
        if shm_neuro_name:
            self.shm_neuro_reader = SharedSimpleFrameReader(frame_h, frame_w, shm_neuro_name)
            self.logger.info("✅ Neuro: SharedSimpleFrameReader создан")

    def set_name_window(self, val):
        self.name_window += val

    def set_name_save(self, val: str):
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

    def set_size_cut_w(self, val: int):
        self.size_cut_w = val

    def set_size_cut_h(self, val: int):
        self.size_cut_h = val

    def set_min_conf(self, val: float):
        self.min_conf = val

    def run_saver(self):
        process_save_det = Save_detect(self.saved_mode, classes_names=self.name_classes)
        process_save_det.set_names_file(self.names_files)
        process_save_det.set_path_save(self.name_folder)
        process_save_det.set_size_cut_w(self.size_cut_w)
        process_save_det.set_size_cut_h(self.size_cut_h)
        process_save_det.set_min_conf(self.min_conf)
        process_save_det.main_start_process()
        self.logger.info(f"STARTED: {self.name_classes}")

    def run_nets(self, daemon=True):
        if self.saved_mode != None:
            self.logger.info("RUNING_SAVE_DETECT!")
            self.saved_mode = mp.JoinableQueue(1)
            self.run_saver()

        neuro_data_collector = mp.Process(target=self.collect_n_cast_neuro,
                                          args=(self.q_from_neuro, self.saved_mode),
                                          daemon=daemon)
        neuro_data_collector.start()
        return neuro_data_collector

    def collect_n_cast_neuro(self, q_dets: mp.Queue, saved_mode):
        if not logging.getLogger().handlers:
            setup_logging()

        start_time = time.time()
        fr_c = 0

        while True:
            if not q_dets.empty():
                data = q_dets.get()

                # Проверяем формат: [True, predictions] = кадр в SHM
                if isinstance(data, list) and len(data) == 2 and isinstance(data[0], bool) and data[0]:
                    predictions = data[1]
                    # Читаем кадр из Shared Memory
                    if self.shm_neuro_reader:
                        frame = self.shm_neuro_reader.read_frame()
                    else:
                        continue
                else:
                    # Старый формат: [frame, predictions]
                    frame, predictions = data

                copy_img = copy.deepcopy(frame)
                fr_c += 1

                for det_n, det in enumerate(predictions):
                    x_lft, y_lft = det.left_top()
                    x_rght, y_rght = det.right_bottom()
                    conf, obj = det.p, det.obj_class
                    cv.rectangle(frame, (x_lft, y_lft), (x_rght, y_rght), (0, 0, 255), 6)
                    cv.putText(frame, f'{self.name_classes[obj]}[{round(conf, 3)}]',
                                (x_lft, y_lft - 10), 3, 1.3, (0, 0, 255), 2)

                current_time = time.time()
                fps = 1.0 / (current_time - start_time) if fr_c > 0 else 0.0
                start_time = current_time

                self.renderer.show(frame, f"DET FPS: {fps:.2f}")
                self.logger.info(f"FPS_DET: {fps}")

                if saved_mode != None:
                    if saved_mode.empty():
                        if predictions:
                            saved_mode.put([copy_img, predictions, True])