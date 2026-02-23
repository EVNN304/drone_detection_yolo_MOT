
from __future__ import absolute_import
import numpy as np
from geometry_lib import *
import cv2 as cv
import multiprocessing as mp
import copy
from yolov8_lib_2 import collect_single_step_dets


from Eval.siamfc import TrackerSiamFC


def run_nets(q_control, q_from_neuro, q_cord):
    neuro_data_collector = mp.Process(target=collect_n_cast_neuro2,
                                      args=(q_control, q_from_neuro, q_cord,))
    neuro_data_collector.start()





def iou_thresh(bbox1, bbox2, threshold=0.2):
    """
    Calculates the intersection-over-union of two bounding boxes and compares it to a threshold.
    Args:
        bbox1 (numpy.array, list of floats): bounding box in format x,y,w,h.
        bbox2 (numpy.array, list of floats): bounding box in format x,y,w,h.
        threshold (float): minimal IoU to consider as intersection (default=0.5).
    Returns:
        bool: True if IoU >= threshold, otherwise False.
    """
    bbox1 = [float(x) for x in bbox1]
    bbox2 = [float(x) for x in bbox2]

    (x0_1, y0_1, w1_1, h1_1) = bbox1
    (x0_2, y0_2, w1_2, h1_2) = bbox2
    x1_1 = x0_1 + w1_1
    x1_2 = x0_2 + w1_2
    y1_1 = y0_1 + h1_1
    y1_2 = y0_2 + h1_2

    # get the overlap rectangle
    overlap_x0 = max(x0_1, x0_2)
    overlap_y0 = max(y0_1, y0_2)
    overlap_x1 = min(x1_1, x1_2)
    overlap_y1 = min(y1_1, y1_2)

    # check if there is an overlap
    if overlap_x1 - overlap_x0 <= 0 or overlap_y1 - overlap_y0 <= 0:
        return False

    # calculate intersection and union
    size_1 = w1_1 * h1_1
    size_2 = w1_2 * h1_2
    size_intersection = (overlap_x1 - overlap_x0) * (overlap_y1 - overlap_y0)
    size_union = size_1 + size_2 - size_intersection

    iou_value = size_intersection / size_union
    print("STATUS_BBOXES_IOU", iou_value >= threshold)
    return iou_value >= threshold


def iou(bbox1, bbox2):
    """
    Calculates the intersection-over-union of two bounding boxes.
    Args:
        bbox1 (numpy.array, list of floats): bounding box in format x,y,w,h.
        bbox2 (numpy.array, list of floats): bounding box in format x,y,w,h.
    Returns:
        int: intersection-over-onion of bbox1, bbox2
    """
    bbox1 = [float(x) for x in bbox1]
    bbox2 = [float(x) for x in bbox2]

    (x0_1, y0_1, w1_1, h1_1) = bbox1
    (x0_2, y0_2, w1_2, h1_2) = bbox2
    x1_1 = x0_1 + w1_1
    x1_2 = x0_2 + w1_2
    y1_1 = y0_1 + h1_1
    y1_2 = y0_2 + h1_2
    # get the overlap rectangle
    overlap_x0 = max(x0_1, x0_2)
    overlap_y0 = max(y0_1, y0_2)
    overlap_x1 = min(x1_1, x1_2)
    overlap_y1 = min(y1_1, y1_2)

    # check if there is an overlap
    if overlap_x1 - overlap_x0 <= 0 or overlap_y1 - overlap_y0 <= 0:
        return 0

    # if yes, calculate the ratio of the overlap to each ROI size and the unified size
    size_1 = (x1_1 - x0_1) * (y1_1 - y0_1)
    size_2 = (x1_2 - x0_2) * (y1_2 - y0_2)
    size_intersection = (overlap_x1 - overlap_x0) * (overlap_y1 - overlap_y0)
    size_union = size_1 + size_2 - size_intersection

    return size_intersection / size_union


def not_exist(pred):
    return len(pred) == 1 and pred == 0


def eval(out_res, label_res):
    measure_per_frame = []
    for _pred, _gt, _exist in zip(out_res, label_res['gt_rect'], label_res['exist']):
        measure_per_frame.append(not_exist(_pred) if not _exist else iou(_pred, _gt))
    return np.mean(measure_per_frame)









def collect_n_cast_neuro2(q_control: mp.Queue, q_dets: mp.Queue, q_cord:mp.Queue):
    net_path = '/home/usr/PycharmProjects/yolo_proj/ultralytics/Eval/model.pth'
    tracker = TrackerSiamFC(net_path=net_path)
    status = 0
    out_res = []
    name = "visible"
    out = None

    while True:
        all_detections = []
        args_control = q_control.get()
        frame_id = args_control[0]
        n_subframes = args_control[1]
        meta = args_control[2]
        frame = args_control[3]
        x_az, y_el, size_obj = args_control[4], args_control[5], args_control[6]
        fr = copy.deepcopy(frame)
        print("get_cord", x_az, y_el, fr.shape)

        filtered_dets = collect_single_step_dets(q_dets, frame_id, n_subframes)
        h, w = fr.shape
        #print("FT", filtered_dets)
        dist, filter_detect = [], []
        if filtered_dets:

            for det_n, det in enumerate(filtered_dets):
                x, y = det.get_int_center()
                wei, hei = det.get_wh()
                #x_lft, y_lft = det.left_top()
                #x_r, y_r = det.right_bottom()
                if int(det.obj_class) == 0:
                    #filter_detect.append([det.obj_class, det.p, x_lft, y_lft, x_r, y_r])
                    filter_detect.append([det.obj_class, det.p, x, y, int(wei), int(hei)])

                #filter_detect.append(det)
                det.draw(frame)
                cord_calc = cover_pt_by_area((x, y), area_w_h=[288, 288], limit_box=[0, 0, w, h])
                cv.imshow("fragmet", frame[cord_calc[1]:cord_calc[3], cord_calc[0]:cord_calc[2]])
                cv.waitKey(1)

        filter_detect = sorted(filter_detect, key=lambda x: x[1])

        if status == 0 and filter_detect:
            tracker.init(cv.cvtColor(fr, cv.COLOR_GRAY2RGB), (filter_detect[-1][2:][0], filter_detect[-1][2:][1], filter_detect[-1][2:][2], filter_detect[-1][2:][3]))  # initialization
            out = (filter_detect[-1][2:][0], filter_detect[-1][2:][1], filter_detect[-1][2:][2], filter_detect[-1][2:][3])
            out_res.append(out)
            status = 1

        else:
            out = tracker.update(cv.cvtColor(fr, cv.COLOR_GRAY2RGB))  # tracking
            out_res.append(out.tolist())
            if filter_detect:
                print("FFFFFFFFFFFFFFFFFFF", list(map(int, out.tolist())), filter_detect[-1][2:])
                if iou_thresh(filter_detect[-1][2:], list(map(int, out.tolist()))):
                    print("ITS_OK")
                else:
                    status = 0

        if not (out is None):
            cv.rectangle(fr, (int(out[0] - (out[2] // 2)), int(out[1] - (out[3] // 2))),
                          (int(out[0] + (out[2] // 2)), int(out[1] + (out[3] // 2))), (0, 255, 255))
        print("FT", filter_detect)
        # if filter_detect:
        #     q_cord.put(filter_detect[-1])
        cv.imshow(f"{name}", cv.resize(fr, (1200, 1200)))
        cv.imshow("NEURO", cv.resize(frame, (1200, 1200)))
        cv.waitKey(33)

def cropper_image(q_track:mp.Queue, q_control:mp.Queue, q_to_neural:mp.Queue, size_images=[800, 800], area_bbox=[0, 0, 1920, 1080]):


    cords_cropp = calc_scan_areas(area_bbox, window_w_h=size_images, overlay=(0.1, 0.1))
    while True:
        if not q_track.empty():
            images, meta, frame_count = q_track.get()
            for i, k in enumerate(cords_cropp):

                q_control.put([frame_count, 1, meta, images, k[0], k[1], k])
                q_to_neural.put((cv.cvtColor(images[k[1]:k[3], k[0]:k[2]],cv.COLOR_GRAY2RGB),[k[0], k[1], k[2], k[3]],[frame_count, len([[k[0], k[1], k[2], k[3]]]), 1]))


def start_process(q_in, q_control, q_to_neural, size_img, area_bbox, daemon=False):
    prc_1 = mp.Process(target=cropper_image, args=(q_in, q_control, q_to_neural, size_img, area_bbox), daemon=daemon)
    prc_1.start()