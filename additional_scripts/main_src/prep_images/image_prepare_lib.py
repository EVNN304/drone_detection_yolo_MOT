import cv2 as cv
import numpy as np
import additional_scripts.main_src.geometry_and_math.geometry_lib as glib
import time
def binarize_with_std(image,inner_area_size = 51,outer_area_size = 101,std_thresh = 5.0,bin_img = None):
    # std_thresh = 10.0
    # inner_area_size = 200
    # outer_area_size = 400
    create_bin = True
    try:
        if np.shape(bin_img) == np.shape(image):
            create_bin = False
        else:
            create_bin = True
    except:

        create_bin = True

    # if bin_img:
    #     if np.shape(bin_img)== np.shape(image):
    #         create_bin = False
    if create_bin:
        bin_img = np.ones(np.shape(image), dtype=np.uint8)

    h,w = np.shape(image)
    print(f'высота {h}, ширина {w}')
    margin = int((outer_area_size-inner_area_size)/2)
    start_t = time.time()
    for inner_cell_y1 in range(0,h,inner_area_size):
        inner_cell_y2 = min(inner_cell_y1+inner_area_size,h)
        outer_cell_y1 = max(0,inner_cell_y1-margin)
        outer_cell_y2 = outer_cell_y1+outer_area_size
        if outer_cell_y2>h:
            outer_cell_y2 = h
            outer_cell_y1 = outer_cell_y2-outer_area_size
        for inner_cell_x1 in range(0,w, inner_area_size):
            inner_cell_x2 = min(inner_cell_x1+inner_area_size,w)
            outer_cell_x1 = max(0, inner_cell_x1 - margin)
            outer_cell_x2 = outer_cell_x1 + outer_area_size
            if outer_cell_x2 > w:
                outer_cell_x2 = w
                outer_cell_x1 = outer_cell_x2 - outer_area_size
            # print(f'outer bbox {outer_cell_x1, outer_cell_x2, outer_cell_y1, outer_cell_y2}')
            outer_m = np.mean(image[outer_cell_y1:outer_cell_y2,outer_cell_x1:outer_cell_x2])
            outer_std = np.std(image[outer_cell_y1:outer_cell_y2,outer_cell_x1:outer_cell_x2])
            bin_img[inner_cell_y1:inner_cell_y2, inner_cell_x1:inner_cell_x2] = np.where((abs(image[inner_cell_y1:inner_cell_y2, inner_cell_x1:inner_cell_x2]-outer_m)/outer_std)>std_thresh,image[inner_cell_y1:inner_cell_y2, inner_cell_x1:inner_cell_x2], 0)
            # bin_img[inner_cell_y1:inner_cell_y2,inner_cell_x1:inner_cell_x2] = np.where(image[inner_cell_y1:inner_cell_y2,inner_cell_x1:inner_cell_x2]>(outer_m+std_thresh*outer_std),image[inner_cell_y1:inner_cell_y2,inner_cell_x1:inner_cell_x2],0)
    print(f'elapsed: {time.time()-start_t}')

    cv.threshold(bin_img, 1, 200, cv.THRESH_BINARY,bin_img)
    return bin_img

def standart_proc(dif, thresh = 10,erode_size = 3,dilate_size = 16):
    kernel = np.ones((erode_size, erode_size), np.uint8)
    # kernel = np.ones((5, 5), np.uint8)
    kernel2 = np.ones((dilate_size, dilate_size), np.uint8)
    # thresh = 10
    # thresh = 25
    cv.threshold(dif, thresh, 100, cv.THRESH_BINARY, dif)
    '''Эрозия'''
    cv.erode(dif, kernel, dif, iterations=1)
    '''Дилатация'''
    cv.dilate(dif, kernel2, dif, iterations=2)


class Contour_finder:
    def __init__(self,w,h):
        self.w, self.h = w, h
        self.prev_image = np.zeros((self.h,self.w),np.uint8)
        self.diff1 = np.zeros((self.h, self.w),np.uint8)
        self.diff2 = np.zeros((self.h, self.w),np.uint8)
        self.diff_arr = [self.diff1, self.diff2]
        self.current_idx = 0
        self.prev_direction_params =[0,0] #Параметры поворотов для отслеживания движения [az,el]
        self.stable_count = 0               #Счетчик стационарных кадров (считает до 3, чтобы не выдавать контуры при движении)

        self.simple_bin_thresh = 4
        self.std_bin_thresh = 4
        self.binarization_mode = 0 #0 - простая бинаризация, 1 = бинаризация по стандартному отклонению
        self.erode_size = 3
        self.dilate_size = 16
        self.std_bin_inner_size = 200
        self.std_bin_outer_size = 300
        self.erode_kernel = np.ones((self.erode_size, self.erode_size), np.uint8)
        self.dilate_kernel = np.ones((self.dilate_size, self.dilate_size), np.uint8)


    def set_erode_kernel(self,size):
        self.erode_size = size
        self.erode_kernel = np.ones((self.erode_size, self.erode_size), np.uint8)

    def set_dilate_kernel(self,size):
        self.dilate_size = size
        self.dilate_kernel = np.ones((self.dilate_size, self.dilate_size), np.uint8)

    def prepare_diff(self,image):
        prev_idx = self.current_idx
        self.current_idx += 1
        self.current_idx %= 2
        print("FUCKK")
        cv.absdiff(image, self.prev_image, self.diff_arr[self.current_idx])
        cv.copyTo(image, None, self.prev_image)
        i_d_f = cv.subtract(self.diff_arr[self.current_idx], self.diff_arr[prev_idx])
        standart_proc(i_d_f)
        return (i_d_f)

    def prepare_binarized(self,image):
        diff = self.prepare_raw_diff(image)
        if self.binarization_mode == 0:
            standart_proc(diff,self.simple_bin_thresh,self.erode_size,self.dilate_size)
        elif self.binarization_mode ==1:
            # cv.imshow('w', cv.resize(diff,(1000,1000)))
            # cv.waitKey()
            binarize_with_std(diff,self.std_bin_inner_size,self.std_bin_outer_size,self.std_bin_thresh,diff )
            # bin = binarize_with_std(diff)

            '''Эрозия'''
            cv.erode(diff , self.erode_kernel, diff , iterations=1)
            '''Дилатация'''
            cv.dilate(diff , self.dilate_kernel, diff , iterations=2)
            cv.imshow('dd', cv.resize(diff , (1000, 1000)))
        return diff


    def prepare_raw_diff(self,image):
        prev_idx = self.current_idx
        self.current_idx += 1
        self.current_idx %= 2

        cv.absdiff(image, self.prev_image, self.diff_arr[self.current_idx])
        cv.copyTo(image, None, self.prev_image)
        i_d_f = cv.subtract(self.diff_arr[self.current_idx], self.diff_arr[prev_idx])
        return i_d_f

    def get_contours_from_binarized(self,binarized):
        contours, _ = cv.findContours(binarized, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cvt_contour_boxes = []
        for c in contours:
            cvt_contour_boxes.append(cv.boundingRect(c))
        return cvt_contour_boxes

    def process_step_simple(self,image):
        i_d_f = self.prepare_binarized(image)
        return self.get_contours_from_binarized(i_d_f)


    def process_step_directed(self,image,direction_params):
        check = True
        for idx,param in enumerate(self.prev_direction_params):
            check&= (param == direction_params[idx])
        if check:
            self.stable_count+=1
        else:
            self.stable_count = 0
        self.prev_direction_params = direction_params
        i_d_f = self.prepare_binarized(image)
        cvt_contour_boxes = []
        valid_flag = False
        if self.stable_count>=2:
            valid_flag = True
            self.stable_count = 3
            cvt_contour_boxes = self.get_contours_from_binarized(i_d_f)
        # print(f'count {self.stable_count}, len {len(cvt_contour_boxes)}')
        return valid_flag,cvt_contour_boxes










        # image = copy.deepcopy(image)
        # prev_image = copy.deepcopy(image)
        # diff1 = copy.deepcopy(image * 0)
        # diff2 = copy.deepcopy(diff1)
        # diff_arr = [diff1, diff2]
        # mask = np.zeros((4504, 4504), dtype=np.uint8)
        # mask[1652:2852, 1652:2852] = 1



# current_idx = count % 2
# prev_idx = (count + 1) % 2
# cv.absdiff(image, prev_image, diff_arr[current_idx])
# # cv.pow(diff_arr[current_idx],2,dst=diff_arr[current_idx])
# # cv.imshow('current_diff', diff_arr[current_idx])
# prev_image = copy.deepcopy(image)
# I_d_f = cv.subtract(diff_arr[current_idx], diff_arr[prev_idx])
# # cv.multiply(I_d_f, mask, dst=I_d_f)
# # mask = np.zeros((4504,4504))
# # mask[1652: 2852, 1652: 2852] = 1
# # I_d_f = I_d_f&mask
#
#
# if done & changed:
#     step_c = 0
#
#     print('OPU MOVED')
# elif done & (not changed):
#     step_c += 1
#     standart_proc(I_d_f)
#
# if (step_c > 2) & (np.sum(I_d_f) < 4000000):
#     count += 1
#
#     print(f'step_c: {step_c}, {np.sum(I_d_f)}')
#     cv.imshow('idf', cv.resize(I_d_f * 200, (1200, 1200)))
#     t_middle = time.time()
#
#     timer.start()
#     contours, _ = cv.findContours(I_d_f, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

if __name__ == '__main__':
    finder = Contour_finder(4504,4504)
    meta = glib.Image_meta()
    image = np.zeros((4504,4504),np.uint8)
    valid,cs = finder.process_step_directed(image,[meta.az,meta.el])

    # idf = finder.process_step(image)
    # print(np.shape(idf))