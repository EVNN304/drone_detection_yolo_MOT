import math
import os.path

import numpy as np
import cv2 as cv
import copy

class Obj_annot():
    '''Класс для хранения информиции о местоположении объекта'''
    def __init__(self,class_id,bbox):
        self.class_id = class_id
        self.bbox = bbox
        self.size = (self.bbox[2]-self.bbox[0],self.bbox[3]-self.bbox[1])
        self.center = ((self.bbox[0]+self.bbox[2])/2,(self.bbox[1]+self.bbox[3])/2)

    def valid_check(self,frame_width,frame_height):
        stat = True
        if (self.bbox[0]<0) or (self.bbox[2]>=frame_width) or (self.bbox[1]<0) or (self.bbox[3]>=frame_height):
            stat = False
        return stat
    def get_scaled_bbox(self,sc_x,sc_y):
        return [self.bbox[0]*sc_x,self.bbox[1]*sc_y,self.bbox[2]*sc_x,self.bbox[3]*sc_y ]
    def get_scaled_size(self,sc_x,sc_y):
        return (self.size[0]*sc_x,self.size[1]*sc_y)
    def get_scaled_center(self,sc_x,sc_y):
        return (self.center[0] * sc_x, self.center[1] * sc_y)

    def print(self):
        print(f'Class: {self.class_id}')
        print(f'Box: {self.bbox}')
        print(f'Size: {self.size}')

class Object_f_detector():
    ORB = 0
    SURF = 1
    '''
    Класс для поиска опорного объекта на кадре, основанного на анализе особых точек
    '''
    def __init__(self,mode = ORB):
        self.mode = mode
        self.template = np.array(0)
        if self.mode == self.ORB:
            self.detector = cv.ORB_create(WTA_K=4)
            print('Сконфигурирован детектор на ORB-дискрипторах')
            # create BFMatcher object
            self.matcher =cv.BFMatcher(cv.NORM_HAMMING2, crossCheck=True)

        elif self.mode == self.SURF:
            self.detector = cv.xfeatures2d.SURF_create(400)
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            self.matcher = cv.FlannBasedMatcher(index_params, search_params)
            print('Сконфигурирован детектор на SURF-дискрипторах')
            pass
        else:
            print('Заданный режим не распознан')
        self.img_search = None
        self.matches = None
        self.matches_h_mask = None
        self.kp_template = None
        self.des_template = None
        self.kp_search = None
        self.des_search = None
        self.good = []
        self.orb_distance = 50


    def find_orb_matches(self,img_search,update_img = True):
        # find the keypoints and descriptors with ORB
        if update_img:
            self.img_search = img_search
        self.kp_search, self.des_search = self.detector.detectAndCompute(img_search, None)
        # Match descriptors.
        self.matches = self.matcher.match(self.des_template, self.des_search)
        # Sort them in the order of their distance.
        self.matches = sorted(self.matches, key=lambda x: x.distance)
        self.good = []
        for match in self.matches:
            # print(
            #     f'match: imgIdx: {match.imgIdx},  queryIdx: {match.queryIdx}, trainIdx: {match.trainIdx}, distance: {match.distance}')
            if match.distance < self.orb_distance:
                self.good.append(match)
        return self.matches

    def find_surf_matches(self,img_search,update_img = True):
        # # find the keypoints and descriptors with SURF
        if update_img:
            self.img_search = img_search
        self.kp_search, self.des_search = self.detector.detectAndCompute(img_search, None)

        self.matches = self.matcher.knnMatch(self.des_template, self.des_search, k=2)

        # store all the good matches as per Lowe's ratio test.
        self.good = []
        for m, n in self.matches:
            if m.distance < 0.7 * n.distance:
                self.good.append(m)
                if len(self.good) >= 100:
                    break
        return self.matches

    def find_matches(self,img_search,update_img = True):
        if self.mode == self.ORB:
            return self.find_orb_matches(img_search,update_img)
        elif self.mode == self.SURF:
            return self.find_surf_matches(img_search,update_img)

    def find_homography_mask(self):
        src_pts = np.float32([self.kp_template[m.queryIdx].pt for m in self.good]).reshape(-1, 1, 2)
        # print(src_pts[0])
        dst_pts = np.float32([self.kp_search[m.trainIdx].pt for m in self.good]).reshape(-1, 1, 2)
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
        self.matches_h_mask = mask.ravel().tolist()

    def show_matches(self,timeout = None):
        draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                           singlePointColor=None,
                           matchesMask=self.matches_h_mask,  # draw only inliers
                           flags=2)

        img_hom = cv.drawMatches(self.template, self.kp_template, self.img_search, self.kp_search, self.good, None, **draw_params)
        cv.imshow(f'homography, mode: {self.mode}', cv.resize(img_hom,None,fx = 1,fy = 1))
        cv.waitKey(timeout)

    def find_keypoints(self,img):
        if self.mode == self.ORB:
            return self.detector.detectAndCompute(img, None)
        elif self.mode == self.SURF:
            return self.detector.detectAndCompute(img, None)

    def set_template(self,template_img:np.array):
        self.template = template_img
        self.kp_template, self.des_template = self.find_keypoints(template_img)

    def show_template(self):
        cv.imshow('detector_template', self.template)
        cv.waitKey()

    def get_h_matched_pos(self):
        matched_template_pos = []
        matched_search_pos = []
        for i,flag in enumerate(self.matches_h_mask):
            if flag:
                m = self.good[i]
                matched_template_pos.append(self.kp_template[m.queryIdx].pt)
                matched_search_pos.append(self.kp_search[m.trainIdx].pt)
        return matched_template_pos,matched_search_pos

def find_area_center(points:[]):
    if len(points) !=0:
        x_sum = 0
        y_sum = 0
        for p in points:
            x_sum+=p[0]
            y_sum+=p[1]
        return (x_sum/len(points),y_sum/len(points))

def find_average_distance(points:[]):
    if len(points) != 0:
        x_sum = 0
        y_sum = 0
        dist_sum = 0
        count = 0
        for i in range(len(points)):
            for j in range(i+1,len(points)):
                # print(f'{i}->{j}')
                count+=1
                dx = abs(points[i][0]-points[j][0])
                x_sum+=dx
                dy = abs(points[i][1] - points[j][1])
                y_sum += dy
                dist_sum+=(math.sqrt(dx**2+dy**2))

        x_average = x_sum/count
        y_average = y_sum/count
        dist_average = dist_sum/count
        print(f'x_average {x_average} , y_average {y_average}, dist_average = {dist_average}')

if __name__ == '__main__':
    path1 = r'0_2.jpg'
    first_template = cv.imread(path1, cv.IMREAD_GRAYSCALE)  # queryImage

    detector_orb = Object_f_detector(Object_f_detector.ORB)
    detector_orb.set_template(first_template)
    detector_orb.show_template()

    path2 = r'C:\User_files\projects\cv_test\images\copters\cr1\2cam1.jpg'
    img_search = cv.imread(path2, cv.IMREAD_GRAYSCALE)[1500:3000,1500:3000]
    detector_orb.img_search = img_search
    detector_orb.find_orb_matches(img_search)
    detector_orb.find_homography_mask()
    detector_orb.show_matches()

    template_points, search_points = detector_orb.get_h_matched_pos()
    search_center = find_area_center(search_points)
    print('search')
    find_average_distance(search_points)
    print('template')
    find_average_distance(template_points)

    img_search_color = cv.cvtColor(img_search, cv.COLOR_GRAY2BGR)

    cv.circle(img_search_color,(int(search_center[0]),int(search_center[1])),10,(0,255,0),2)
    cv.imshow('search',img_search_color)
    cv.waitKey()

    # detector_surf = Object_f_detector(Object_f_detector.SURF)
    # detector_surf.set_template(first_template)
    # detector_surf.img_search = img_search
    # detector_surf.find_surf_matches(img_search)
    # detector_surf.find_homography_mask()
    # detector_surf.show_matches()
    # template_pos, search_pos = detector_surf.get_h_matched_pos()
'''
    folder_path = r'images\copters\cr1'
    img_paths = []
    for i in range(21):
        img_paths.append(os.path.join(folder_path,f'{i+2}cam1.jpg'))

    for p in img_paths:
        img_search = cv.imread(p, cv.IMREAD_GRAYSCALE)[1500:3000, 1500:3000]
        detector_orb.img_search = img_search
        detector_orb.find_orb_matches(img_search)
        detector_orb.find_homography_mask()
        detector_orb.show_matches(100)
'''
    # for p in img_paths:
    #     print(p)
    #     img_search = cv.imread(p, cv.IMREAD_GRAYSCALE)
    #     cv.imshow('test',cv.resize(img_search,(1000,1000)))
    #     cv.waitKey()
    #     detector_surf.find_surf_matches(img_search)
    #     detector_surf.find_homography_mask()
    #     detector_surf.show_matches()



    # img_search_color = cv.cvtColor(img_search,cv.COLOR_GRAY2BGR)
    # for p in template_pos:
    #     cv.circle(first_template,(int(p[0]),int(p[1])),4,0,2)
    # for p in search_pos:
    #     cv.circle(img_search_color, (int(p[0]), int(p[1])), 10, (0,255,0), 2)
    # cv.imshow('template',first_template)
    # cv.imshow('search', cv.resize(img_search_color,(1200,1200)))
    # cv.waitKey()


