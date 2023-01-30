import numpy as np
import cv2 as cv
import json
import math
import copy
from ui_utils import Config

def select_points_src(event,x,y,flags,param):
    global src_x, src_y, drawing , src_copy , new_src,select_points_srcn
    new_src = cv.line(src_copy.copy(), (x-15,y),(x+15,y) , (0,0,255), 1)
    new_src = cv.line(new_src, (x,y-15),(x,y+15) , (0,0,255), 1)
    if event == cv.EVENT_MBUTTONDOWN:
        drawing = True
        select_points_srcn = True
        # print("selecting_src")
        src_x, src_y = x,y
        cv.circle(src_copy,(x,y),10,(0,0,255),-1)
    elif event == cv.EVENT_MBUTTONUP:
        drawing = False

# mouse callback function
def select_points_dst(event,x,y,flags,param):
    global dst_x, dst_y, drawing ,dst_copy ,new_dst,select_points_dstn
    new_dst = cv.line(dst_copy.copy(), (x-5,y),(x+5,y) , (0,0,255), 1)
    new_dst = cv.line(new_dst, (x,y-5),(x,y+5) , (0,0,255), 1)
    cv.imshow("dst", new_dst)
    if event == cv.EVENT_MBUTTONDOWN:
        drawing = True
        select_points_dstn = True
        # print("selecting_dst")
        dst_x, dst_y = x,y
        cv.circle(dst_copy,(x,y),10,(0,0,255),-1)
    elif event == cv.EVENT_MBUTTONUP:
        drawing = False


def calculate_homography(src_point, cam_matrix):
    # homo_start_time = time.time()
    
    x_current = src_point[0]
    y_current = src_point[1]
    points_map = np.array([[x_current, y_current]], dtype='float32')
    points_map = np.array([points_map])
    return cv.perspectiveTransform(points_map, cam_matrix)


src_x, src_y = None,None

with open("../Settings/cam_params.json", 'r') as json_file:
    camera_params = json.load(json_file)
cam_matrix = np.array(camera_params['CAM'])
src = cv.imread('../Settings/src.jpg', -1)
src_copy = src.copy()
new_src = src_copy
cv.namedWindow("src",cv.WINDOW_NORMAL)
cv.moveWindow("src", 80,80)
cv.setMouseCallback('src', select_points_src)
src_w,src_h ,src_c= src.shape
src_d = np.sqrt(np.square(src_w)+np.square(src_h))
dst = cv.imread('../Settings/dst.jpg', -1)
dst_copy = dst.copy()
new_dst = dst_copy
cv.namedWindow("dst",cv.WINDOW_NORMAL)
cv.moveWindow("dst", 780,80)
cv.setMouseCallback('dst', select_points_dst)



while True:
    cv.imshow('src',src_copy)
    cv.imshow("src", new_src)
    new_src = src_copy
    cv.imshow('dst',dst_copy)
    cv.imshow("dst", new_dst)
    new_dst = dst_copy


    k = cv.waitKey(1) & 0xFF

    if k == ord("q"): 
        if src_x and src_y != None:
            cv.circle(src_copy,(src_x,src_y),10,(255,0,0),-1)
            # print(sort_config.sort_config.lens_dist)
            homo_pt = calculate_homography([src_x,src_y], cam_matrix=cam_matrix)

            homo_pt = (int(homo_pt[0][0][0]),
                        int(homo_pt[0][0][1]))
            
            cv.circle(dst_copy,(homo_pt[0],homo_pt[1]),10,(255,0,0),-1)