import json

import cv2
import numpy as np
import pandas as pd
from shapely.geometry.polygon import Polygon


class Config():
    # def __init__(self):
    #     self.activate_crop = ""

    def ui_config(self):
        with open("Settings/config.json", 'r') as _file:
            _data = json.load(_file)
            # 1 for cropping else no cropping
        self.activate_crop = _data['activate_crop']
        
        df_f = pd.read_csv("Team/fielder.csv",header = None,names=["fielder"])
        df_f["fielder"] = df_f["fielder"].str.upper()
        self.fielder_list = list(df_f["fielder"])
        
        df_f = pd.read_csv("Team/fielder_1.csv",header = None,names=["fielder"])    
        df_f["fielder"] = df_f["fielder"].str.upper()
        self.fielder_list_1 = list(df_f["fielder"])
        
        df_f = pd.read_csv("Team/fielder_position.csv",header = None,names=["fielder"])    
        df_f["fielder"] = df_f["fielder"].str.upper()
        self.fielder_position = list(df_f["fielder"])

    def process_config(self):
        # with open('Settings/crop_coordinates_ocr.json', 'r') as f:
        #     self.ocr_crop_data = json.load(f)
        # with open("Settings/crop_coordinates_ump1.json", 'r') as _file:
        #     _data = json.load(_file)
        # self.crop_u1_x1 = _data['x1']
        # self.crop_u1_x2 = _data['x2']
        # self.crop_u1_y1 = _data['y1']
        # self.crop_u1_y2 = _data['y2']

        with open("Settings/config.json", 'r') as _file:
            _data = json.load(_file)
            self.config_data = _data
            self.db_name = _data["db_name"]
            self.camera_model = _data['camera_model']  # 1 from baumer
            
            # 1 for cropping else no cropping
            self.activate_crop = _data['activate_crop']
            # "detection_weights/best_mar3.pt"
            self.weights = _data['detection_weight']
            # for  camera set it to 1 and to read it from a file "/path"
            self.source = _data["video_source"]
            
            self.dk_vno = _data["decklink_port_no"]
            self.dk_12g_mode = _data["decklink_12g_mode"]
            self.viz_udp_ip_address = _data["viz_udp_ip_address"]
            self.viz_udp_port = _data["viz_udp_port"]
            self.middleman_ip_address_port = _data["middleman_ip_address_port"]

            self.viz_tcp_ip_address = _data["viz_tcp_ip_address"]
            self.viz_tcp_port = _data["viz_tcp_port"]
            self.buggy_ip_address_port = _data["buggy_ip_address_port"]

            self.unreal_ip_address = _data["unreal_ip_address"]
            self.unreal_udp_port = _data["unreal_udp_port"]
            
            self.m_for_pix = _data["m_for_pix"]
            stump = _data["stump"]
            crease = _data["crease"]
            self.near_end_stump = stump["near_end"]
            self.far_end_stump = stump["far_end"]
            self.near_end_crease = crease["near_end"]
            self.far_end_crease = crease["far_end"]
            self.lens_distortion_flag = _data["lens_distortion"]
            self.score_file_path = _data["score_file_path"]
            self.innings = _data["innings"]
            self.score_file_mode = _data["score_file_mode"]
            self.crop_x1 = _data['crop_x1']
            self.crop_x2 = _data['crop_x2']
            self.crop_y1 = _data['crop_y1']
            self.crop_y2 = _data['crop_y2']
            self.collision_mode = _data["collision_mode"]
            self.print_udp_command = _data["print_udp_command"]
            self.ue4_print_udp_command = _data["ue4_print_udp_command"]
            self.middleman_video_stream_ip = _data["middleman_video_stream_ip"]
            self.middleman_video_stream_port = _data["middleman_video_stream_port"]
            if self.lens_distortion_flag == 1:
                mtx1 = [[1760.5563, 0, 2104.0234], [
                    0, 1772.3534, 1059.7439], [0, 0, 1]]
                dist1 = [-0.2906, 0.1071, 0.0011, 0.0008, -0.0206]
                self.lens_mtx = np.array(mtx1)
                self.lens_dist = np.array(dist1)
                self.w, self.h = 3840, 2160
                self.newcameramtx, self.lens_roi = cv2.getOptimalNewCameraMatrix(
                    self.lens_mtx, self.lens_dist, (self.w, self.h), 0, (self.w, self.h))

        # with open("Settings/crop_coordinates.json", 'r') as _file:
        #     _data = json.load(_file)
        # self.crop_x1 = _data['x1']
        # self.crop_x2 = _data['x2']
        # self.crop_y1 = _data['y1']
        # self.crop_y2 = _data['y2']
        with open("Settings/far_end_right_handed.json", "r") as infile:
            self.far_end_right_handed = json.load(infile)    
        with open("Settings/far_end_left_handed.json", "r") as infile:
            self.far_end_left_handed = json.load(infile)
        with open("Settings/near_end_right_handed.json", "r") as infile:
            self.near_end_right_handed = json.load(infile)
        with open("Settings/near_end_left_handed.json", "r") as infile:
            self.near_end_left_handed = json.load(infile)

        with open("Settings/cam_params.json", 'r') as json_file:
            self.camera_params = json.load(json_file)
        self.cam_matrix = np.array(self.camera_params['CAM'])

        with open('Settings/inner_circle.json', 'r') as f:
            _data = json.load(f)
        in_src = _data["inner_circle"]
        self.in_polygon = Polygon(in_src)

        self.df2 = pd.read_csv('Settings/batsmen_data.csv', header=None)
        for i in range(len(self.df2[0])):
            self.df2[0][i] = self.df2[0][i].lower()

        self.batsman_data = self.df2
        try:
            self.seg_mask = cv2.imread("Settings/segmentation_mask.jpg")
            self.seg_mask = cv2.cvtColor(self.seg_mask, cv2.COLOR_BGR2GRAY)

            self.seg_mask = np.stack((self.seg_mask,)*3, axis=-1)
        except Exception as e:
            self.seg_mask = None

    def sort_config(self):
        with open("Settings/cam_params.json", 'r') as json_file:
            self.camera_params = json.load(json_file)
        self.cam_matrix = np.array(self.camera_params['CAM'])

        with open("Settings/config.json", 'r') as file_:
            _data = json.load(file_)
        self.lens_distortion_flag = _data["lens_distortion"]

        with open("Settings/crop_coordinates_ump1.json", 'r') as _file:
            _data = json.load(_file)
        self.crop_u1_x1 = _data['x1']
        self.crop_u1_x2 = _data['x2']
        self.crop_u1_y1 = _data['y1']
        self.crop_u1_y2 = _data['y2']

        with open("Settings/crop_coordinates_ump2.json", 'r') as _file:
            _data = json.load(_file)
        self.crop_u2_x1 = _data['x1']
        self.crop_u2_x2 = _data['x2']
        self.crop_u2_y1 = _data['y1']
        self.crop_u2_y2 = _data['y2']

        with open("Settings/hyperparms.json", 'r') as _file:
            _data = json.load(_file)
        self.reassign_dist_thresh = _data['reassign']
        self.direction_threshold = _data['mv_distance']
        self.mv_threshold = _data['mv_frameskip']
        
        if self.lens_distortion_flag == 1:
            mtx1 = [[1760.5563, 0, 2104.0234], [
                0, 1772.3534, 1059.7439], [0, 0, 1]]
            dist1 = [-0.2906, 0.1071, 0.0011, 0.0008, -0.0206]
            self.lens_mtx = np.array(mtx1)
            self.lens_dist = np.array(dist1)
            self.w, self.h = 3840, 2160
            self.newcameramtx, self.lens_roi = cv2.getOptimalNewCameraMatrix(
                self.lens_mtx, self.lens_dist, (self.w, self.h), 0, (self.w, self.h))