import copy
from pickle import NONE
import queue
import socket
import sys
import threading
import time
from datetime import datetime as dt
from threading import Lock
from skimage.color import rgb2lab, deltaE_cie76
from scipy.spatial import distance as dist
from skimage import io
from sklearn.cluster import KMeans
from PIL import Image
import torch

import cv2
import numpy as np
import pandas as pd
import zmq
from imutils.video import FileVideoStream as Fvs
from pymongo import MongoClient
from PyQt5.QtCore import QThread, pyqtSignal
from Utils.downstreamer import RTPVideoStream as Rvs
from Track.detection import detect as detector
from Track.sort import Sort
from Utils.detect_utils import *

from Utils.ui_utils import Config


def reset_calib_handler(self):
    mutex.acquire()
    process_config_new = Config()
    process_config_new.process_config()
    mutex.release()
    self.process_config = process_config_new


def reset_handler(self):
    if not self.livelock_status:
        mutex.acquire()
        for i, bbox in enumerate(self.bbox_xyxy):
            try:
                x1, y1, x2, y2 = [int(i) for i in bbox]
                id = int(self.identities[i]
                         ) if self.identities is not None else 0
            except Exception as e:
                print("Exception reset:", e)
                continue
            if self.player_types[i] == 0 or self.player_types[i] == 9:
                print("regain", id, self.player_types[i])
                self.regain_active[id] = [x1, y1, x2, y2, self.player_types[i]]
        self.sort_tracker = Sort(max_age=600,
                                 min_hits=3,
                                 iou_threshold=0.3)  # {plug into parser}

        self.reset_FN()
        self.reset_FN_highlights()
        self.regain_flag = True
        self.homo_track = {}
        self.current_coords = {}
        self.dropped_frames = 0
        self.reset_naming()
        self.reset_umpire()
        self.bowler_id = -1
        self.wk_id = -1
        mutex.release()


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray, dict)
    show_field_plot_signal = pyqtSignal(np.ndarray)
    update_player_data_combo_box_signal = pyqtSignal(list)

    def __init__(self):
        super().__init__()
        ### Variables ###
        self.mutex1 = Lock()
        self.qin = queue.Queue(maxsize=5)
        self.qout_frame = queue.Queue(maxsize=5)
        self.qout_qtman_data = queue.Queue(maxsize=5)
        self.qout_fieldplt = queue.Queue(maxsize=5)
        self.qin_tracker = queue.Queue(maxsize=5)
        self.serverSock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.fn_threshold = 100
        self.n_init = 25
        self.dist_thresh = 50
        self.over_ball_dict = {}
        self.stop_threads = False
        self.udp_string = ""
        self.ue4_string = ""
        self.ue4_fielder_string = ""
        self.axis_offset = []
        self.vid = "Video here"
        self.qtm_socket_middleman = "Socket here"
        self.qtm_socket_buggy = "Socket here"
        self.dropped_frames = 0
        self.stop_stream = False
        self.stop_output = False
        self.left_handed = False
        self.stop_tracker = False
        self.save_frame_flag = False
        self.FRAME_HT = None
        self.FRAME_WD = None
        self.frm_count = -1
        self.homo_track = {}
        self.insertDb_status = False
        self.dbName_status = False
        self.dynamicDb_name = ""
        self.dynamicLivelock_dbName = ""
        self.dynamicScoreData_dbName = ""
        self.dynamicHistoryData_dbName = ""
        self.livelock_ids_saved = False
        self.livelock_status = False
        self.livelock_ids = {}
        self.livelock_ids_extra = {}
        self.air_init = True
        self.air_status = False
        self.flip_field_plot = 0
        self.player_connect_l__dict = {}
        self.left_right_automated = False
        self.highlights_list = []
        self.gap_ids = []
        self.multi_gap_ids = []
        self.ingap_ids = []
        self.batsmen_ids = []
        self.highlight_fns = []
        self.false_negatives = {}
        self.false_batsmen = {}
        self.false_negatives_slipFielders = {}
        self.false_negatives_mark_point_p = {}
        self.false_negatives_mark_point_o = {}
        self.false_negatives_outside_frame_z = {}
        self.false_negatives_outside_frame_a = {}
        self.false_negatives_outside_frame_u = {}
        self.distance_list = []
        self.current_coords = {}
        self.activate_distance_id = -1
        self.FN_HT = 70
        self.FN_WD = 50
        self.fn_count = 0
        self.max_id = 0
        self.fielder_count = 0
        self.others_count = 0
        self.capture_frames = False
        self.batsman_no = 0
        self.outside_circle_players = 0
        self.outside_circle = False
        self.umpire_id = []
        self.regain_active = {}
        self.bbox_xyxy = []
        self.identities = []
        self.player_types = []
        self.regain_flag = False
        self.score_file_data = {}
        self.old_score_line = ""
        self.score_line = ""
        self.over_wide = 0
        self.old_over = 1
        self.old_ball = 1
        self.old_score_line = ""
        self.old_score_line2 = ""
        self.last_captured_frame = 0
        self.wk_pt = []
        self.process_config = Config()
        self.process_config.process_config()
        self.collection = ""
        self.collection_livelock = ""
        self.score_file_collection = ""
        self.fielder_dict = {}
        self.fielder_dict_PO = {}
        self.detect_fielder_id_TA = -1
        self.detect_fielder_id_TB = -1
        self.detect_fielder_id_PO = -1
        self.bowler_id = -1
        self.wk_id = -1
        self.p_fielder = ""
        self.dummy_connect_id = -1
        self.send_config_flag = False
        self.reset_handler_flag = False
        self.reset_Highlight_flag = False
        self.reset_bowler_flag = False
        self.reset_naming_flag = False
        self.reset_umpire_flag = False
        self.reset_FN_flag = False
        self.past_score_modiefied_time = 0
        self.current_score_modiefied_time = 0
        self.dummy_player_id = -1
        self.dummy_player_falg = False
        self.middleman_video_stream_flag = False
        self.flying_id = 0
        self.flying_id_log = []
        self.fastid_frno = 0
        self.fastest_id = 0
        self.highest_speed = 0
        self.send_player_speed_id = -1
        self.send_player_speed_val = []
        self.send_player_speed_flag = False
        self.buggy_init = True
        self.speed_new_id = False
        self.send_player_speed_flag_downer = False
        self.ump_tuple = {}
        self.bastmen_tuple = {}
        self.main_ump_tuple = {}
        self.batsmen_ids_automated = []
        self.last_livelock_status = 0
        self.reswap_ids =[]
        self.last_frame_locations_homograph = {}
        self.slop_dict ={}
        self.lost_ids = {}
        self.default_red_ids =[]
        self.reswap_flag = 0
        self.record_flag = 0
        self.count_images = 0
        self.data = pd.DataFrame()
        self.ubnormal_speed_data = {}
        self.occ_frames = pd.DataFrame()
        self.m_of_stright_line = (self.process_config.far_end_stump[1]-self.process_config.near_end_stump[1])/(self.process_config.far_end_stump[0]-self.process_config.near_end_stump[0])
        # self.c_of_stright_line = self.process_config.far_end_stump[1] - self.m_of_stright_line*self.process_config.far_end_stump[0]
        self.copy_scorefile_data_flag = 0
        self.copy_scorefile_data_line_number =51
        if self.process_config.camera_model == 1:
            self.source = self.process_config.source
        if self.process_config.camera_model == 2:
            if self.process_config.dk_12g_mode == 1:
                self.source = 'decklinkvideosrc mode=0 connection=0 ! videoconvert ! appsink'
            else:
                self.source = f'decklinkvideosrc device-number={self.process_config.dk_vno} profile=5 mode=0 connection=0 ! videoconvert ! appsink'
            print(self.source)
        self.sort_tracker = Sort(max_age=600, min_hits=3, iou_threshold=0.3)

        try:
            self.conn = MongoClient()
            print("DB Connected successfully!!!")
        except:
            print("DB Could not connect to MongoDB")
        self.db = self.conn.quidich_db_v3
        if self.process_config.db_name + str("_livelock_data") in self.db.collection_names():
            print(error)

    ### Functions ###
    update_player_connect = update_player_connect
    update_lr_automation = update_lr_automation
    livelock_clicked = livelock_clicked
    insertDb_clicked = insertDb_clicked
    air_clicked = air_clicked
    batsmen_pos_flip = batsmen_pos_flip
    change_player_type = change_player_type
    highlight_player = highlight_player
    reset_Highlight = reset_Highlight
    highlight_fn = highlight_fn
    highlight_player_streak = highlight_player_streak
    reset_HighlightStreak = reset_HighlightStreak
    calculate_distance = calculate_distance
    remove_distance = remove_distance
    activate_distance = activate_distance
    set_batsmen_id = set_batsmen_id
    create_FN_player = create_FN_player
    remove_FN_player = remove_FN_player
    # remove_batsmen_player = remove_batsmen_player
    reset_FN = reset_FN
    reset_FN_highlights = reset_FN_highlights
    create_FN_player_slipFielders = create_FN_player_slipFielders
    create_FN_player_mark_point_p = create_FN_player_mark_point_p
    remove_FN_player_markpoint_p = remove_FN_player_markpoint_p
    create_FN_player_mark_point_o = create_FN_player_mark_point_o
    remove_FN_player_markpoint_o = remove_FN_player_markpoint_o
    remove_FN_player_slipFielders = remove_FN_player_slipFielders
    create_FN_player_mark_point_b = create_FN_player_mark_point_b
    create_FN_player_outside_frame_z = create_FN_player_outside_frame_z
    create_FN_player_outside_frame_a = create_FN_player_outside_frame_a
    create_FN_player_outside_frame_u = create_FN_player_outside_frame_u
    remove_FN_player_outside_frame_u = remove_FN_player_outside_frame_u

    remove_FN_player_outside_frame_a = remove_FN_player_outside_frame_a
    remove_FN_player_outside_frame_z = remove_FN_player_outside_frame_z

    each_slice = each_slice
    save_frame = save_frame
    reset_batsmen = reset_batsmen
    activate_count_outside_player = activate_count_outside_player
    set_gap_ids = set_gap_ids
    set_multi_gap_ids = set_multi_gap_ids
    set_ingap_ids = set_ingap_ids
    set_umpire_id = set_umpire_id
    reset_umpire = reset_umpire
    reset_naming = reset_naming
    find_clicked_player_id = find_clicked_player_id
    # find_clicked_batsman_player_id = find_clicked_batsman_player_id
    # create_batsmen_player = create_batsmen_player
    find_clicked_FN_player_id = find_clicked_FN_player_id
    find_clicked_FN_player_id_slipFielders = find_clicked_FN_player_id_slipFielders
    find_clicked_FN_player_id_markpoint_p = find_clicked_FN_player_id_markpoint_p
    find_clicked_FN_player_id_markpoint_o = find_clicked_FN_player_id_markpoint_o
    find_clicked_FN_player_id_outside_frame_a = find_clicked_FN_player_id_outside_frame_a
    find_clicked_FN_player_id_outside_frame_u = find_clicked_FN_player_id_outside_frame_u
    find_clicked_FN_player_id_outside_frame_z = find_clicked_FN_player_id_outside_frame_z

    reset_calib_handler = reset_calib_handler
    reset_handler = reset_handler
    get_mongoData = get_mongoData
    outside_circle_calc = outside_circle_calc
    set_clicked_fielder_TA = set_clicked_fielder_TA
    set_clicked_fielder_TB = set_clicked_fielder_TB
    set_clicked_fielder_PO = set_clicked_fielder_PO
    set_fielder_name_TA = set_fielder_name_TA
    set_fielder_name_TB = set_fielder_name_TB
    set_fielder_name_PO = set_fielder_name_PO
    clear_naming_coloring = clear_naming_coloring
    reset_bowler = reset_bowler
    command_f4 = command_f4
    command_f6 = command_f6
    command_f7 = command_f7
    command_f8 = command_f8
    command_f9 = command_f9
    command_f = command_f
    command_w = command_w
    send_tcp_message = send_tcp_message
    command_front_end_mode = command_front_end_mode

    reset_handler_flager = reset_handler_flager
    reset_FN_flager = reset_FN_flager
    reset_Highlight_flager = reset_Highlight_flager
    reset_bowler_flager = reset_bowler_flager
    reset_umpire_flager = reset_umpire_flager
    reset_naming_flager = reset_naming_flager

    get_wt_scorefile_data = get_wt_scorefile_data
    get_ae_scorefile_data = get_ae_scorefile_data
    save_db_colection_to_json = save_db_colection_to_json
    delay_send = delay_send

    def get_pct_color(img_rgb, rgb_color, threshold=10):
        img_lab = rgb2lab(img_rgb)
        rgb_color_3d = np.uint8(np.asarray([[rgb_color]]))
        rgb_color_lab = rgb2lab(rgb_color_3d)
        delta = deltaE_cie76(rgb_color_lab, img_lab)
        x_positions, y_positions = np.where(delta < threshold)
        nb_pixel = img_rgb.shape[0] * img_rgb.shape[1]
        pct_color = len(x_positions) / nb_pixel
        return pct_color

    def get_position(self,r,theta,r_other_end,theta_other_end,player_):
        pos = -1
        # print(r,theta,r_other_end,theta_other_end,player_)
        if r<2:
            if theta>20 and theta<160:
                if "WKT KEEPER" not in list(self.fielder_dict_PO.values()):
                    pos = "WKT KEEPER"
                    return pos
        
        if r<27:
            if theta >=80 and theta <= 100:
                if "WKT KEEPER" not in list(self.fielder_dict_PO.values()):
                    pos = "WKT KEEPER"
                    return pos

        if theta_other_end >240 and theta_other_end <300:
            if r_other_end>4 and r_other_end<10 :
                if "BOWLER" not in list(self.fielder_dict_PO.values()):
                    pos = "BOWLER"
                    return pos
        
        if theta_other_end >260 and theta_other_end <280:
            if "BOWLER" not in list(self.fielder_dict_PO.values()):
                pos = "BOWLER"
                return pos
        
        
      
        return pos

    def reswap(self,reswap_plyer_id):
        if len(self.reswap_ids) < 2:
            self.reswap_ids.append(reswap_plyer_id)
        if len(self.reswap_ids) == 2:
            if (((int(self.reswap_ids[0])-1) >= 0) and (int(self.reswap_ids[1])-1 >= 0)):
                self.reswap_flag = 1
                
    def copy_scorefile_data(self):
        self.copy_scorefile_data_flag = 1
        
    def check_for_loss_ids(self):
        for ll_ids in self.livelock_ids.keys():
            if int(ll_ids) not in list(map(int,self.identities)):
                if self.lost_ids.get(ll_ids) == None:
                    self.lost_ids[ll_ids] = [1,int(self.frm_count)]
                else:
                    self.lost_ids[ll_ids] = lost_ids[ll_ids]+1

        for ll_ids in self.livelock_ids.keys():
            # print(int(ll_ids) , list(map(int,self.identities)))
            # print(int(ll_ids) in list(map(int,self.identities)))
            # print(ll_ids)
            if int(ll_ids) in list(map(int,self.identities)):
                pass
            else:
                green_location = self.livelock_ids[str(ll_ids)]
                red_ids = list(set(list(np.where(np.array(self.player_types) == 3))[0])^set(self.default_red_ids))
                for id in red_ids:
                    # print(self.red_ids)
                    red_location = self.homo_track[int(id)]
                    if (green_location[0]-red_location[0])**2 + (green_location[1]-red_location[1])**2 < 300:
                        self.sort_tracker.reswap(int(ll_ids)-1,int(id)-1)
        
        for bastmen_ll_ids in self.bastmen_tuple.keys():
            if int(bastmen_ll_ids) in list(map(int,self.identities)):
                pass
            else:
                bat_location = self.bastmen_tuple[bastmen_ll_ids]
                red_ids = list(set(list(np.where(np.array(self.player_types) == 3))[0])^set(self.default_red_ids))
                for id in red_ids:
                    red_location = self.homo_track[int(id)]
                    if (bat_location[0]-red_location[0])**2 + (bat_location[1]-red_location[1])**2 < 300:
                        self.sort_tracker.reswap(int(bastmen_ll_ids)-1,int(id)-1)
        
        for ump_ll_ids in self.ump_tuple.keys():
            if int(ump_ll_ids) in list(map(int,self.identities)):
                pass
            else:
                ump_location = self.ump_tuple[ump_ll_ids]
                red_ids = list(set(list(np.where(np.array(self.player_types) == 3))[0])^set(self.default_red_ids))
                for id in red_ids:
                    red_location = self.homo_track[int(id)]
                    if (ump_location[0]-red_location[0])**2 + (ump_location[1]-red_location[1])**2 < 300:
                        self.sort_tracker.reswap(int(ump_ll_ids)-1,int(id)-1)
    
    def map_PO(self):
        for player_ in self.livelock_ids.keys():
            #print(player_,self.batsmen_ids,self.umpire_id,(player_ in self.batsmen_ids) , (player_ in self.umpire_id))
            
            P1_X,P1_Y,P2_X,P2_Y = get_relative_location(int(player_),self.flip_field_plot,self.left_handed,self.process_config.far_end_stump,
                                                            self.process_config.near_end_stump,self.homo_track)
            # print(player_,P1_X,P1_Y)
            r ,tan = get_polar_coordinates(P1_X,P1_Y)
            r = r*self.process_config.m_for_pix
            r_other_end ,tan_other_end = get_polar_coordinates(P2_X,P2_Y)
            r_other_end = r_other_end*self.process_config.m_for_pix
            if int(player_) in self.umpire_id:
                continue
            if int(player_) in self.batsmen_ids:
                continue 
            position_po = self.get_position(r,tan,r_other_end,tan_other_end,player_)
            if position_po != -1:
                self.fielder_dict_PO[player_] = position_po

    def assign_players(self):
        # print("HERERERERERER")
        for detected_id in self.identities:
            P1_X, P1_Y, P2_X, P2_Y = get_relative_location(int(detected_id), self.flip_field_plot, self.left_handed, self.process_config.far_end_stump,
                                                           self.process_config.near_end_stump, self.homo_track)
            # print(P1_X,P1_Y, P2_X, P2_Y)
            r_, tan_ = get_polar_coordinates(P1_X, P1_Y)
            r_ = r_*self.process_config.m_for_pix
            r_other_end, tan_other_end = get_polar_coordinates(P2_X, P2_Y)
            r_other_end = r_other_end*self.process_config.m_for_pix
            # print(self.flip_field_plot,self.left_handed)
            if self.flip_field_plot == 1 and self.left_handed == 0:
                angle_max_r = self.process_config.far_end_right_handed

            if self.flip_field_plot == 1 and self.left_handed == 1:
                angle_max_r = self.process_config.far_end_left_handed

            if self.flip_field_plot == 0 and self.left_handed == 0:
                # print('here')
                angle_max_r = self.process_config.near_end_right_handed

            if self.flip_field_plot == 0 and self.left_handed == 1:
                angle_max_r = self.process_config.near_end_left_handed

            max_r_index = np.argmin([abs(float(an)-float(tan_))
                                    for an in list(angle_max_r.keys())])
            max_r = angle_max_r[list(angle_max_r.keys())[max_r_index]][0]
            max_r = max_r*self.process_config.m_for_pix

            # print(r_,tan_,r_other_end, tan_other_end,detected_id)

            if r_ > max_r+max_r*0.05:
                continue

            # print(r_other_end,tan_other_end,detected_id,P2_X,P2_Y)
            if r_ < 3 and tan_ > 180:
                if len(self.batsmen_ids_automated) == 2:
                    continue
                self.sort_tracker.change_playertype(detected_id, 1)
                self.sort_tracker.batsmen_ids_automated.append(
                    int(detected_id))
                self.batsmen_ids_automated.append(int(detected_id))
                # self.fielder_dict_PO[str(int(detected_id))] = 'BATSMAN'
            elif (r_ > 20 and r_ < 35) and (tan_ > 165 and tan_ < 195):
                if len(self.umpire_id) == 2:
                    continue
                self.sort_tracker.change_playertype(detected_id, 2)
                self.sort_tracker.umpire_id.append(int(detected_id))
                self.umpire_id.append(int(detected_id))
                # self.fielder_dict_PO[str(int(detected_id))] = 'UMPIRE'
            elif r_other_end < 4 and (tan_other_end < 260 or tan_other_end > 310):
                if len(self.batsmen_ids_automated) == 2:
                    continue
                self.sort_tracker.change_playertype(detected_id, 1)
                self.sort_tracker.batsmen_ids_automated.append(
                    int(detected_id))
                self.batsmen_ids_automated.append(int(detected_id))
                # self.fielder_dict_PO[str(int(detected_id))] = 'BATSMAN'
            elif (r_other_end > 3 and r_other_end < 6) and ((tan_other_end > 260 and tan_other_end < 280) or (tan_ > 267 and tan_ < 273)):
                # print(len(self.ump_tuple))
                if len(self.main_ump_tuple) == 1:
                    continue
                self.sort_tracker.change_playertype(detected_id, 2)
                self.sort_tracker.umpire_id.append(int(detected_id))
                self.umpire_id.append(int(detected_id))
                self.main_ump_tuple = {int(detected_id):tuple(self.homo_track[detected_id])}
                # self.fielder_dict_PO[str(int(detected_id))] = 'UMPIRE'

            else:
                self.sort_tracker.change_playertype(detected_id, 0)
                # self.fielder_dict_PO[str(int(detected_id))] = -1

    def go_green(self):
        print("\n")
        
        if self.livelock_status ==0:
            for all_ids in self.identities:
                self.sort_tracker.change_playertype(all_ids, 3)
                self.fielder_dict_PO = {}
                self.sort_tracker.umpire_id = []
                self.umpire_id = []
                self.sort_tracker.batsmen_ids_automated = []
                self.batsmen_ids_automated = []
                self.ump_tuple = {}
                self.bastmen_tuple = {}
                self.main_ump_tuple = {}
                self.reswap_flag = 0
            
            self.assign_players()

    def make_all_red(self):
        for all_ids in self.identities:
            self.sort_tracker.change_playertype(all_ids, 3)
        self.fielder_dict_PO = {}
        self.sort_tracker.umpire_id = []
        self.umpire_id = []
        self.sort_tracker.batsmen_ids = []
        self.batsmen_ids = []
        self.batsmen_ids_automated = []
        self.sort_tracker.batsmen_ids_automated = []
        self.reswap_ids = []
        self.ump_tuple = {}
        self.bastmen_tuple = {}
        self.main_ump_tuple = {}
        self.highlight_fns = []
        self.false_negatives = {}
        self.false_batsmen = {}
        self.false_negatives_slipFielders = {}
        self.false_negatives_mark_point_p = {}
        self.false_negatives_mark_point_o = {}
        self.false_negatives_mark_point_o_copy = {}
        self.false_negatives_outside_frame_z = {}
        self.false_negatives_outside_frame_a = {}
        self.false_negatives_outside_frame_u = {}
        self.send_player_speed_id = -1
        self.flying_id = 0
        self.flying_id_log = []
        self.highlights_list = []
        self.gap_ids = []
        self.multi_gap_ids = []
        self.ingap_ids = []
        self.highlight_fns = []
        self.distance_list = []
        self.bowler_id = -1
        self.wk_id = -1
        self.reswap_flag = 0
        # self.wk_pt = []
    def record(self):
        self.record_flag = 1 - self.record_flag

    def unmark_player(self,unmark_id):
        if self.livelock_status:
            self.sort_tracker.change_playertype(unmark_id,3)
            if str(unmark_id) in self.livelock_ids:
                del self.livelock_ids[str(unmark_id)]
            # del self.livelock_ids[str(unmark_id)]
            if str(unmark_id-1) in self.sort_tracker.trackers.keys():
                del self.sort_tracker.trackers[str(unmark_id-1)]

    def collect_fielder_info(self,livelock_ids_copy):
        if self.last_livelock_status ==0:
            if self.livelock_status:
                self.fielder_dict_PO = {}
                self.livelock_frame_number = self.frm_count
                self.last_livelock_status = 1
                self.map_PO()
                try:
                    self.bastmen_tuple = {self.batsmen_ids_automated[0]:tuple(self.homo_track[self.batsmen_ids_automated[0]]),
                                            self.batsmen_ids_automated[1]:tuple(self.homo_track[self.batsmen_ids_automated[1]])}
                except:
                    pass
                
                try:
                    self.ump_tuple = {self.umpire_id[0]:tuple(self.homo_track[self.umpire_id[0]]),
                                            self.umpire_id[1]:tuple(self.homo_track[self.umpire_id[1]])}
                except:
                    pass

                for id in livelock_ids_copy.keys():
                    if int(id) in self.umpire_id:
                        print(id,"true")
                        continue 
                    if int(id) in self.batsmen_ids_automated:
                        print(id,"true")
                        continue   
                    try:
                        self.sort_tracker.tagging_info[id].append(self.fielder_dict_PO[id])
                    except:
                        pass
            for unknown_idx in list(np.where(np.array(self.player_types) == 3))[0]:
                self.default_red_ids.append(self.identities[unknown_idx])

    def unnormal_speed_detected(self):
        for ubnormal_id in list(self.ubnormal_speed_data.keys()):        
            ubnormal_id_loc = self.ubnormal_speed_data[ubnormal_id]
            red_idx = list(set(list(np.where(np.array(self.player_types) == 3))[0])) # ^set(self.default_red_ids))
            for r_idx in red_idx:
                r_id = self.identities[r_idx]
                red_location = self.homo_track[int(r_id)]
                print(math.sqrt((ubnormal_id_loc[0]-red_location[0])**2 + (ubnormal_id_loc[1]-red_location[1])**2),r_id,ubnormal_id)
                if math.sqrt((ubnormal_id_loc[0]-red_location[0])**2 + (ubnormal_id_loc[1]-red_location[1])**2) < 100:
                    self.sort_tracker.reswap(int(ubnormal_id)-1,int(r_id)-1)
                    del self.ubnormal_speed_data[ubnormal_id]
                    break
        
    def store_frames_read(self):
        # ret = True
        # count1 = 0
        while True:
            if self.process_config.camera_model == 1:
                ret, frame = self.vid.read()
                if not ret:
                    # frame = None
                    frame = np.zeros(
                        [self.FRAME_HT, self.FRAME_WD, 3], dtype=np.uint8)
                    frame.fill(255)
            elif self.process_config.camera_model == 2:
                frame = self.vid.read()

            if (frame is not None) and (self.FRAME_HT <= frame.shape[0]) and (self.FRAME_WD <= frame.shape[1]):
                # count1 += 1
                if self.process_config.activate_crop == 1:
                    frame = frame[
                        self.process_config.crop_y1:self.process_config.crop_y2,
                        self.process_config.crop_x1:self.process_config.crop_x2]
                if (self.FRAME_HT <= frame.shape[0]) and (self.FRAME_WD <= frame.shape[1]):
                    self.mutex1.acquire()
                    if self.qin.full() is True:
                        self.qin.get()
                        self.dropped_frames += 1
                    self.qin.put(frame)
                    if self.air_status and self.middleman_video_stream_flag:
                        self.middleman_video_stream.write(copy.deepcopy(frame))
                    # rtptransfer here
                    self.mutex1.release()
                self.FRAME_HT = frame.shape[0]
                self.FRAME_WD = frame.shape[1]
            if self.stop_stream:
                self.vid.stop()
                break
            # count1 += 1
        self.stop_stream = False

    def show_frames(self):
        udp_string = ""
        last_speed_value = -1
        while True:
            if self.stop_threads:
                break
            frame = self.qout_frame.get()
            data = self.qout_qtman_data.get()
            player_speed_data = -1
            data["config_data"] = -1
            data["cam_params"] = -1
            data["frame"] = -1
            if self.send_config_flag:
                data["config_data"] = self.process_config.config_data
                data["cam_params"] = self.process_config.camera_params
                data["frame"] = cv2.imread("Settings/frame.jpg")
                print("Sending Config Data to Middleman")
                self.send_config_flag = False

            if self.send_player_speed_flag:
                # player_speed_data[str(self.send_player_speed_id)] = self.send_player_speed_val
                # print("here")
                player_speed_data = self.delay_send(
                    self.send_player_speed_val, 0.52)

                # print("sending:", player_speed_data)
                if self.send_player_speed_val != [] and player_speed_data != None:
                    self.send_player_speed_val.pop(0)
                if player_speed_data != None:
                    last_speed_value = copy.deepcopy(player_speed_data)
                else:
                    player_speed_data = last_speed_value
                try:
                    print("player_data", player_speed_data)
                    self.qtm_socket_buggy.send_pyobj(player_speed_data)
                    # print("data sent!")
                except Exception as e:
                    self.buggy_init = True
                    # print("Unable to AIR", e)
            elif self.send_player_speed_flag_downer:
                self.send_player_speed_flag_downer = False
                try:
                    self.qtm_socket_buggy.send_pyobj(-1)
                    # print("data sent!")
                except Exception as e:
                    self.buggy_init = True
                    # print("Unable to AIR", e)

            if self.air_status and self.middleman_video_stream_flag:
                try:
                    self.qtm_socket_middleman.send_pyobj(data)
                except Exception as e:
                    self.air_init = True
                    print("Unable to AIR", e)
            udp_string, ue4_string = self.qout_fieldplt.get()
            if self.process_config.print_udp_command == 1:
                print(udp_string)
            if self.process_config.ue4_print_udp_command == 1:
                print(ue4_string)

            self.change_pixmap_signal.emit(frame, self.current_coords)

            if udp_string != "":
                try:
                    # print("here")
                    self.serverSock.sendto(udp_string.encode(
                        'utf-8'), (self.process_config.viz_udp_ip_address, self.process_config.viz_udp_port))
                except Exception as e:
                    # print(e)
                    pass

            if ue4_string != "":
                try:
                    self.serverSock.sendto(ue4_string.encode(
                        'utf-8'), (self.process_config.unreal_ip_address, self.process_config.unreal_udp_port))
                except Exception as e:
                    # print(e)
                    pass
            if self.stop_output:
                break
        self.stop_output = False

    def train_svm(self):
        self.sort_tracker.get_features()
        self.sort_tracker.train_on_featuers()

    def run_tracker(self):
        livelock_data = {}
        left_handed_cache = self.left_handed
        dist_dict = {}
        self.score_file_data = {}
        self.score_line = ""
        self.score_lines = ""
        scoring_flag = True
        self.livelock_ids_extra = {}
        self.axis_offset = [self.process_config.near_end_stump,
                            self.process_config.far_end_stump]

        starttime = time.time()
        old_bowler_pos = []
        bowler_dist = 0
        bowler_speed = 0
        old_homo_track = {}

        # Initiliaze SORT
        while True:
            # reset fns
            if self.reset_handler_flag:
                self.reset_handler()
                self.reset_handler_flag = False

            if self.reset_Highlight_flag:
                self.reset_Highlight()
                self.reset_Highlight_flag = False

            if self.reset_FN_flag:
                self.reset_FN()
                self.reset_FN_flag = False

            if self.reset_bowler_flag:
                self.reset_bowler()
                self.reset_bowler_flag = False

            if self.reset_umpire_flag:
                self.reset_umpire()
                self.reset_umpire_flag = False

            if self.reset_naming_flag:
                self.reset_naming()
                self.reset_naming_flag = False

            if self.buggy_init:
                try:
                    # print("Publishing... to QT-Buggy")
                    context = zmq.Context()
                    self.qtm_socket_buggy = context.socket(zmq.PUB)
                    self.qtm_socket_buggy.bind(
                        self.process_config.buggy_ip_address_port)
                    # print("Published to QT-Buggy")
                    self.buggy_init = False
                except Exception as e:
                    pass
                    # print(e)

            if self.air_status and self.air_init:
                try:
                    print("Publishing... to QT-Middleman")
                    context = zmq.Context()
                    self.qtm_socket_middleman = context.socket(zmq.PUB)
                    self.qtm_socket_middleman.bind(
                        self.process_config.middleman_ip_address_port)
                    print("Published to QT-Middleman")
                    self.air_init = False
                    self.middleman_video_stream = Rvs(
                        ip=self.process_config.middleman_video_stream_ip, port=self.process_config.middleman_video_stream_port).start()
                    self.middleman_video_stream_flag = True
                except Exception as e:
                    print("QT AIR FAILED", e)
            elif not self.air_status:
                try:
                    self.qtm_socket_middleman.close()
                    self.middleman_video_stream_flag = False

                except Exception as e:
                    # print("Unable to close socket",e)
                    pass
            if self.stop_tracker:
                self.stop_tracker = False
                break
            start_tracker_time = time.time()
            self.udp_string = ""
            self.ue4_string = ""
            self.ue4_fielder_string = ""
            livelock_data = {}
            history_data = {}
            dets_to_sort = np.empty((0, 6))
            # dets_to_sort = []
            im0s, bboxes = self.qin_tracker.get()
            save_im0s = im0s.copy()
            if bboxes is None:
                bboxes = []
            for box in bboxes:
                dets_to_sort = np.vstack((dets_to_sort, np.array(
                    [box[0], box[1], box[2], box[3], 0.95, 0])))
            tracked_dets, parameters, self.homo_track, push_data, downstream_data = self.sort_tracker.update(
                dets=dets_to_sort,
                frm_count=self.frm_count,
                FRAME_HT=self.FRAME_HT,
                FRAME_WD=self.FRAME_WD,
                homo_track=self.homo_track,
                livelock_ids=self.livelock_ids,
                fielder_dict_PO =self.fielder_dict_PO,
                img = im0s.copy())
            
            
            active_track = []
            highlight_track = []
            highlight_streaks_list = []
            current_coords_temp = {}
            self.others_count = 0
            self.fielder_count = 0
            self.fn_count = 0
            close_tracks = []
            proper_regain = False
            id_set = set()
            key_set = set()
            same_set = False
            outside_circle_set = set()

            push_data["frame_count"] = self.frm_count
            push_data["livelock"] = self.livelock_status

            downstream_data["frame_count"] = self.frm_count
            downstream_data["livelock"] = self.livelock_status

            if self.process_config.score_file_mode == "wt":
                self.score_file_data, self.score_line = self.get_wt_scorefile_data(
                    self.score_file_data, self.score_line,
                    self.process_config.score_file_path,
                    self.process_config.innings
                )
            elif self.process_config.score_file_mode == "ae":
                self.score_file_data, self.score_line,self.score_lines = self.get_ae_scorefile_data(
                    self.score_file_data, self.score_line,
                    self.process_config.score_file_path,
                    self.process_config.innings
                )

            # if not self.score_file_data:
            #     # print("score file haven't updated yet")
            #     pass
            if self.score_line != self.old_score_line and self.left_right_automated and self.score_file_data:
                print("NEW LINE")
                self.old_score_line = self.score_line
                print(f"\nbefore old over: {self.old_over}")

                self.left_handed, self.flip_field_plot, self.old_over, self.over_wide = process_lr_nf(
                    self.score_file_data,
                    self.old_over,
                    self.over_wide,
                    self.process_config.df2,
                    self.flip_field_plot)

                print(f"Flip: {self.flip_field_plot}")
                if self.left_handed:
                    print(f"L H : {self.left_handed} L")
                else:
                    print(f"L H : {self.left_handed} R")
                print(f"After old over: {self.old_over}")
                print(f"over_wide: {self.over_wide}")

            push_data["flip"] = self.flip_field_plot
            push_data["pos"] = self.left_handed

            downstream_data["flip"] = self.flip_field_plot
            downstream_data["pos"] = self.left_handed

            if self.regain_flag:
                active_track = []
            
            if self.copy_scorefile_data_flag ==1:
                self.copy_scorefile_data_flag = 0
                self.copy_scorefile_data_line_number = self.copy_scorefile_data_line_number + 1 
                with open('score_updates.txt', "a") as f:
                    f.write(self.score_lines[self.copy_scorefile_data_line_number])
                
            
            if len(tracked_dets) > 0:
                self.bbox_xyxy = tracked_dets[:, :4]
                self.identities = tracked_dets[:, 8]
                highlights = parameters[0]
                print(self.identities)
                print(parameters[1])
                self.player_types = parameters[1]
                
                
                directions = parameters[2]
                highlight_streaks = parameters[3]
                self.max_id = max(self.identities)
                regain_counter = 0
                for i, bbox in enumerate(self.bbox_xyxy):
                    x1, y1, x2, y2 = [int(i) for i in bbox]
                    id = int(
                        self.identities[i]) if self.identities is not None else 0
                    false_negatives_copy = copy.deepcopy(self.false_negatives)
                    for key in false_negatives_copy:
                        if (id in self.homo_track) and (self.player_types[i] == 0) and (key in self.homo_track):
                            centroid_box = (
                                int(self.homo_track[id][0]), int(self.homo_track[id][1]))
                            centroid_fn = (
                                int(self.homo_track[key][0]), int(self.homo_track[key][1]))
                            # if id == 11:
                            # print("false negatives", id, key,
                            #       centroid_box, centroid_fn)
                            if (abs(centroid_fn[0] - centroid_box[0]) < self.fn_threshold) and (abs(centroid_fn[1] - centroid_box[1]) < self.fn_threshold):
                                self.player_types[i] = 3
                                # changes the player categry to others
                                self.change_player_type(id, 3)

                    false_negatives_outside_frame_z_copy = copy.deepcopy(
                        self.false_negatives_outside_frame_z)
                    for key in false_negatives_outside_frame_z_copy:
                        if((id in self.homo_track) and (self.player_types[i] == 0) and (key in self.homo_track)):
                            centroid_box = (
                                int(self.homo_track[id][0]), int(self.homo_track[id][1]))
                            centroid_fn = (
                                int(self.homo_track[key][0]), int(self.homo_track[key][1]))
                            if ((abs(centroid_fn[0] - centroid_box[0]) < self.fn_threshold) and (abs(centroid_fn[1] - centroid_box[1]) < self.fn_threshold)):
                                self.player_types[i] = 3
                                # changes the player categry to others
                                self.change_player_type(id, 3)

                    false_negatives_outside_frame_a_copy = copy.deepcopy(
                        self.false_negatives_outside_frame_a)
                    for key in false_negatives_outside_frame_a_copy:
                        if((id in self.homo_track) and (self.player_types[i] == 0) and (key in self.homo_track)):
                            centroid_box = (
                                int(self.homo_track[id][0]), int(self.homo_track[id][1]))
                            centroid_fn = (
                                int(self.homo_track[key][0]), int(self.homo_track[key][1]))
                            if ((abs(centroid_fn[0] - centroid_box[0]) < self.fn_threshold) and (abs(centroid_fn[1] - centroid_box[1]) < self.fn_threshold)):
                                self.player_types[i] = 3
                                # changes the player categry to others
                                self.change_player_type(id, 3)

                    regain_active_copy = copy.deepcopy(self.regain_active)
                    if self.regain_flag and len(active_track) == regain_counter and same_set is False:
                        for key in regain_active_copy:
                            centroid_box = (int((x1+x2)/2), int((y1+y2)/2))
                            centroid_ra = int((regain_active_copy[key][0] + regain_active_copy[key][2])/2), int(
                                (regain_active_copy[key][1] + regain_active_copy[key][3]) / 2)
                            pixel_distance = math.hypot(
                                (centroid_box[0] - centroid_ra[0]), (centroid_box[1] - centroid_ra[1]))
                            if pixel_distance <= 20:
                                regain_counter += 1
                                self.player_types[i] = 0
                                self.change_player_type(id, 0)
                                id_set.add(id)
                                key_set.add(key)
                                proper_regain = True

                    if same_set is True:
                        for key in regain_active_copy:
                            centroid_box = (int((x1+x2)/2), int((y1+y2)/2))
                            centroid_ra = int((regain_active_copy[key][0] + regain_active_copy[key][2])/2), int(
                                (regain_active_copy[key][1] + regain_active_copy[key][3]) / 2)
                            pixel_distance = math.hypot(
                                (centroid_box[0] - centroid_ra[0]), (centroid_box[1] - centroid_ra[1]))
                            if pixel_distance <= 20:
                                regain_counter += 1
                                self.player_types[i] = 0
                                self.change_player_type(id, 0)
                                id_set.add(id)
                                key_set.add(key)
                                proper_regain = True
                        same_set = False

                    t_size = cv2.getTextSize(
                        str(id)+":"+str(directions[i]), cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
                    color = (0, 0, 255)
                    if self.player_types[i] == 0:
                        color = (66, 89, 13)
                        self.fielder_count += 1
                    elif self.player_types[i] == 1:
                        color = (0, 0, 255)
                        self.others_count += 1
                    elif self.player_types[i] == 2:
                        color = (0, 0, 255)
                        self.others_count += 1
                    elif self.player_types[i] == 9:
                        self.fielder_count += 1
                        color = (0, 172, 120)
                    else:
                        self.others_count += 1
                    if highlights[i] == 1:
                        color = (245, 66, 200)
                    if str(id) in self.batsmen_ids:
                        color = (0, 100, 255)
                    if(id in self.umpire_id):
                        color = (255, 0, 0)
                        if self.livelock_ids_saved:  # what is happening here ?
                            history_data[str(id)] = [
                                self.homo_track[int(id)][0],
                                self.homo_track[int(id)][1],
                                3,
                                directions[i]
                            ]

                    if id in self.ingap_ids:
                        color = (160, 0, 160)
                    if id in self.gap_ids or id in self.multi_gap_ids:
                        color = (150, 0, 150)
                    if highlight_streaks[i] == 1:
                        color = (255, 0, 0)
                        highlight_streaks_list.append(str(id))
                    for ele in self.distance_list:
                        if id == ele[1] or id == ele[2]:
                            color = (0, 255, 255)
                    for key in self.player_connect_l__dict.keys():
                        if self.player_connect_l__dict[key] == id:
                            color = (255, 255, 255)
                    if self.activate_distance_id != -1 and self.activate_distance_id == id:
                        color = (0, 255, 255)
                    if (str(id) == self.detect_fielder_id_TA):
                        color = (139, 69, 19)
                    elif (str(id) == self.detect_fielder_id_TB):
                        color = (170, 255, 0)
                    elif (str(id) == self.detect_fielder_id_PO):
                        color = (255, 0, 179)
                    if id == self.dummy_connect_id:
                        color = (159, 197, 232)
                    if id == self.dummy_player_id:
                        color = (159, 197, 0)
                    if id == self.send_player_speed_id:
                        color = (138, 40, 70)

                    current_coords_temp[id] = [
                        int(bbox[0]) - 10, int(bbox[1]) - 10, int(bbox[2]) + 10, int(bbox[3]) + 10]
                    txt_name = str(id) + ":" + str(directions[i])
                    if str(id) in self.fielder_dict.keys():
                        txt_name += ":" + str(self.fielder_dict[str(id)])
                    if str(id) in self.fielder_dict_PO.keys():
                        if self.fielder_dict_PO[str(id)] == -1:
                            pass
                        else:
                            txt_name += ":" + str(self.fielder_dict_PO[str(id)])

                    if not self.livelock_ids_saved:
                        cv2.rectangle(
                            im0s,
                            (x1-10, y1-10),
                            (x2+10, y2+10),
                            color, 3)
                        cv2.rectangle(
                            im0s,
                            (x1-10, y1-10),
                            (x1 - 10 + t_size[0] + 3, y1 - 10 + t_size[1] + 4),
                            color, -1)
                        cv2.putText(
                            im0s,
                            txt_name,
                            (x1 - 10, y1 - 10 + t_size[1] + 4),
                            cv2.FONT_HERSHEY_PLAIN,
                            2, [255, 255, 255], 2)
                    if self.livelock_ids_saved and (str(id) not in self.batsmen_ids) and (str(id) not in self.livelock_ids.keys()):
                        cv2.rectangle(
                            im0s,
                            (x1-10, y1-10),
                            (x2+10, y2+10),
                            color, 3)
                        cv2.rectangle(
                            im0s,
                            (x1-10, y1-10),
                            (x1 - 10 + t_size[0] + 3, y1 - 10 + t_size[1] + 4),
                            color, -1)
                        cv2.putText(
                            im0s,
                            txt_name,
                            (x1 - 10, y1 - 10 + t_size[1] + 4),
                            cv2.FONT_HERSHEY_PLAIN,
                            2, [255, 255, 255], 2)
                    if self.player_types[i] == 0 or self.player_types[i] == 9:
                        if int(id) not in [self.dummy_player_id]:
                            if self.udp_string == "":
                                self.udp_string += "PLAYER" + str(id) + "|"
                                self.ue4_string += "P" + str(id) + "$$"
                            else:
                                self.udp_string += "\0" + \
                                    "PLAYER" + str(id) + "|"
                                self.ue4_string += "##" + "P" + str(id) + "$$"
                        active_track.append(str(id))
                        if int(highlights[i]) == 1:
                            highlight_track.append(str(id))
                    if id in self.batsmen_ids:
                        if self.udp_string == "":
                            self.udp_string += "PLAYER" + str(id) + "|"
                            active_track.append(str(id))
                        else:
                            self.udp_string += "\0" + "PLAYER" + str(id) + "|"
                            active_track.append(str(id))
                    self.current_coords = current_coords_temp
                    # homography
                    if self.frm_count % 5 == 0:
                        obj = [int(bbox[0])+10, int(bbox[1])+10,
                               int(bbox[2])-10, int(bbox[3])-10]

                        if self.process_config.lens_distortion_flag == 1:
                            homo_pt = calculate_homography(
                                obj=obj,
                                cam_matrix=self.process_config.cam_matrix,
                                lens_distortion_flag=self.process_config.lens_distortion_flag,
                                newcameramtx=self.process_config.newcameramtx,
                                lens_dist=self.process_config.lens_dist)
                        else:
                            homo_pt = calculate_homography(
                                obj=obj,
                                cam_matrix=self.process_config.cam_matrix)

                        homo_pt = (int(homo_pt[0][0][0]),
                                   int(homo_pt[0][0][1]))
                        x1 = homo_pt[0]
                        y1 = homo_pt[1]

                        if id not in self.homo_track.keys():
                            self.homo_track[id] = [x1, y1]
                        else:
                            if abs(self.homo_track[id][0] - x1) < 7:
                                x1 = self.homo_track[id][0]
                            else:
                                self.homo_track[id][0] = x1
                            if abs(self.homo_track[id][1] - y1) < 7:
                                y1 = self.homo_track[id][1]
                            else:
                                self.homo_track[id][1] = y1

                        homo_pt = (x1, y1)
                        if self.outside_circle is True:
                            if not self.outside_circle_calc(x1, y1):
                                if self.player_types[i] == 0:
                                    outside_circle_set.add(id)
                            else:
                                outside_circle_set.discard(id)
                            outside_circle_status = False
                        else:
                            outside_circle_set = set()

                    if (self.frm_count > self.n_init+4) and (id in self.homo_track):
                        if int(id) not in [self.dummy_player_id]:
                            if self.player_types[i] == 0 or self.player_types[i] == 9:
                                if str(id) in self.fielder_dict.keys():
                                    player_name = str(
                                        self.fielder_dict[str(id)])
                                else:
                                    player_name = str(-1)

                                if str(id) in self.fielder_dict_PO.keys():
                                    player_position = str(
                                        self.fielder_dict_PO[str(id)])
                                else:
                                    player_position = str(-1)

                                self.udp_string += str(
                                    self.homo_track[id][0]) + "_"
                                self.udp_string += str(
                                    (self.homo_track[id][1]) * -1) + "_"
                                self.udp_string += str(
                                    int(self.player_types[i]))
                                self.ue4_string += "X=" + str(round((self.homo_track[int(id)][0] - self.axis_offset[1][0])*self.process_config.m_for_pix*100)) + ",Y=" + str(
                                    round((self.homo_track[int(id)][1] - self.axis_offset[1][1])*self.process_config.m_for_pix*100)) + ",Z=0"  # ,NAME="+player_name+",POSITION="+player_position
                                if self.ue4_fielder_string == "":
                                    self.ue4_fielder_string = "@@P" + \
                                        str(id) + ","+player_name + \
                                        ","+player_position
                                else:
                                    self.ue4_fielder_string += "##P" + \
                                        str(id) + ","+player_name + \
                                        ","+player_position

                        if id in self.batsmen_ids:
                            self.udp_string += str(
                                self.homo_track[id][0]) + "_"
                            self.udp_string += str(
                                (self.homo_track[id][1]) * -1) + "_"
                            self.udp_string += str(int(5))

                if self.frm_count % 1 == 0:
                    self.outside_circle_players = len(outside_circle_set)
                    try:
                        self.over_ball_dict["over"]
                    except KeyError:
                        self.over_ball_dict["over"] = -1
                    try:
                        self.over_ball_dict["ball"]
                    except KeyError:
                        self.over_ball_dict["ball"] = -1

            while key_set == id_set and self.regain_flag and same_set is False and key_set != set() and id_set != set():
                same_set = True
                key_set = set()
                id_set = set()
                active_track = []
                id = -1
                continue

            if self.regain_flag is True and proper_regain is True and same_set is False:
                # print(key_set, id_set, active_track)
                self.regain_flag = False
                proper_regain = False
                self.regain_active = {}
                continue

            # calculate homo for FN points and add to udp str and draw rectangles
            false_negatives_copy = copy.deepcopy(self.false_negatives)
            for key in false_negatives_copy.keys():
                obj = [
                    int(false_negatives_copy[key][0])+10,
                    int(false_negatives_copy[key][1])+10,
                    int(false_negatives_copy[key][2] +
                        false_negatives_copy[key][0])-10,
                    int(false_negatives_copy[key][3] +
                        false_negatives_copy[key][1])-10
                ]
                if self.process_config.lens_distortion_flag == 1:
                    homo_pt = calculate_homography(
                        obj=obj,
                        cam_matrix=self.process_config.cam_matrix,
                        lens_distortion_flag=self.process_config.lens_distortion_flag,
                        newcameramtx=self.process_config.newcameramtx,
                        lens_dist=self.process_config.lens_dist)
                else:
                    homo_pt = calculate_homography(
                        obj=obj,
                        cam_matrix=self.process_config.cam_matrix)
                homo_pt = (int(homo_pt[0][0][0]), int(homo_pt[0][0][1]))
                x1 = homo_pt[0]
                y1 = homo_pt[1]
                self.homo_track[key] = [x1, y1]
                color = (0, 0, 0)
                if self.frm_count > self.n_init+4:
                    if key != self.dummy_player_id:
                        self.udp_string += "\0PLAYER" + str(int(key)) + "|"
                        self.udp_string += str(x1) + "_"
                        self.udp_string += str((y1) * -1) + "_"
                    if str(key) in self.fielder_dict.keys():
                        player_name = str(
                            self.fielder_dict[str(key)])
                    else:
                        player_name = str(-1)

                    if str(key) in self.fielder_dict_PO.keys():
                        player_position = str(
                            self.fielder_dict_PO[str(key)])
                    else:
                        player_position = str(-1)
                    if self.ue4_string == "":
                        self.ue4_string += "P" + str(int(key)) + "$$"
                    else:
                        self.ue4_string += "##" + "P" + str(int(key)) + "$$"
                    self.ue4_string += "X=" + str(round((x1 - self.axis_offset[1][0])*self.process_config.m_for_pix*100)) + ",Y=" + str(
                        round((y1 - self.axis_offset[1][1])*self.process_config.m_for_pix*100)) + ",Z=0"
                    if int(key) == self.bowler_id:
                        self.udp_string += str(9)
                    else:
                        self.udp_string += str(0)
                    if self.ue4_fielder_string == "":
                        self.ue4_fielder_string = "@@P" + \
                            str(int(key)) + ","+player_name + \
                            ","+player_position
                    else:
                        self.ue4_fielder_string += "##P" + \
                            str(int(key)) + ","+player_name + \
                            ","+player_position
                    active_track.append(str(int(key)))
                    if key == self.bowler_id:
                        color = (0, 172, 120)
                    
                    for ele in self.distance_list:
                        if key == ele[1] or key == ele[2]:
                            color = (0, 255, 255)
                    if self.activate_distance_id != -1 and self.activate_distance_id == key:
                        color = (0, 255, 255)
                    if key in self.ingap_ids:
                        color = (160, 0, 160)
                    if key in self.gap_ids or key in self.multi_gap_ids:
                        color = (150, 0, 150)
                    if key in self.highlight_fns:
                        color = (245, 66, 200)
                        highlight_track.append(str(int(key)))
                    for key_ in self.player_connect_l__dict.keys():
                        if self.player_connect_l__dict[key_] == key:
                            color = (255, 255, 255)
                    if key == self.dummy_connect_id:
                        color = (159, 197, 232)
                    if key == self.dummy_player_id:
                        color = (159, 197, 0)

                    if (str(key) == self.detect_fielder_id_TA):
                        color = (139, 69, 19)
                    elif (str(key) == self.detect_fielder_id_TB):
                        color = (139, 69, 19)
                    elif (str(key) == self.detect_fielder_id_PO):
                        color = (139, 69, 19)

                txt_name = "ND"
                if str(key) in self.fielder_dict.keys():
                    txt_name += ":" + str(self.fielder_dict[str(key)])
                if str(key) in self.fielder_dict_PO.keys():
                    txt_name += ":" + str(self.fielder_dict_PO[str(key)])
                cv2.rectangle(
                    im0s,
                    (int(false_negatives_copy[key][0])+10,
                     int(false_negatives_copy[key][1])+10),
                    (int(false_negatives_copy[key][2] + false_negatives_copy[key][0])-10,
                     int(false_negatives_copy[key][3] + false_negatives_copy[key][1])-10),
                    color, 2)

                cv2.rectangle(
                    im0s,
                    (int(false_negatives_copy[key][0])-10,
                     int(false_negatives_copy[key][1])-10),
                    (int(false_negatives_copy[key][0]) - 10 + t_size[0] + 3,
                     int(false_negatives_copy[key][1]) - 10 + t_size[1] + 4),
                    color, -1)
                cv2.putText(
                    im0s,
                    txt_name,
                    (int(false_negatives_copy[key][0]) - 10,
                     int(false_negatives_copy[key][1]) - 10 + t_size[1] + 4),
                    cv2.FONT_HERSHEY_PLAIN,
                    2, [255, 255, 255], 2)
                push_data["players"].append({
                    "identities": key,
                    "bbox": (int(false_negatives_copy[key][0])+10,
                             int(false_negatives_copy[key][1])+10,
                             int(false_negatives_copy[key][2] +
                                 false_negatives_copy[key][0])-10,
                             int(false_negatives_copy[key][3]+false_negatives_copy[key][1])-10),
                    "player_type": -1,
                    "direction": -1,
                    "homo_track": (x1, y1)
                }
                )
                downstream_data["players"][str(key)] = {
                    "identities": key,
                    "bbox": (int(false_negatives_copy[key][0])+10,
                             int(false_negatives_copy[key][1])+10,
                             int(false_negatives_copy[key][2] +
                                 false_negatives_copy[key][0])-10,
                             int(false_negatives_copy[key][3]+false_negatives_copy[key][1])-10),
                    "player_type": -1,
                    "direction": -1,
                    "homo_track": (x1, y1)
                }

            
            # print("selected j player :", self.send_player_speed_id)
            ########
            if self.speed_new_id:
                print("new________________")
                bowler_dist = 0
                self.send_player_speed_val = []
                self.highest_speed = 0
                self.speed_new_id = False
            if self.send_player_speed_id != -1:
                if old_homo_track:
                    for index, new_id in enumerate(self.homo_track.keys()):
                        if new_id == self.send_player_speed_id:
                            if new_id in old_homo_track.keys() and new_id in list(self.identities):

                                new_id_index = list(
                                    self.identities).index(new_id)
                                # if list(self.player_types)[new_id_index] == 0: //green player condition
                                pt1 = self.homo_track[new_id]
                                pt2 = old_homo_track[new_id]
                                bowler_dist += math.hypot(
                                    (pt1[0] - pt2[0]), (pt1[1] - pt2[1]))*self.process_config.m_for_pix
                                # print("updating",bowler_dist)
                            # else:
                            #     print(
                            #         "Marked id not present in last frame or in current frame")
                            #     self.send_player_speed_id = -1
                            #     bowler_dist = 0
                old_homo_track = copy.deepcopy(self.homo_track)
            else:
                bowler_dist = 0
                self.send_player_speed_val = []
            # print(self.send_player_speed_val,bowler_dist)
            if (int(time.time() - starttime) == 1) and self.send_player_speed_id != -1:
                starttime = time.time()
                # sending m/s
                # self.send_player_speed_val.append([copy.deepcopy(time.time()),copy.deepcopy(
                #     round(bowler_dist, 2))])
                # sending km/hr
                self.send_player_speed_val.append(
                    [copy.deepcopy(time.time()), copy.deepcopy(round(bowler_dist*3.6, 2))])

                # if round(bowler_dist,2) > self.highest_speed:
                #     self.highest_speed = bowler_dist
                #     self.fastest_id = new_id
                #     self.fastid_frno = self.frm_count
                # print(
                #     f"{self.send_player_speed_id} player speed - {self.send_player_speed_val}m/s {round(self.send_player_speed_val*3.6,2)}km/hr")
                # print("highest speed",self.highest_speed)

                bowler_dist = 0
            elif (int(time.time() - starttime) > 1):
                starttime = time.time()
                old_homo_track = {}
                bowler_dist = 0

            ########

            # if (int(time.time() - starttime) == 1):
            #     if old_homo_track:
            #         if self.send_player_speed_id != -1:
            #             for index, new_id in enumerate(self.homo_track.keys()):
            #                 if new_id == self.send_player_speed_id:
            #                     if new_id in old_homo_track.keys() and new_id in list(self.identities):

            #                         new_id_index = list(
            #                             self.identities).index(new_id)
            #                         # if list(self.player_types)[new_id_index] == 0: //green player condition
            #                         pt1 = self.homo_track[new_id]
            #                         pt2 = old_homo_track[new_id]
            #                         bowler_dist = round(math.hypot(
            #                             (pt1[0] - pt2[0]), (pt1[1] - pt2[1]))*self.process_config.m_for_pix, 2)
            #                         self.send_player_speed_val = bowler_dist
            #                         # if bowler_dist > self.highest_speed:
            #                         #     self.highest_speed = bowler_dist
            #                         #     self.fastest_id = new_id
            #                         #     self.fastid_frno = self.frm_count
            #                         print(
            #                             f"{new_id} player speed - {bowler_dist}m/s {round(bowler_dist*3.6,2)}km/hr")
            #                     else:
            #                         print(
            #                             "Marked id not present in last frame or in current frame")
            #                         self.send_player_speed_id = -1
            #         else:
            #             self.send_player_speed_val = -1
            #     old_homo_track = copy.deepcopy(self.homo_track)
            #     starttime = time.time()
            # elif (int(time.time() - starttime) > 1):
            #     starttime = time.time()
            #     old_homo_track = {}

            false_negatives_slipFielders_copy = copy.deepcopy(
                self.false_negatives_slipFielders)
            for key in false_negatives_slipFielders_copy.keys():
                obj = [
                    int(false_negatives_slipFielders_copy[key][0])+10,
                    int(false_negatives_slipFielders_copy[key][1])+10,
                    int(false_negatives_slipFielders_copy[key][2] +
                        false_negatives_slipFielders_copy[key][0])-10,
                    int(false_negatives_slipFielders_copy[key][3] +
                        false_negatives_slipFielders_copy[key][1])-10
                ]
                if self.process_config.lens_distortion_flag == 1:
                    homo_pt = calculate_homography(
                        obj=obj,
                        cam_matrix=self.process_config.cam_matrix,
                        lens_distortion_flag=self.process_config.lens_distortion_flag,
                        newcameramtx=self.process_config.newcameramtx,
                        lens_dist=self.process_config.lens_dist)
                else:
                    homo_pt = calculate_homography(
                        obj=obj,
                        cam_matrix=self.process_config.cam_matrix)
                homo_pt = (int(homo_pt[0][0][0]), int(homo_pt[0][0][1]))
                x1 = homo_pt[0]
                y1 = homo_pt[1]
                self.homo_track[key] = [x1, y1]
                color = (128, 0, 128)
                if self.frm_count > self.n_init+4:
                    if key != self.dummy_player_id:
                        self.udp_string += "\0PLAYER" + str(int(key)) + "|"
                        self.udp_string += str(x1) + "_"
                        self.udp_string += str((y1) * -1) + "_"
                    if str(key) in self.fielder_dict.keys():
                        player_name = str(
                            self.fielder_dict[str(key)])
                    else:
                        player_name = str(-1)

                    if str(key) in self.fielder_dict_PO.keys():
                        player_position = str(
                            self.fielder_dict_PO[str(key)])
                    else:
                        player_position = str(-1)
                    if self.ue4_string == "":
                        self.ue4_string += "P" + str(int(key)) + "$$"
                    else:
                        self.ue4_string += "##" + "P" + str(int(key)) + "$$"
                    self.ue4_string += "X=" + str(round((x1 - self.axis_offset[1][0])*self.process_config.m_for_pix*100)) + ",Y=" + str(
                        round((y1 - self.axis_offset[1][1])*self.process_config.m_for_pix*100)) + ",Z=0"
                    if int(key) == self.bowler_id:
                        self.udp_string += str(9)
                    else:
                        self.udp_string += str(0)
                    if self.ue4_fielder_string == "":
                        self.ue4_fielder_string = "@@P" + \
                            str(int(key)) + ","+player_name + \
                            ","+player_position
                    else:
                        self.ue4_fielder_string += "##P" + \
                            str(int(key)) + ","+player_name + \
                            ","+player_position
                    active_track.append(str(int(key)))
                    if key == self.bowler_id:
                        color = (0, 172, 120)
                    
                    for ele in self.distance_list:
                        if key == ele[1] or key == ele[2]:
                            color = (0, 255, 255)
                    if self.activate_distance_id != -1 and self.activate_distance_id == key:
                        color = (0, 255, 255)
                    if key in self.ingap_ids:
                        color = (160, 0, 160)
                    if key in self.gap_ids or key in self.multi_gap_ids:
                        color = (150, 0, 150)
                    if key in self.highlight_fns:
                        color = (245, 66, 200)
                        highlight_track.append(str(int(key)))
                    for key_ in self.player_connect_l__dict.keys():
                        if self.player_connect_l__dict[key_] == key:
                            color = (255, 255, 255)

                    if key == self.dummy_connect_id:
                        color = (159, 197, 232)
                    if key == self.dummy_player_id:
                        color = (159, 197, 0)

                    if (str(key) == self.detect_fielder_id_TA):
                        color = (139, 69, 19)
                    elif (str(key) == self.detect_fielder_id_TB):
                        color = (139, 69, 19)
                    elif (str(key) == self.detect_fielder_id_PO):
                        color = (139, 69, 19)

                txt_name = "LD"
                if str(key) in self.fielder_dict.keys():
                    txt_name += ":" + str(self.fielder_dict[str(key)])
                if str(key) in self.fielder_dict_PO.keys():
                    txt_name += ":" + str(self.fielder_dict_PO[str(key)])

                cv2.rectangle(im0s, (int(false_negatives_slipFielders_copy[key][0])+10, int(false_negatives_slipFielders_copy[key][1])+10), (int(
                    false_negatives_slipFielders_copy[key][2] + false_negatives_slipFielders_copy[key][0])-10, int(false_negatives_slipFielders_copy[key][3] + false_negatives_slipFielders_copy[key][1])-10), color, 2)

                cv2.rectangle(
                    im0s,
                    (int(false_negatives_slipFielders_copy[key][0])-10, int(
                        false_negatives_slipFielders_copy[key][1])-10),
                    (int(false_negatives_slipFielders_copy[key][0]) - 10 + t_size[0] + 3, int(
                        false_negatives_slipFielders_copy[key][1]) - 10 + t_size[1] + 4),
                    color, -1)
                cv2.putText(
                    im0s,
                    txt_name,
                    (int(false_negatives_slipFielders_copy[key][0]) - 10, int(
                        false_negatives_slipFielders_copy[key][1]) - 10 + t_size[1] + 4),
                    cv2.FONT_HERSHEY_PLAIN,
                    2, [255, 255, 255], 2)
                push_data["players"].append({
                    "identities": key,
                    "bbox": (int(false_negatives_slipFielders_copy[key][0])+10,
                             int(false_negatives_slipFielders_copy[key][1])+10,
                             int(false_negatives_slipFielders_copy[key][2] +
                                 false_negatives_slipFielders_copy[key][0])-10,
                             int(false_negatives_slipFielders_copy[key][3]+false_negatives_slipFielders_copy[key][1])-10),
                    "player_type": -1,
                    "direction": -1,
                    "homo_track": (x1, y1)
                }
                )
                downstream_data["players"][str(key)] = {
                    "identities": key,
                    "bbox": (int(false_negatives_slipFielders_copy[key][0])+10,
                             int(false_negatives_slipFielders_copy[key][1])+10,
                             int(false_negatives_slipFielders_copy[key][2] +
                                 false_negatives_slipFielders_copy[key][0])-10,
                             int(false_negatives_slipFielders_copy[key][3]+false_negatives_slipFielders_copy[key][1])-10),
                    "player_type": -1,
                    "direction": -1,
                    "homo_track": (x1, y1)
                }

            false_negatives_outside_frame_z_copy = copy.deepcopy(
                self.false_negatives_outside_frame_z)
            for key in false_negatives_outside_frame_z_copy.keys():
                obj = [
                    int(false_negatives_outside_frame_z_copy[key][0])+10,
                    int(false_negatives_outside_frame_z_copy[key][1])+10,
                    int(false_negatives_outside_frame_z_copy[key][2] +
                        false_negatives_outside_frame_z_copy[key][0])-10,
                    int(false_negatives_outside_frame_z_copy[key][3] +
                        false_negatives_outside_frame_z_copy[key][1])-10
                ]
                if self.process_config.lens_distortion_flag == 1:
                    homo_pt = calculate_homography(
                        obj=obj,
                        cam_matrix=self.process_config.cam_matrix,
                        lens_distortion_flag=self.process_config.lens_distortion_flag,
                        newcameramtx=self.process_config.newcameramtx,
                        lens_dist=self.process_config.lens_dist)
                else:
                    homo_pt = calculate_homography(
                        obj=obj,
                        cam_matrix=self.process_config.cam_matrix)
                homo_pt = (int(homo_pt[0][0][0]), int(homo_pt[0][0][1]))
                x1 = homo_pt[0]
                y1 = homo_pt[1]
                self.homo_track[key] = [x1, y1]
                color = (0, 0, 0)
                if self.frm_count > self.n_init+4:
                    self.udp_string += "\0PLAYER" + str(int(key)) + "|"
                    self.udp_string += str(x1) + "_"
                    self.udp_string += str((y1) * -1) + "_"
                    self.udp_string += str(0)
                    if str(key) in self.fielder_dict.keys():
                        player_name = str(
                            self.fielder_dict[str(key)])
                    else:
                        player_name = str(-1)

                    if str(key) in self.fielder_dict_PO.keys():
                        player_position = str(
                            self.fielder_dict_PO[str(key)])
                    else:
                        player_position = str(-1)
                    if self.ue4_string == "":
                        self.ue4_string += "P" + str(int(key)) + "$$"
                    else:
                        self.ue4_string += "##" + "P" + str(int(key)) + "$$"
                    self.ue4_string += "X=" + str(round((x1 - self.axis_offset[1][0])*self.process_config.m_for_pix*100)) + ",Y=" + str(
                        round((y1 - self.axis_offset[1][1])*self.process_config.m_for_pix*100)) + ",Z=0"
                    if self.ue4_fielder_string == "":
                        self.ue4_fielder_string = "@@P" + \
                            str(int(key)) + ","+player_name + \
                            ","+player_position
                    else:
                        self.ue4_fielder_string += "##P" + \
                            str(int(key)) + ","+player_name + \
                            ","+player_position
                    active_track.append(str(int(key)))
                    for ele in self.distance_list:
                        if key == ele[1] or key == ele[2]:
                            color = (0, 255, 255)
                    if self.activate_distance_id != -1 and self.activate_distance_id == key:
                        color = (0, 255, 255)
                    if key in self.ingap_ids:
                        color = (160, 0, 160)
                    if key in self.gap_ids or key in self.multi_gap_ids:
                        color = (150, 0, 150)
                    if key in self.highlight_fns:
                        color = (245, 66, 200)
                        highlight_track.append(str(int(key)))
                    for key_ in self.player_connect_l__dict.keys():
                        if self.player_connect_l__dict[key_] == key:
                            color = (255, 255, 255)
                    if key == self.dummy_connect_id:
                        color = (159, 197, 232)
                    if key == self.dummy_player_id:
                        color = (159, 197, 0)

                    if (str(key) == self.detect_fielder_id_TA):
                        color = (139, 69, 19)
                    elif (str(key) == self.detect_fielder_id_TB):
                        color = (139, 69, 19)
                    elif (str(key) == self.detect_fielder_id_PO):
                        color = (139, 69, 19)

                txt_name = "ZD"
                if str(key) in self.fielder_dict.keys():
                    txt_name += ":" + str(self.fielder_dict[str(key)])
                if str(key) in self.fielder_dict_PO.keys():
                    txt_name += ":" + str(self.fielder_dict_PO[str(key)])
                cv2.rectangle(
                    im0s,
                    (int(false_negatives_outside_frame_z_copy[key][0])+10,
                     int(false_negatives_outside_frame_z_copy[key][1])+10),
                    (int(false_negatives_outside_frame_z_copy[key][2] + false_negatives_outside_frame_z_copy[key][0])-10,
                     int(false_negatives_outside_frame_z_copy[key][3] + false_negatives_outside_frame_z_copy[key][1])-10),
                    color, 2)
                cv2.rectangle(
                    im0s,
                    (int(false_negatives_outside_frame_z_copy[key][0])-10, int(
                        false_negatives_outside_frame_z_copy[key][1])-10),
                    (int(false_negatives_outside_frame_z_copy[key][0]) - 10 + t_size[0] + 3, int(
                        false_negatives_outside_frame_z_copy[key][1]) - 10 + t_size[1] + 4),
                    color, -1)
                cv2.putText(
                    im0s,
                    txt_name,
                    (int(false_negatives_outside_frame_z_copy[key][0]) - 10, int(
                        false_negatives_outside_frame_z_copy[key][1]) - 10 + t_size[1] + 4),
                    cv2.FONT_HERSHEY_PLAIN,
                    2, [255, 255, 255], 2)
                push_data["players"].append({
                    "identities": key,
                    "bbox": (int(false_negatives_outside_frame_z_copy[key][0])+10,
                             int(
                                 false_negatives_outside_frame_z_copy[key][1])+10,
                             int(false_negatives_outside_frame_z_copy[key][2] +
                                 false_negatives_outside_frame_z_copy[key][0])-10,
                             int(false_negatives_outside_frame_z_copy[key][3]+false_negatives_outside_frame_z_copy[key][1])-10),
                    "player_type": -1,
                    "direction": -1,
                    "homo_track": (x1, y1)
                }
                )
                downstream_data["players"][str(key)] = {
                    "identities": key,
                    "bbox": (int(false_negatives_outside_frame_z_copy[key][0])+10,
                             int(
                                 false_negatives_outside_frame_z_copy[key][1])+10,
                             int(false_negatives_outside_frame_z_copy[key][2] +
                                 false_negatives_outside_frame_z_copy[key][0])-10,
                             int(false_negatives_outside_frame_z_copy[key][3]+false_negatives_outside_frame_z_copy[key][1])-10),
                    "player_type": -1,
                    "direction": -1,
                    "homo_track": (x1, y1)
                }

            false_negatives_outside_frame_a_copy = copy.deepcopy(
                self.false_negatives_outside_frame_a)
            for key in false_negatives_outside_frame_a_copy.keys():
                obj = [
                    int(false_negatives_outside_frame_a_copy[key][0])+10,
                    int(false_negatives_outside_frame_a_copy[key][1])+10,
                    int(false_negatives_outside_frame_a_copy[key][2] +
                        false_negatives_outside_frame_a_copy[key][0])-10,
                    int(false_negatives_outside_frame_a_copy[key][3] +
                        false_negatives_outside_frame_a_copy[key][1])-10
                ]
                if self.process_config.lens_distortion_flag == 1:
                    homo_pt = calculate_homography(
                        obj=obj,
                        cam_matrix=self.process_config.cam_matrix,
                        lens_distortion_flag=self.process_config.lens_distortion_flag,
                        newcameramtx=self.process_config.newcameramtx,
                        lens_dist=self.process_config.lens_dist)
                else:
                    homo_pt = calculate_homography(
                        obj=obj,
                        cam_matrix=self.process_config.cam_matrix)
                homo_pt = (int(homo_pt[0][0][0]), int(homo_pt[0][0][1]))
                x1 = homo_pt[0]
                y1 = homo_pt[1]
                self.homo_track[key] = [x1, y1]
                color = (0, 0, 0)
                if self.frm_count > self.n_init+4:
                    self.udp_string += "\0PLAYER" + str(int(key)) + "|"
                    self.udp_string += str(x1) + "_"
                    self.udp_string += str((y1) * -1) + "_"
                    self.udp_string += str(0)
                    if str(key) in self.fielder_dict.keys():
                        player_name = str(
                            self.fielder_dict[str(key)])
                    else:
                        player_name = str(-1)

                    if str(key) in self.fielder_dict_PO.keys():
                        player_position = str(
                            self.fielder_dict_PO[str(key)])
                    else:
                        player_position = str(-1)
                    if self.ue4_string == "":
                        self.ue4_string += "P" + str(int(key)) + "$$"
                    else:
                        self.ue4_string += "##" + "P" + str(int(key)) + "$$"
                    self.ue4_string += "X=" + str(round((x1 - self.axis_offset[1][0])*self.process_config.m_for_pix*100)) + ",Y=" + str(
                        round((y1 - self.axis_offset[1][1])*self.process_config.m_for_pix*100)) + ",Z=0"
                    if self.ue4_fielder_string == "":
                        self.ue4_fielder_string = "@@P" + \
                            str(int(key)) + ","+player_name + \
                            ","+player_position
                    else:
                        self.ue4_fielder_string += "##P" + \
                            str(int(key)) + ","+player_name + \
                            ","+player_position
                    active_track.append(str(int(key)))
                    for ele in self.distance_list:
                        if key == ele[1] or key == ele[2]:
                            color = (0, 255, 255)
                    if self.activate_distance_id != -1 and self.activate_distance_id == key:
                        color = (0, 255, 255)
                    if key in self.ingap_ids:
                        color = (160, 0, 160)
                    if key in self.gap_ids or key in self.multi_gap_ids:
                        color = (150, 0, 150)
                    if key in self.highlight_fns:
                        color = (245, 66, 200)
                        highlight_track.append(str(int(key)))
                    for key_ in self.player_connect_l__dict.keys():
                        if self.player_connect_l__dict[key_] == key:
                            color = (255, 255, 255)
                    if key == self.dummy_connect_id:
                        color = (159, 197, 232)
                    if key == self.dummy_player_id:
                        color = (159, 197, 0)

                    if (str(key) == self.detect_fielder_id_TA):
                        color = (139, 69, 19)
                    elif (str(key) == self.detect_fielder_id_TB):
                        color = (139, 69, 19)
                    elif (str(key) == self.detect_fielder_id_PO):
                        color = (139, 69, 19)
                txt_name = "AD"
                if str(key) in self.fielder_dict.keys():
                    txt_name += ":" + str(self.fielder_dict[str(key)])
                if str(key) in self.fielder_dict_PO.keys():
                    txt_name += ":" + str(self.fielder_dict_PO[str(key)])

                cv2.rectangle(
                    im0s,
                    (int(false_negatives_outside_frame_a_copy[key][0])+10,
                     int(false_negatives_outside_frame_a_copy[key][1])+10),
                    (int(false_negatives_outside_frame_a_copy[key][2] + false_negatives_outside_frame_a_copy[key][0])-10,
                     int(false_negatives_outside_frame_a_copy[key][3] + false_negatives_outside_frame_a_copy[key][1])-10),
                    color, 2)

                cv2.rectangle(
                    im0s,
                    (int(false_negatives_outside_frame_a_copy[key][0])-10, int(
                        false_negatives_outside_frame_a_copy[key][1])-10),
                    (int(false_negatives_outside_frame_a_copy[key][0]) - 10 + t_size[0] + 3, int(
                        false_negatives_outside_frame_a_copy[key][1]) - 10 + t_size[1] + 4),
                    color, -1)
                cv2.putText(
                    im0s,
                    txt_name,
                    (int(false_negatives_outside_frame_a_copy[key][0]) - 10, int(
                        false_negatives_outside_frame_a_copy[key][1]) - 10 + t_size[1] + 4),
                    cv2.FONT_HERSHEY_PLAIN,
                    2, [255, 255, 255], 2)
                push_data["players"].append({
                    "identities": key,
                    "bbox": (int(false_negatives_outside_frame_a_copy[key][0])+10,
                             int(
                                 false_negatives_outside_frame_a_copy[key][1])+10,
                             int(false_negatives_outside_frame_a_copy[key][2] +
                                 false_negatives_outside_frame_a_copy[key][0])-10,
                             int(false_negatives_outside_frame_a_copy[key][3]+false_negatives_outside_frame_a_copy[key][1])-10),
                    "player_type": -1,
                    "direction": -1,
                    "homo_track": (x1, y1)
                }
                )
                downstream_data["players"][str(key)] = {
                    "identities": key,
                    "bbox": (int(false_negatives_outside_frame_a_copy[key][0])+10,
                             int(
                                 false_negatives_outside_frame_a_copy[key][1])+10,
                             int(false_negatives_outside_frame_a_copy[key][2] +
                                 false_negatives_outside_frame_a_copy[key][0])-10,
                             int(false_negatives_outside_frame_a_copy[key][3]+false_negatives_outside_frame_a_copy[key][1])-10),
                    "player_type": -1,
                    "direction": -1,
                    "homo_track": (x1, y1)
                }

            self.p_fielder = ""
            false_negatives_outside_frame_u_copy = copy.deepcopy(
                self.false_negatives_outside_frame_u)
            for key in false_negatives_outside_frame_u_copy.keys():
                obj = [
                    int(false_negatives_outside_frame_u_copy[key][0])+10,
                    int(false_negatives_outside_frame_u_copy[key][1])+10,
                    int(false_negatives_outside_frame_u_copy[key][2] +
                        false_negatives_outside_frame_u_copy[key][0])-10,
                    int(false_negatives_outside_frame_u_copy[key][3] +
                        false_negatives_outside_frame_u_copy[key][1])-10
                ]
                if self.process_config.lens_distortion_flag == 1:
                    homo_pt = calculate_homography(
                        obj=obj,
                        cam_matrix=self.process_config.cam_matrix,
                        lens_distortion_flag=self.process_config.lens_distortion_flag,
                        newcameramtx=self.process_config.newcameramtx,
                        lens_dist=self.process_config.lens_dist)
                else:
                    homo_pt = calculate_homography(
                        obj=obj,
                        cam_matrix=self.process_config.cam_matrix)
                homo_pt = (int(homo_pt[0][0][0]), int(homo_pt[0][0][1]))
                x1 = homo_pt[0]
                y1 = homo_pt[1]
                self.homo_track[key] = [x1, y1]
                color = (167, 145, 16)
                if self.frm_count > self.n_init+4:
                    if self.dummy_connect_id == -1 and self.dummy_player_id == -1:
                        if self.p_fielder == "":
                            self.p_fielder = "\0DUMMY|PLAYER" + \
                                str(int(key)) + "_"
                        else:
                            self.p_fielder += "#PLAYER" + str(int(key)) + "_"

                        self.p_fielder += str(x1) + "_"
                        self.p_fielder += str((y1) * -1) + "_"
                        self.p_fielder += str(0) + \
                            "\0DUMMY_CONNECT|-1" + "\0DUMMY_PLAYER|-1"
                    elif self.dummy_connect_id != -1 and self.dummy_player_id == -1:
                        self.p_fielder = "\0DUMMY|-1"+"\0DUMMY_CONNECT|" + str(x1) + "_" + str(
                            (y1) * -1) + "_" + str(0) + "_" + str(self.dummy_connect_id) + "\0DUMMY_PLAYER|-1"
                    elif self.dummy_connect_id == -1 and self.dummy_player_id != -1 and self.dummy_player_id in self.homo_track.keys():
                        if self.dummy_player_falg:
                            self.dummy_player_cache = copy.deepcopy(
                                self.homo_track[self.dummy_player_id])
                            self.dummy_player_falg = False
                        else:
                            self.p_fielder = "\0DUMMY|-1"+"\0DUMMY_CONNECT|-1"+"\0DUMMY_PLAYER|"+str(self.dummy_player_cache[0]) + "_" + str(
                                self.dummy_player_cache[1]*-1)+"_" + str(0) + "_" + str(x1) + "_" + str((y1) * -1) + "_" + str(0)
                    for ele in self.distance_list:
                        if key == ele[1] or key == ele[2]:
                            color = (0, 255, 255)
                    if self.activate_distance_id != -1 and self.activate_distance_id == key:
                        color = (0, 255, 255)
                    if key in self.ingap_ids:
                        color = (160, 0, 160)
                    if key in self.gap_ids or key in self.multi_gap_ids:
                        color = (150, 0, 150)
                    if key in self.highlight_fns:
                        color = (245, 66, 200)
                        highlight_track.append(str(int(key)))
                    for key_ in self.player_connect_l__dict.keys():
                        if self.player_connect_l__dict[key_] == key:
                            color = (255, 255, 255)
                cv2.rectangle(
                    im0s,
                    (int(false_negatives_outside_frame_u_copy[key][0])+10,
                     int(false_negatives_outside_frame_u_copy[key][1])+10),
                    (int(false_negatives_outside_frame_u_copy[key][2] + false_negatives_outside_frame_u_copy[key][0])-10,
                     int(false_negatives_outside_frame_u_copy[key][3] + false_negatives_outside_frame_u_copy[key][1])-10),
                    color, 2)
            if self.p_fielder == "":
                self.p_fielder = "\0DUMMY|-1"+"\0DUMMY_CONNECT|-1" + "\0DUMMY_PLAYER|-1"
            false_negatives_mark_point_p_copy = copy.deepcopy(
                self.false_negatives_mark_point_p)
            for key in false_negatives_mark_point_p_copy.keys():
                obj = [
                    int(false_negatives_mark_point_p_copy[key][0])+10,
                    int(false_negatives_mark_point_p_copy[key][1])+10,
                    int(false_negatives_mark_point_p_copy[key][2] +
                        false_negatives_mark_point_p_copy[key][0])-10,
                    int(false_negatives_mark_point_p_copy[key][3]+false_negatives_mark_point_p_copy[key][1])-10]
                if self.process_config.lens_distortion_flag == 1:
                    homo_pt = calculate_homography(
                        obj=obj,
                        cam_matrix=self.process_config.cam_matrix,
                        lens_distortion_flag=self.process_config.lens_distortion_flag,
                        newcameramtx=self.process_config.newcameramtx,
                        lens_dist=self.process_config.lens_dist)
                else:
                    homo_pt = calculate_homography(
                        obj=obj,
                        cam_matrix=self.process_config.cam_matrix)
                homo_pt = (int(homo_pt[0][0][0]), int(homo_pt[0][0][1]))
                x1 = homo_pt[0]
                y1 = homo_pt[1]
                self.homo_track[key] = [x1, y1]
                color = (165, 42, 42)  # brown
                if self.frm_count > self.n_init+4:
                    for ele in self.distance_list:
                        if key == ele[1] or key == ele[2]:
                            color = (0, 255, 255)
                    if self.activate_distance_id != -1 and self.activate_distance_id == key:
                        color = (0, 255, 255)
                    if key in self.ingap_ids:
                        color = (160, 0, 160)
                    if key in self.gap_ids or key in self.multi_gap_ids:
                        color = (150, 0, 150)
                cv2.rectangle(
                    im0s,
                    (int(false_negatives_mark_point_p_copy[key][0])+10,
                     int(false_negatives_mark_point_p_copy[key][1])+10),
                    (int(false_negatives_mark_point_p_copy[key][2] + false_negatives_mark_point_p_copy[key][0])-10,
                     int(false_negatives_mark_point_p_copy[key][3] + false_negatives_mark_point_p_copy[key][1])-10),
                    color, 2)
            false_negatives_mark_point_o_copy = copy.deepcopy(
                self.false_negatives_mark_point_o)
            for key in false_negatives_mark_point_o_copy.keys():
                obj = [int(false_negatives_mark_point_o_copy[key][0])+10, int(false_negatives_mark_point_o_copy[key][1])+10, int(false_negatives_mark_point_o_copy[key]
                                                                                                                                 [2]+false_negatives_mark_point_o_copy[key][0])-10, int(false_negatives_mark_point_o_copy[key][3]+false_negatives_mark_point_o_copy[key][1])-10]
                if self.process_config.lens_distortion_flag == 1:
                    homo_pt = calculate_homography(
                        obj=obj,
                        cam_matrix=self.process_config.cam_matrix,
                        lens_distortion_flag=self.process_config.lens_distortion_flag,
                        newcameramtx=self.process_config.newcameramtx,
                        lens_dist=self.process_config.lens_dist)
                else:
                    homo_pt = calculate_homography(
                        obj=obj,
                        cam_matrix=self.process_config.cam_matrix)
                homo_pt = (int(homo_pt[0][0][0]), int(homo_pt[0][0][1]))
                x1 = homo_pt[0]
                y1 = homo_pt[1]
                self.homo_track[key] = [x1, y1]
                color = (255, 42, 42)  # brown
                if(self.frm_count > self.n_init+4):
                    for ele in self.distance_list:
                        if((key == ele[1]) or (key == ele[2])):
                            color = (0, 255, 255)
                    if((self.activate_distance_id != -1) and (self.activate_distance_id == key)):
                        color = (0, 255, 255)
                    if(key in self.ingap_ids):
                        color = (160, 0, 160)
                    if(key in self.gap_ids) or key in self.multi_gap_ids:
                        color = (150, 0, 150)
                if(len(self.wk_pt) == 4):
                    cv2.rectangle(im0s, (int(false_negatives_mark_point_o_copy[key][0])+10, int(false_negatives_mark_point_o_copy[key][1])+10), (int(
                        false_negatives_mark_point_o_copy[key][2] + false_negatives_mark_point_o_copy[key][0])-10, int(false_negatives_mark_point_o_copy[key][3] + false_negatives_mark_point_o_copy[key][1])-10), color, 2)

            if(self.wk_pt and (124 in self.homo_track.keys())):
                if self.udp_string == "":
                    self.udp_string += "PLAYER" + str(124) + "|" + str(
                        self.homo_track[124][0]) + "_" + str(self.homo_track[124][1]*-1) + "_" + str(5)
                    # self.ue4_string += "P" + str(124) + "$$" + "X=" + str(round((self.homo_track[124][0] - self.axis_offset[1][0])*self.process_config.m_for_pix*100)) + ",Y=" + str(round((self.homo_track[124][1] - self.axis_offset[1][1])*self.process_config.m_for_pix*100)) +",Z=0"
                else:
                    self.udp_string += "\0PLAYER" + str(124) + "|" + str(
                        self.homo_track[124][0]) + "_" + str(self.homo_track[124][1]*-1) + "_" + str(5)
                    # self.ue4_string += "##P" + str(124) + "$$" + "X=" + str(round((self.homo_track[124][0] - self.axis_offset[1][0])*self.process_config.m_for_pix*100)) + ",Y=" + str(round((self.homo_track[124][1] - self.axis_offset[1][1])*self.process_config.m_for_pix*100)) +",Z=0"

                active_track.append(str(124))

            self.fn_count = len(self.false_negatives) + \
                len(self.false_negatives_slipFielders) + len(
                    self.false_negatives_outside_frame_a)+len(self.false_negatives_outside_frame_z)
            if self.frm_count % 1 == 0:
                self.outside_circle_players = len(outside_circle_set)
                try:
                    temp_over = self.over_ball_dict["over"]
                except KeyError:
                    self.over_ball_dict["over"] = -1
                try:
                    temp_ball = self.over_ball_dict["ball"]
                except KeyError:
                    self.over_ball_dict["ball"] = -1
                push_data["over"] = self.over_ball_dict["over"]
                push_data["ball"] = self.over_ball_dict["ball"]
                downstream_data["over"] = self.over_ball_dict["over"]
                downstream_data["ball"] = self.over_ball_dict["ball"]
            
            # if len(self.last_frame_locations_homograph) == 0:
            #     for homography_slop_id in self.livelock_ids.keys():
            #         self.slop_dict[homography_slop_id] = 0

            if len(self.last_frame_locations_homograph) > 0:
                for homography_slop_id in self.livelock_ids.keys():
                    try:
                        slop = math.sqrt(self.homo_track[homography_slop_id][0]**2 + self.homo_track[homography_slop_id][1]**2)-math.sqrt(self.last_frame_locations_homograph[homography_slop_id][0]**2 + self.last_frame_locations_homograph[homography_slop_id][1]**2)
                        self.slop_dict[homography_slop_id] = slop
                        # print(slop,homography_slop_id)
                        if slop > 100.0:
                            # self.unnormal_speed_detected(homography_slop_id)
                            self.ubnormal_speed_data[homography_slop_id] = self.last_frame_locations_homograph[homography_slop_id]
                    except:
                        pass
                
                for u_homography_slop_id in self.umpire_id:
                    try:
                        slop = math.sqrt(self.homo_track[u_homography_slop_id][0]**2 + self.homo_track[u_homography_slop_id][1]**2)-math.sqrt(self.last_frame_locations_homograph[u_homography_slop_id][0]**2 + self.last_frame_locations_homograph[u_homography_slop_id][1]**2)
                        self.slop_dict[u_homography_slop_id] = slop
                        # print(slop,u_homography_slop_id)
                        if slop > 100.0:
                            # self.unnormal_speed_detected(u_homography_slop_id)
                            self.ubnormal_speed_data[u_homography_slop_id] = self.last_frame_locations_homograph[u_homography_slop_id]
                    except:
                        pass

                for b_homography_slop_id in self.batsmen_ids_automated:
                    try:
                        slop = math.sqrt(self.homo_track[b_homography_slop_id][0]**2 + self.homo_track[b_homography_slop_id][1]**2)-math.sqrt(self.last_frame_locations_homograph[b_homography_slop_id][0]**2 + self.last_frame_locations_homograph[b_homography_slop_id][1]**2)
                        self.slop_dict[b_homography_slop_id] = slop
                        # print(slop,b_homography_slop_id)
                        if slop > 100.0:
                            # self.unnormal_speed_detected(b_homography_slop_id)
                            self.ubnormal_speed_data[b_homography_slop_id] = self.last_frame_locations_homograph[b_homography_slop_id]
                    except:
                        pass

            if self.frm_count %25 ==0:
                self.last_frame_locations_homograph = copy.deepcopy(self.homo_track)

            for uid in self.umpire_id:
                self.fielder_dict_PO[str(int(uid))] = 'UMPIRE'
                if len(list(self.main_ump_tuple.keys())) ==0:
                    P1_X, P1_Y, P2_X, P2_Y = get_relative_location(int(uid), self.flip_field_plot, self.left_handed, self.process_config.far_end_stump,
                                                           self.process_config.near_end_stump, self.homo_track)

                    r_, tan_ = get_polar_coordinates(P1_X, P1_Y)
                    if tan_ >250:
                        self.main_ump_tuple = {int(uid):tuple(self.homo_track[uid])}

            for bid in self.batsmen_ids_automated:
                self.fielder_dict_PO[str(int(bid))] = 'BATSMAN'
            
            self.sort_tracker.main_ump_tuple = self.main_ump_tuple
            self.sort_tracker.ump_tuple = self.ump_tuple
            self.sort_tracker.bastmen_tuple = self.bastmen_tuple
            # print(downstream_data["players"].keys())
            if (-1 in self.reswap_ids) and (len(self.reswap_ids) ==2):
                self.reswap_ids = []
            if self.reswap_flag == 1:
                if (str(self.reswap_ids[0]-1) in list(self.sort_tracker.trackers.keys())) and (str(self.reswap_ids[1]-1) in list(self.sort_tracker.trackers.keys())):
                    self.sort_tracker.reswap(int(self.reswap_ids[0])-1,int(self.reswap_ids[1])-1)
                self.reswap_flag = 0
                self.reswap_ids = []
            
            if self.record_flag == 1:
                
                # cv2.imwrite('save_data/'+filename+'.jpg', save_im0s)
                # f = open('save_data/'+filename+".txt", "w")
                for i, bbox in enumerate(self.bbox_xyxy):
                    x1, y1, x2, y2 = [int(i) for i in bbox]
                    
                    if self.identities[i] in self.batsmen_ids_automated:
                        # print(self.player_types[i])
                        filename = str(int(len(os.listdir('save_data/gallery/1'))))
                        cv2.imwrite('save_data/gallery/1/'+filename+'.jpg', save_im0s[y1:y2,x1:x2])
                        # pil_image_1 = Image.fromarray(cv2.cvtColor(save_im0s[y1:y2,x1:x2], cv2.COLOR_BGR2RGB))
                        # img_tensor_1 = self.sort_tracker.data_transforms(pil_image_1).flatten()
                        # torch.cat((img_tensor_1, torch.tensor(0)), 1)
                        # self.data = pd.concat([self.data, pd.DataFrame(torch.cat((img_tensor_1, torch.tensor([0])), 0).numpy())], ignore_index=True)
                        self.count_images +=1
                    elif self.identities[i] in self.umpire_id: 
                        # print(self.player_types[i])
                        # filename = str(int(len(os.listdir('save_data/gallery/2'))))
                        # cv2.imwrite('save_data/gallery/2/'+filename+'.jpg', save_im0s[y1:y2,x1:x2])
                        pil_image_1 = Image.fromarray(cv2.cvtColor(save_im0s[y1:y2,x1:x2], cv2.COLOR_BGR2RGB))
                        img_tensor_1 = self.sort_tracker.data_transforms(pil_image_1).flatten()
                        self.data = pd.concat([self.data, pd.DataFrame(torch.cat((img_tensor_1, torch.tensor([0])), 0).numpy())], ignore_index=True)
                        self.count_images +=1
                    else:
                        if self.player_types[i] == 0:
                            # filename = str(int(len(os.listdir('save_data/gallery/0'))))
                            # cv2.imwrite('save_data/gallery/0/'+filename+'.jpg', save_im0s[y1:y2,x1:x2])
                            pil_image_1 = Image.fromarray(cv2.cvtColor(save_im0s[y1:y2,x1:x2], cv2.COLOR_BGR2RGB))
                            img_tensor_1 = self.sort_tracker.data_transforms(pil_image_1).flatten()
                            self.data = pd.concat([self.data, pd.DataFrame(torch.cat((img_tensor_1, torch.tensor([0])), 0).numpy())], ignore_index=True)
                            self.count_images +=1
                    if self.count_images > 5000:
                        self.data.to_csv('save_data/training_data.csv',index = False)
                        print("data saved")
                        self.record_flag = 0
                        # self.train_svm()
                        break
                 

            # print("\n")
            # self.unnormal_speed_detected()
            if self.flip_field_plot == 0:
                batting_end = self.process_config.near_end_stump
            else:
                batting_end = self.process_config.far_end_stump
            
            if self.livelock_status:
                # print("inside livelock")
                if self.fielder_count + self.fn_count == 11 and self.livelock_ids_saved is False:
                    self.livelock_ids_saved = True
                    for id in active_track:
                        if id in downstream_data["players"].keys():
                            self.livelock_ids[str(
                                id)] = self.homo_track[int(id)]
                            self.livelock_ids_extra[str(id)] = copy.deepcopy(
                                downstream_data["players"][str(id)])
                if self.livelock_ids_saved:
                    self.udp_string = ""
                    self.ue4_string = ""
                    self.ue4_fielder_string = ""
                    for id in self.batsmen_ids:
                        if id not in self.livelock_ids:
                            for element in reversed(self.distance_list):
                                if str(element[1]) in self.batsmen_ids or str(element[2]) in self.batsmen_ids:
                                    self.livelock_ids_extra[str(id)] = copy.deepcopy(
                                        downstream_data["players"][str(id)])
                                    self.livelock_ids[str(
                                        id)] = self.homo_track[int(id)]

                    livelock_ids_copy = copy.deepcopy(self.livelock_ids)
                    self.collect_fielder_info(livelock_ids_copy)
                    # print(self.player_types)
                    # print(self.identities)
                    
                    
                    # self.check_for_loss_ids()
                    for id in livelock_ids_copy.keys():
                        if id in active_track:
                            if id in downstream_data["players"].keys():
                                self.livelock_ids[id] = self.homo_track[int(
                                    id)]
                                self.livelock_ids_extra[str(id)] = copy.deepcopy(
                                    downstream_data["players"][str(id)])
                        if id not in self.livelock_ids_extra.keys():
                            print("skipping")
                            continue
                        bbox = self.livelock_ids_extra[str(id)]["bbox"]
                        player_type = self.livelock_ids_extra[str(
                            id)]["player_type"]
                        direction = self.livelock_ids_extra[str(
                            id)]["direction"]

                        txt_name = str(id) + ":" + str(direction)
                        if str(id) in self.fielder_dict.keys():
                            txt_name += ":" + str(self.fielder_dict[str(id)])

                        if str(id) in self.fielder_dict_PO.keys():
                            txt_name += ":" + \
                                str(self.fielder_dict_PO[str(id)])

                        t_size = cv2.getTextSize(
                            txt_name, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]

                        color = (0, 0, 255)
                        if player_type == -1:
                            color = (0, 0, 0)
                        if player_type == 0:
                            color = (66, 89, 13)
                        elif player_type == 1:
                            player_type = 3
                            color = (0, 0, 255)
                        elif player_type == 2:
                            player_type = 3
                            color = (0, 0, 255)

                        if int(id) == self.bowler_id:
                            # print("hello")
                            color = (0, 172, 120)
                        

                        if str(id) in self.highlights_list:
                            # print("highlight")
                            color = (245, 66, 200)
                        if str(id) in self.batsmen_ids:
                            color = (0, 100, 255)
                        if self.umpire_id == int(id):
                            color = (255, 0, 0)
                        if int(id) in self.ingap_ids:
                            color = (160, 0, 160)
                        if int(id) in self.gap_ids or int(id) in self.multi_gap_ids:
                            color = (150, 0, 150)
                        if str(id) in highlight_streaks_list:
                            color = (255, 0, 0)

                        for ele in self.distance_list:
                            if int(id) == ele[1] or int(id) == ele[2]:
                                color = (0, 255, 255)
                        for key in self.player_connect_l__dict.keys():
                            if self.player_connect_l__dict[key] == int(id):
                                color = (255, 255, 255)
                        if self.activate_distance_id != -1 and self.activate_distance_id == int(id):
                            color = (0, 255, 255)

                        if (str(id) == self.detect_fielder_id_TA):
                            color = (139, 69, 19)
                        elif (str(id) == self.detect_fielder_id_TB):
                            color = (139, 69, 19)
                        elif (str(id) == self.detect_fielder_id_PO):
                            color = (139, 69, 19)

                        if int(id) == self.dummy_connect_id:
                            color = (159, 197, 232)
                        if int(id) == self.dummy_player_id:
                            color = (159, 197, 0)

                        if int(id) == self.send_player_speed_id:
                            color = (138, 40, 70)

                        current_coords_temp[int(id)] = [
                            int(bbox[0]) - 10,
                            int(bbox[1]) - 10,
                            int(bbox[2]) + 10,
                            int(bbox[3]) + 10]
                        if player_type != -1 and id not in self.batsmen_ids and id in active_track:
                            cv2.rectangle(
                                im0s,
                                (bbox[0]-10, bbox[1]-10),
                                (bbox[2]+10, bbox[3]+10),
                                color, 3)
                            cv2.rectangle(
                                im0s,
                                (bbox[0]-10, bbox[1]-10),
                                (bbox[0] - 10 + t_size[0] + 3,
                                 bbox[1] - 10 + t_size[1] + 4),
                                color, -1)
                            cv2.putText(
                                im0s,
                                txt_name,
                                (bbox[0] - 10, bbox[1] - 10 + t_size[1] + 4),
                                cv2.FONT_HERSHEY_PLAIN,
                                2, [255, 255, 255], 2)
                        else:
                            cv2.rectangle(
                                im0s,
                                (bbox[0], bbox[1]),
                                (bbox[2], bbox[3]),
                                color, 3)
                        if int(id) not in [self.dummy_player_id]:
                            if self.udp_string == "":
                                self.udp_string += "PLAYER" + str(id) + "|"
                                self.ue4_string += "P" + str(id) + "$$"
                            else:
                                self.udp_string += "\0" + \
                                    "PLAYER" + str(id) + "|"
                                self.ue4_string += "##" + "P" + str(id) + "$$"

                        if str(id) in self.fielder_dict.keys():
                            player_name = str(self.fielder_dict[str(id)])
                        else:
                            player_name = str(-1)

                        if str(id) in self.fielder_dict_PO.keys():
                            player_position = str(
                                self.fielder_dict_PO[str(id)])
                        else:
                            player_position = str(-1)

                        
                        livelock_data[str(id)] = {}
                        livelock_data[str(id)]["identities"] = id
                        # livelock_data[str(id)]["x_homo"] = self.homo_track[int(id)][0]
                        # livelock_data[str(id)]["y_homo"] = self.homo_track[int(id)][1]
                        livelock_data[str(id)]["homo_track"] = [
                            self.homo_track[int(id)][0], self.homo_track[int(id)][1]]
                        livelock_data[str(id)]["player_name"] = player_name
                        livelock_data[str(
                            id)]["player_position"] = player_position

                        history_data[str(id)] = [
                            self.homo_track[int(id)][0],
                            self.homo_track[int(id)][1],
                            0,
                            direction
                        ]
                        if int(id) not in [self.dummy_player_id]:

                            self.udp_string += str(
                                self.homo_track[int(id)][0]) + "_"
                            self.udp_string += str(
                                (self.homo_track[int(id)][1]) * -1) + "_"
                            self.ue4_string += "X=" + str(round((self.homo_track[int(id)][0] - self.axis_offset[1][0])*self.process_config.m_for_pix*100)) + ",Y=" + str(
                                round((self.homo_track[int(id)][1] - self.axis_offset[1][1])*self.process_config.m_for_pix*100)) + ",Z=0"  # ,NAME="+player_name+",POSITION="+player_position
                            if self.ue4_fielder_string == "":
                                self.ue4_fielder_string = "@@P" + \
                                    str(id) + ","+player_name + \
                                    ","+player_position
                            else:
                                self.ue4_fielder_string += "##P" + \
                                    str(id) + ","+player_name + \
                                    ","+player_position
                            # print(self.axis_offset[1])
                            if id in self.batsmen_ids:
                                self.udp_string += str(5)
                            elif int(id) == self.bowler_id:
                                self.udp_string += str(9)
                                # print("here|",int(id))
                            else:
                                self.udp_string += str(0)
                    active_track = list(self.livelock_ids.keys())
                    if(self.wk_pt and (124 in self.homo_track.keys())):
                        if self.udp_string == "":
                            self.udp_string += "PLAYER" + str(124) + "|" + str(
                                self.homo_track[124][0]) + "_" + str(self.homo_track[124][1]*-1) + "_" + str(5)
                            # self.ue4_string += "P" + str(124) + "$$" + "X=" + str(round((self.homo_track[124][0] - self.axis_offset[1][0])*self.process_config.m_for_pix*100)) + ",Y=" + str(round((self.homo_track[124][1] - self.axis_offset[1][1])*self.process_config.m_for_pix*100)) +",Z=0"
                        else:
                            self.udp_string += "\0PLAYER" + str(124) + "|" + str(
                                self.homo_track[124][0]) + "_" + str(self.homo_track[124][1]*-1) + "_" + str(5)
                            # self.ue4_string += "##P" + str(124) + "$$" + "X=" + str(round((self.homo_track[124][0] - self.axis_offset[1][0])*self.process_config.m_for_pix*100)) + ",Y=" + str(round((self.homo_track[124][1] - self.axis_offset[1][1])*self.process_config.m_for_pix*100)) +",Z=0"
                        active_track.append(str(124))

            


            if self.last_livelock_status == 1:
                if self.livelock_status == 0:
                    self.livelock_count = 0
                    self.last_livelock_status = 0
                    self.sort_tracker.tagging_info={}


            if self.frm_count > self.n_init+4:
                self.ue4_string += self.ue4_fielder_string
                self.udp_string += "\0" + "FLIP|" + str(self.flip_field_plot)
                self.ue4_string += "@@FLIP$$" + str(self.flip_field_plot)
                self.udp_string += "\0" + "POS|" + \
                    str(int(self.left_handed is False))
                self.ue4_string += "@@POS$$" + \
                    str(int(self.left_handed is False))

                if str(self.dummy_player_id) in active_track:
                    active_track.remove(str(self.dummy_player_id))
                active_track_str = "_".join(active_track)
                self.udp_string += "\0" + "ACTIVE|" + active_track_str
                for element in self.distance_list:
                    if str(element[1]) not in self.highlights_list:
                        self.highlights_list.append(str(element[1]))
                    if str(element[2]) not in self.highlights_list:
                        self.highlights_list.append(str(element[2]))
                if self.highlights_list == []:
                    highlight_track_str = str(-1)
                    push_data["highlight"] = "-1"
                    unreal_highlight_list = str(-1)
                else:
                    highlight_track_str = ""
                    push_data["highlight"] = self.highlights_list
                    highlight_track_str = "_".join(self.highlights_list)
                    unreal_highlight_track_str =  []
                    for ue4_x in range(len((self.highlights_list))):
                       if self.highlights_list[ue4_x]!= '124' and self.highlights_list[ue4_x]!= '131' and self.highlights_list[ue4_x]!= '151':
                            unreal_highlight_track_str.append(str(self.highlights_list[ue4_x]))
                    if unreal_highlight_track_str == []:
                        unreal_highlight_list= str(-1)
                    else:
                        unreal_P_highlight = map(
                            (lambda x: 'P' + x), unreal_highlight_track_str)
                        unreal_highlight_list = "$$".join(unreal_P_highlight)
                self.udp_string += "\0" + "HIGHLIGHT|" + highlight_track_str
                self.ue4_string += "@@HIGHLIGHT$$" + unreal_highlight_list
                if len(self.player_connect_l__dict.keys()) == 0:
                    self.udp_string += "\0PLAYER_CONNECT|-1"
                    push_data["player_connect"] = "-1"
                    self.ue4_string += "@@PLAYER_CONNECT$$-1"
                else:
                    connect_ids = [str(self.player_connect_l__dict[key])
                                   for key in self.player_connect_l__dict.keys()]
                    self.ue4_string += "@@PLAYER_CONNECT$$"
                    if(len(connect_ids) == 1):
                        self.ue4_string += "P" + str(connect_ids[0]) + "$$"
                    else:
                        for x in range(len(connect_ids)-1):
                            dist = round(distance_between(
                                player1=int(connect_ids[x]),
                                player2=int(connect_ids[x+1]),
                                m_for_pix=self.process_config.m_for_pix,
                                homo_track=self.homo_track))
                            self.ue4_string += "P" + \
                                str(connect_ids[x]) + "$$" + "P" + \
                                str(connect_ids[x+1]) + \
                                "$$" + str(dist)+"m" + "$$"
                    if len(connect_ids) < 5:
                        self.ue4_string += str(-1)
                    self.udp_string += "\0PLAYER_CONNECT|" + \
                        "_".join(connect_ids)
                    push_data["player_connect"] = connect_ids

                if self.batsmen_ids != []:
                    batsmen_ids_str = "_".join(self.batsmen_ids)
                else:
                    batsmen_ids_str = "-1"

                self.udp_string += "\0" + "BATSMEN|" + str(batsmen_ids_str)
                if len(highlight_streaks_list) > 0:
                    highlight_streaks_str = "_".join(highlight_streaks_list)
                else:
                    highlight_streaks_str = str(0)
                self.udp_string += "\0" + "HIGHLIGHT_STREAK|" + highlight_streaks_str
                if self.process_config.collision_mode == 1:
                    for trackk_1 in active_track:
                        for trackk_2 in active_track:
                            if int(trackk_1) != int(trackk_2):
                                if int(trackk_1) in self.homo_track.keys() and int(trackk_2) in self.homo_track.keys():
                                    if math.hypot(self.homo_track[int(trackk_1)][0] - self.homo_track[int(trackk_2)][0], self.homo_track[int(trackk_1)][1] - self.homo_track[int(trackk_2)][1]) < self.dist_thresh:
                                        if trackk_1 not in close_tracks:
                                            close_tracks.append(str(trackk_1))
                                        if trackk_2 not in close_tracks:
                                            close_tracks.append(str(trackk_2))
                    if close_tracks != []:
                        close_tracks_str = "_".join(close_tracks)
                    else:
                        close_tracks_str = str(-1)
                    self.udp_string += "\0" + "COLLISION|" + close_tracks_str
                boundary_str = ""
                unreal_boundary_str = ""
                if len(self.distance_list) > 0:
                    dist_str = "\0DISTANCE|"
                    unreal_distance_str = "@@DISTANCE$$"
                    i = -1
                    for element in reversed(self.distance_list):
                        i += 1
                        boundary_data = {}
                        if (element[1] in self.false_negatives_mark_point_p.keys()) or (element[2] in self.false_negatives_mark_point_p.keys()):
                            boundary_str = "\0BOUNDARY_DISTANCE|"
                            unreal_boundary_str = "@@BOUNDARY_DISTANCE$$"
                            dist = round(distance_between(
                                player1=element[1],
                                player2=element[2],
                                m_for_pix=self.process_config.m_for_pix,
                                homo_track=self.homo_track))
                            if (element[1] in self.false_negatives_mark_point_p.keys()) and (element[2] not in self.false_negatives_mark_point_p.keys()):
                                boundary_str = boundary_str + str(int(self.homo_track[element[1]][0])) + ":" + str(
                                    int(self.homo_track[element[1]][1]*-1)) + ":0_"
                                boundary_str = boundary_str + \
                                    str(element[2]) + "_" + str(dist)+"m"
                                unreal_boundary_str += "X=" + str(round((self.homo_track[element[1]][0] - self.axis_offset[1][0])*self.process_config.m_for_pix*100)) + ",Y=" + str(
                                    round((self.homo_track[element[1]][1] - self.axis_offset[1][1])*self.process_config.m_for_pix*100)) + ",Z=0" + "$$"
                                unreal_boundary_str += "P" + \
                                    str(element[2]) + "$$" + str(dist)+"m"
                                p1_cord = (int(self.homo_track[element[1]][0]), int(
                                    self.homo_track[element[1]][1]))
                                boundary_data = {
                                    "p1_cordinates": p1_cord,
                                    "player_id": element[2],
                                    "dist": str(dist)+"m"
                                }

                            elif (element[2] in self.false_negatives_mark_point_p.keys()) and (element[1] not in self.false_negatives_mark_point_p.keys()):
                                boundary_str = boundary_str + str(int(self.homo_track[element[2]][0])) + ":" + str(
                                    int(self.homo_track[element[2]][1] * -1)) + ":0_"
                                boundary_str = boundary_str + \
                                    str(element[1]) + "_" + str(dist)+"m"
                                unreal_boundary_str += "X=" + str(round((self.homo_track[element[2]][0] - self.axis_offset[1][0])*self.process_config.m_for_pix*100)) + ",Y=" + str(
                                    round((self.homo_track[element[2]][1] - self.axis_offset[1][1])*self.process_config.m_for_pix*100)) + ",Z=0" + "$$"
                                unreal_boundary_str += "P" + \
                                    str(element[1]) + "$$" + str(dist)+"m"
                                p1_cord = (int(self.homo_track[element[1]][0]), int(
                                    self.homo_track[element[1]][1]))
                                boundary_data = {
                                    "p1_cordinates": p1_cord,
                                    "player_id": element[1],
                                    "dist": str(dist)+"m"
                                }
                            else:
                                boundary_str = boundary_str + str(int(self.homo_track[element[1]][0])) + ":" + str(
                                    int(self.homo_track[element[1]][1] * -1)) + ":0_"
                                boundary_str = boundary_str + str(int(self.homo_track[element[2]][0])) + ":" + str(
                                    int(self.homo_track[element[2]][1] * -1)) + ":0_" + str(dist)+"m"
                                unreal_boundary_str += "X=" + str(round((self.homo_track[element[1]][0] - self.axis_offset[1][0])*self.process_config.m_for_pix*100)) + ",Y=" + str(
                                    round((self.homo_track[element[1]][1] - self.axis_offset[1][1])*self.process_config.m_for_pix*100)) + ",Z=0" + "$$"
                                unreal_boundary_str += "X=" + str(round((self.homo_track[element[2]][0] - self.axis_offset[1][0])*self.process_config.m_for_pix*100)) + ",Y=" + str(
                                    round((self.homo_track[element[2]][1] - self.axis_offset[1][1])*self.process_config.m_for_pix*100)) + ",Z=0" + "$$"
                                unreal_boundary_str += str(dist) + "m"
                                p1_cord = (int(self.homo_track[element[1]][0]), int(
                                    self.homo_track[element[1]][1]))
                                p2_cord = (int(self.homo_track[element[2]][0]), int(
                                    self.homo_track[element[2]][1]))
                                boundary_data = {
                                    "p1_cordinates": p1_cord,
                                    "p2_cordinates": p2_cord,
                                    "dist": str(dist)+"m"
                                }
                            continue
                        dist_str += (str(element[1])+"_"+str(element[2])+"_")

                        unreal_distance_str += ("P" +
                                                str(element[1])+"$$"+"P"+str(element[2])+"$$")

                        # unreal_distance_str += "X=" + str(round((self.homo_track[int(element[1])][0] - self.axis_offset[1][0])*self.process_config.m_for_pix*100)) + ",Y=" + str(round((self.homo_track[element[1]][1] - self.axis_offset[1][1])*self.process_config.m_for_pix*100)) +",Z=0" + "$$"
                        # unreal_distance_str += "X=" + str(round((self.homo_track[int(element[2])][0] - self.axis_offset[1][0])*self.process_config.m_for_pix*100)) + ",Y=" + str(round((self.homo_track[element[2]][1] - self.axis_offset[1][1])*self.process_config.m_for_pix*100)) +",Z=0" + "$$"
                        dist = round(distance_between(
                            player1=element[1],
                            player2=element[2],
                            m_for_pix=self.process_config.m_for_pix,
                            homo_track=self.homo_track))
                        if dist > 9:
                            dist_str += str(dist)+"m"
                            unreal_distance_str += str(dist)+"m"
                        else:
                            dist_str += "0"+str(dist)+"m"
                            unreal_distance_str += "0"+str(dist)+"m"
                        dist_dict = {
                            "id_1": element[1],
                            "id_2": element[2],
                            "dist_str": str(dist)+"m"
                        }

                    for pair in self.distance_list:
                        if 124 in pair[1:] and 151 not in pair[1:]:
                            unreal_distance_str = "@@DISTANCE$$-1"
                            pair = pair[1:]
                            pair.remove(124)
                            unreal_boundary_str = "@@BOUNDARY_DISTANCE$$1$$X={0},Y={1},Z=0$$P{2}$${3}m"
                            unreal_boundary_str = unreal_boundary_str.format(round((self.process_config.near_end_crease[0] - self.axis_offset[1][0])*self.process_config.m_for_pix*100), round(
                                (self.process_config.near_end_crease[1] - self.axis_offset[1][1])*self.process_config.m_for_pix*100), pair[0], dist)
                        elif 124 in pair[1:] and 151 in pair[1:]:
                            unreal_distance_str = "@@DISTANCE$$-1"
                            unreal_boundary_str = "@@BOUNDARY_DISTANCE$$2$$X={0},Y={1},Z=0$$X={2},Y={3},Z=0$${4}m"
                            unreal_boundary_str = unreal_boundary_str.format(round((self.process_config.near_end_crease[0] - self.axis_offset[1][0])*self.process_config.m_for_pix*100), round(
                                (self.process_config.near_end_crease[1] - self.axis_offset[1][1])*self.process_config.m_for_pix*100), round((self.homo_track[151][0] - self.axis_offset[1][0])*self.process_config.m_for_pix*100), round((self.homo_track[151][1] - self.axis_offset[1][1])*self.process_config.m_for_pix*100),dist)
                        elif 131 in pair[1:]:
                            unreal_distance_str = "@@DISTANCE$$-1"
                            pair = pair[1:]
                            pair.remove(131)
                            unreal_boundary_str = "@@BOUNDARY_DISTANCE$$3$$X={0},Y={1},Z=0$$P{2}$${3}m"
                            unreal_boundary_str = unreal_boundary_str.format(round((self.process_config.near_end_crease[0] - self.axis_offset[1][0])*self.process_config.m_for_pix*100), round(
                                (self.process_config.near_end_crease[1] - self.axis_offset[1][1])*self.process_config.m_for_pix*100), pair[0], dist)
                        else:
                            unreal_distance_str = "@@DISTANCE$$-1"

                    if dist_str == "\0DISTANCE|":
                        dist_str = "\0DISTANCE|-1"
                        unreal_distance_str = "@@DISTANCE$$-1"

                    self.udp_string += dist_str
                    self.ue4_string += unreal_distance_str
                    push_data["distance"] = dist_dict
                else:
                    self.udp_string += "\0DISTANCE|-1"
                    push_data["distance"] = "-1"
                    self.ue4_string += "@@DISTANCE$$-1"
                if boundary_str == "":
                    self.udp_string += "\0BOUNDARY_DISTANCE|-1"
                    push_data["boundary_distance"] = "-1"
                else:
                    self.udp_string += boundary_str
                    push_data["boundary_distance"] = boundary_data

                if unreal_boundary_str == "" or unreal_boundary_str == "@@BOUNDARY_DISTANCE$$":
                    unreal_boundary_str = "@@BOUNDARY_DISTANCE$$-1"

                if len(self.multi_gap_ids) > 1:
                    _dist_str = ""
                    push_data["multi_gap_data"] = {}
                    ue4_multi_gap_str = ""
                    for i, pair in enumerate(list(self.each_slice(self.multi_gap_ids))):
                        dist = round(distance_between(
                            player1=pair[0], player2=pair[1], m_for_pix=self.process_config.m_for_pix, homo_track=self.homo_track))
                        push_data["multi_gap_data"].update({
                            "gap_id1": pair[0], "gap_id2": pair[1], "gap_dist": dist})
                        if dist > 9:
                            _dist_str += str(dist)+"m"
                        else:
                            _dist_str += "0"+str(dist)+"m"
                        if ue4_multi_gap_str == "":
                            if self.flip_field_plot == 0:  # near-end
                                ue4_multi_gap_str = "@@MULTI_GAP$$" + "X=" + str(round((self.process_config.near_end_crease[0] - self.axis_offset[1][0])*self.process_config.m_for_pix*100)) + ",Y=" + str(round(
                                    (self.process_config.near_end_crease[1] - self.axis_offset[1][1])*self.process_config.m_for_pix*100)) + ",Z=0" + "$$" + "P" + str(pair[0])+"$$" + "P" + str(pair[1]) + "$$" + str(dist)+"m"
                            if self.flip_field_plot == 1:  # far-end
                                ue4_multi_gap_str = "@@MULTI_GAP$$" + "X=" + str(round((self.process_config.far_end_crease[0] - self.axis_offset[1][0])*self.process_config.m_for_pix*100)) + ",Y=" + str(round(
                                    (self.process_config.far_end_crease[1] - self.axis_offset[1][1])*self.process_config.m_for_pix*100)) + ",Z=0" + "$$" + "P" + str(pair[0])+"$$" + "P" + str(pair[1]) + "$$" + str(dist)+"m"

                        else:
                            if self.flip_field_plot == 0:  # near-end
                                ue4_multi_gap_str += "$$" + "P" + \
                                    str(pair[0])+"$$" + "P" + \
                                    str(pair[1]) + "$$" + str(dist)+"m"
                            if self.flip_field_plot == 1:  # far-end
                                ue4_multi_gap_str += "$$" + "P" + \
                                    str(pair[0])+"$$" + "P" + \
                                    str(pair[1]) + "$$" + str(dist)+"m"

                else:
                    push_data["multi_gap_data"] = "-1"
                    ue4_multi_gap_str = "@@MULTI_GAP$$-1"

                if len(self.gap_ids) == 2:
                    _dist_str = ""
                    dist = round(distance_between(
                        player1=self.gap_ids[0], player2=self.gap_ids[1], m_for_pix=self.process_config.m_for_pix, homo_track=self.homo_track))
                    push_data["gap_data"] = {
                        "gap_id1": self.gap_ids[0], "gap_id2": self.gap_ids[1], "gap_dist": dist}
                    if dist > 9:
                        _dist_str += str(dist)+"m"
                    else:
                        _dist_str += "0"+str(dist)+"m"
                    if self.flip_field_plot == 0:  # near-end
                        gap_string = "\0GAP|" + str(self.process_config.near_end_stump[0])+":"+str(self.process_config.near_end_stump[1]*-1) + ":0_" + str(self.homo_track[self.gap_ids[0]][0])+":"+str(
                            self.homo_track[self.gap_ids[0]][1]*-1) + ":0_" + str(self.homo_track[self.gap_ids[1]][0])+":"+str(self.homo_track[self.gap_ids[1]][1]*-1)+":0_"+_dist_str
                        ue4_gap_str = "@@GAP$$" + "X=" + str(round((self.process_config.near_end_crease[0] - self.axis_offset[1][0])*self.process_config.m_for_pix*100)) + ",Y=" + str(round(
                            (self.process_config.near_end_crease[1] - self.axis_offset[1][1])*self.process_config.m_for_pix*100)) + ",Z=0" + "$$" + "P" + str(self.gap_ids[0])+"$$" + "P" + str(self.gap_ids[1]) + "$$"+str(dist)+"m"
                    if self.flip_field_plot == 1:  # far-end
                        gap_string = "\0GAP|" + str(self.process_config.far_end_stump[0])+":"+str(self.process_config.far_end_stump[1]*-1) + ":0_" + str(self.homo_track[self.gap_ids[0]][0])+":"+str(
                            self.homo_track[self.gap_ids[0]][1]*-1) + ":0_" + str(self.homo_track[self.gap_ids[1]][0])+":"+str(self.homo_track[self.gap_ids[1]][1]*-1)+":0_"+_dist_str
                        ue4_gap_str = "@@GAP$$" + "X=" + str(round((self.process_config.far_end_crease[0] - self.axis_offset[1][0])*self.process_config.m_for_pix*100)) + ",Y=" + str(round(
                            (self.process_config.far_end_crease[1] - self.axis_offset[1][1])*self.process_config.m_for_pix*100)) + ",Z=0" + "$$" + "P" + str(self.gap_ids[0])+"$$" + "P" + str(self.gap_ids[1]) + "$$"+str(dist)+"m"

                else:
                    gap_string = "\0GAP|-1"
                    push_data["gap_data"] = "-1"
                    ue4_gap_str = "@@GAP$$-1"

                if len(self.ingap_ids) == 3:
                    push_data["ingap_data"] = {
                        "ingap_id1": self.ingap_ids[0], "ingap_id2": self.ingap_ids[1], "ingap_id2": self.ingap_ids[2]}
                    if self.flip_field_plot == 0:  # near-end
                        ingap_string = "\0CONE_GAP|" + str(self.process_config.near_end_stump[0])+":"+str(self.process_config.near_end_stump[1]*-1) + ":0_" + str(self.homo_track[self.ingap_ids[0]][0])+":"+str(
                            self.homo_track[self.ingap_ids[0]][1]*-1) + ":0_" + str(self.homo_track[self.ingap_ids[1]][0])+":"+str(self.homo_track[self.ingap_ids[1]][1]*-1)+":0_" + str(self.homo_track[self.ingap_ids[2]][0])+":"+str(self.homo_track[self.ingap_ids[2]][1]*-1)+":0"
                        unreal_ingap = "@@CONE_GAP$$" + "X=" + str(round((self.process_config.near_end_crease[0] - self.axis_offset[1][0])*self.process_config.m_for_pix*100)) + ",Y=" + str(round(
                            (self.process_config.near_end_crease[1] - self.axis_offset[1][1])*self.process_config.m_for_pix*100)) + ",Z=0" + "$$" + "P" + str(self.ingap_ids[0]) + "$$" + "P" + str(self.ingap_ids[1]) + "$$" + "P" + str(self.ingap_ids[2])
                    if self.flip_field_plot == 1:  # far-end
                        ingap_string = "\0CONE_GAP|" + str(self.process_config.far_end_stump[0])+":"+str(self.process_config.far_end_stump[1]*-1) + ":0_" + str(self.homo_track[self.ingap_ids[0]][0])+":"+str(
                            self.homo_track[self.ingap_ids[0]][1]*-1) + ":0_" + str(self.homo_track[self.ingap_ids[1]][0])+":"+str(self.homo_track[self.ingap_ids[1]][1]*-1)+":0_" + str(self.homo_track[self.ingap_ids[2]][0])+":"+str(self.homo_track[self.ingap_ids[2]][1]*-1)+":0"
                        unreal_ingap = "@@CONE_GAP$$" + "X=" + str(round((self.process_config.far_end_crease[0] - self.axis_offset[1][0])*self.process_config.m_for_pix*100)) + ",Y=" + str(round(
                            (self.process_config.far_end_crease[1] - self.axis_offset[1][1])*self.process_config.m_for_pix*100)) + ",Z=0" + "$$" + "P" + str(self.ingap_ids[0]) + "$$" + "P" + str(self.ingap_ids[1]) + "$$" + "P" + str(self.ingap_ids[2])
                else:
                    ingap_string = "\0CONE_GAP|-1"
                    push_data["ingap_data"] = "-1"
                    unreal_ingap = "@@CONE_GAP$$-1"

                self.udp_string += ingap_string
                self.udp_string += gap_string
                self.udp_string += self.p_fielder
                self.ue4_string += unreal_ingap
                self.ue4_string += ue4_gap_str
                self.ue4_string += ue4_multi_gap_str
                self.ue4_string += unreal_boundary_str

                self.current_coords = current_coords_temp
                cv2.putText(
                    im0s,
                    "Frame Count: " + str(self.frm_count),
                    (50, 50),
                    cv2.FONT_HERSHEY_PLAIN,
                    2, [255, 255, 255], 2)
                im0s = np.asarray(im0s)
                self.qout_frame.put(im0s)
                self.ue4_string += "|"

                #file_object = open('ue4_string.txt', 'a+')
                #file_object.write(self.ue4_string)
                #file_object.close()

                self.qout_fieldplt.put([self.udp_string, self.ue4_string])
                now = dt.utcnow()
                dt_string = now.strftime("%d/%m/%Y %H:%M:%S:%f")
                push_data["_id"] = str(dt_string)
                push_data["umpire_id"] = self.umpire_id

                if self.insertDb_status:
                    if self.dbName_status:
                        insert_mongo(
                            collection=self.collection,
                            push_data=push_data)
                        # insert_mongo(
                        #     collection=self.score_file_collection,
                        #     push_data=self.score_file_data)

                        if self.score_line != self.old_score_line2:
                            self.old_score_line2 = self.score_line
                            self.score_file_data["_id"] = str(dt_string)

                            print("new line added, pushing scoring data")
                            insert_mongo(
                                collection=self.score_file_collection,
                                push_data=self.score_file_data)

                    else:
                        print("Name not present so creating a new one")
                        db_date = dt.utcnow().strftime('%d-%m-%Y-%H-%M-%S-%f').replace("-", "_")
                        self.dynamicDb_name = self.process_config.db_name + \
                            str("_normal_data")
                        self.dynamicLivelock_dbName = self.process_config.db_name + \
                            str("_livelock_data")
                        self.dynamicScoreData_dbName = self.process_config.db_name + \
                            str("_score_data")
                        self.dynamicHistoryData_dbName = self.process_config.db_name + \
                            str("_history_data")

                        self.collection = self.db[self.dynamicDb_name]
                        self.collection_livelock = self.db[self.dynamicLivelock_dbName]
                        self.score_file_collection = self.db[self.dynamicScoreData_dbName]
                        self.score_history_data = self.db[self.dynamicHistoryData_dbName]

                        self.dbName_status = True
                        insert_mongo(
                            collection=self.collection,
                            push_data=push_data)

                        insert_mongo(
                            collection=self.score_history_data,
                            push_data=history_data)

                        # insert_mongo(
                        #     collection=self.score_file_collection,
                        #     push_data=self.score_file_data)

                        if self.score_line != self.old_score_line2:
                            self.old_score_line2 = self.score_line
                            self.score_file_data["_id"] = str(dt_string)

                            print("new line added, pushing scoring data")
                            insert_mongo(
                                collection=self.score_file_collection,
                                push_data=self.score_file_data)

                    if self.livelock_status and self.livelock_ids_saved:

                        livelock_data["_id"] = str(dt_string)
                        livelock_data["frame_count"] = self.frm_count
                        livelock_data["umpire_id"] = self.umpire_id
                        livelock_data["flip"] = self.flip_field_plot



                        history_data["_id"] = str(dt_string)
                        history_data["frame_count"] = self.frm_count

                        insert_mongo(
                            collection=self.collection_livelock,
                            push_data=livelock_data)
                        insert_mongo(
                            collection=self.score_history_data,
                            push_data=history_data)
                self.qout_qtman_data.put(push_data)

    def detect(self, save_img=False):
        """
        Method name : detect
        Called from main, Performs
        1. Detection : With yolov5 , checks if box is within segmented area(person detected is in the ground area)
                Once in 5 frames detector is run, rest 4 frames MOSSE tracker is run based on earlier detection
        2. Tracking  : Deep SORT algorithm is used, detections are passed to it and the trackers are updated with these new detections
        3. Output : oordinates of each player(track,trackid) passed to homography and then to unreal
        Paramters: The ones from command line are used, or the default ones if no args are given in command line

        """

        self.t2 = threading.Thread(target=self.show_frames)
        self.t2.setDaemon(True)
        self.t2.start()
        self.t3 = threading.Thread(target=self.run_tracker)
        self.t3.setDaemon(True)
        self.t3.start()

        self.udp_string = ""

        while True:
            if self.stop_threads:
                self.stop_output = True
                while self.stop_output:
                    continue
                self.stop_tracker = True
                while self.stop_tracker:
                    continue
                self.stop_stream = True
                while self.stop_stream:
                    continue
                self.stop_threads = False
                break

            self.frm_count += 1
            
            start_time = time.time()
            if self.frm_count == 0:

                if self.process_config.camera_model == 1:
                    self.vid = cv2.VideoCapture(self.source)
                    ret, frame = self.vid.read()
                    ret, frame = self.vid.read()

                elif self.process_config.camera_model == 2:
                    self.vid = Fvs(path=self.source).start()
                    frame = self.vid.read()
                    if frame is None:
                        print("CHECK VIDEO INPUT")
                        self.exithandler()
                    while frame.shape == (486, 720, 3):
                        frame = self.vid.read()
                        continue

                if frame is None:
                    continue

                cv2.imwrite("Settings/frame.jpg", frame)

                if self.process_config.activate_crop == 1:
                    frame = frame[self.process_config.crop_y1:self.process_config.crop_y2,
                                  self.process_config.crop_x1:self.process_config.crop_x2]
                self.FRAME_HT = frame.shape[0]
                self.FRAME_WD = frame.shape[1]
                self.qin.put(frame)

            elif self.frm_count == 1:
                self.t1 = threading.Thread(target=self.store_frames_read)
                self.t1.setDaemon(True)
                self.t1.start()

            loaded_start_time = time.time()
            new_player_list = []

            bboxes = []
            im0s_copy = self.qin.get()
            if self.save_frame_flag:
                self.save_frame_flag = False
                cv2.imwrite("Settings/frame.jpg", im0s_copy)
            if im0s_copy is None:
                continue
            if self.process_config.seg_mask is not None:
                while len(self.process_config.seg_mask.shape) != 3:
                    pass
                if im0s_copy.shape[0] == self.process_config.seg_mask.shape[0] and im0s_copy.shape[1] == self.process_config.seg_mask.shape[1]:
                    im0s_copy = np.bitwise_and(
                        im0s_copy, self.process_config.seg_mask)
                else:
                    self.process_config.seg_mask = cv2.resize(
                        self.process_config.seg_mask, (self.FRAME_WD, self.FRAME_HT))
                    im0s_copy = np.bitwise_and(
                        im0s_copy, self.process_config.seg_mask)

            start_time_tracker = time.time()
            bboxes = detector(im0s_copy)
            if (self.capture_frames is True) and (self.fn_count > 2) and (self.last_captured_frame + 50 < self.frm_count):
                self.last_captured_frame = copy.deepcopy(self.frm_count)
                print("saving")
                cv2.imwrite(
                    "Settings/Saved_frames/" +
                    str(int(time.time() * 100))+".jpg",
                    im0s_copy)
            det_time = time.time() - start_time_tracker
            self.qin_tracker.put([im0s_copy, bboxes])

    def exithandler(self):
        if self.process_config.camera_model == 1:
            self.vid.release()
        elif self.process_config.camera_model == 2:
            self.vid.stop()

        sys.exit()

    def run(self):
        self.detect()
