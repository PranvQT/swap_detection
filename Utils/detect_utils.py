
import ast
import copy
import json
import math
from threading import Lock
import socket
import os
import time

import cv2
import numpy as np
from numba import jit
from shapely.geometry import Point
# from skimage.measure import compare_ssim
import pymongo
from bson.json_util import dumps



mutex = Lock()

def delay_send(self,data,delay_time):
    print("hello",data)
    if data:
        if float(time.time()-data[0][0])>=delay_time:
            print("sending!")
            return data[0][1]
    else:
        # print("waiting!")
        return None
        
    
    
def batsman_position(ball, run, over_wide):
    last_ball = 6+over_wide
    if ball < last_ball and run % 2 == 1:
        return 1
    elif ball == last_ball and run % 2 == 0:
        return 1
    elif ball < last_ball and run % 2 == 0:
        return 0
    elif ball == last_ball and run % 2 == 1:
        return 0


def overend(ball, over_wide):
    last_ball = 6+over_wide
    if ball == last_ball:
        return 1
    elif ball < last_ball:
        return 0


def get_wt_scorefile_data(self, data, line, path, innings):
    data = {}
    _file = path
    # check modified time
    self.current_score_modiefied_time = os.path.getmtime(_file)
    if self.past_score_modiefied_time != self.current_score_modiefied_time:
        self.past_score_modiefied_time = copy.deepcopy(
            self.current_score_modiefied_time)
        with open(_file) as f:
            lines = f.readlines()
        # print(f"len {len(lines)}")
        if len(lines) > 0:
            if len(lines[-1]) >= 170:
                curr_innings = int(lines[-1][:3].strip())
                if curr_innings == innings:
                    try:
                        line = lines[-1]
                        data["Innings"] = int(line[:3].strip())
                        data["Batsman"] = line[8:33].strip()
                        data["Bowler"] = line[34:59].strip()
                        data["Over"] = int(line[60:63].strip())
                        data["Ball"] = int(line[65:67].strip())
                        data["Runs"] = int(line[74])
                        data["WagonX"] = int(line[82:85].strip())
                        data["WagonY"] = int(line[88:91].strip())
                        data["Wkt?"] = line[95].strip()
                        data["Bat_LH/RH"] = line[102].strip()
                        data["Shot1"] = line[109].strip()
                        data["Shot2"] = line[110].strip()
                        data["Hgt"] = line[115].strip()
                        data["LandX"] = line[119:122].strip()
                        data["LandY"] = line[125:128].strip()
                        data["Bowl_LH/RH"] = line[129].strip()
                        data["Other_Batsman"] = line[131:156].strip()
                        data["T/Ov"] = line[157:161].strip().lower()
                        data["6D"] = line[162:165].strip()
                        data["Shot_type"] = line[166:172].strip()
                        data["Spin_text"] = line[173:178].strip()
                    except Exception as e:
                        print(e)
                        line = ""
                        data = {}

                else:
                    print(f"Innings {innings} not present in scoring file")
        else:
            line = ""
    else:
        line = ""

    return data, line


def get_ae_scorefile_data(self, data, line, path, innings):
    data = {}
    if innings == 1:
        _file = path+"AELInteractiveHawkeye_1.TXT"
    elif innings == 2:
        _file = path+"AELInteractiveHawkeye_2.TXT"
    # check modified time
    # print(path+"AELInteractiveHawkeye_2.TXT")
    self.current_score_modiefied_time = os.path.getmtime(_file)
    if True:
        self.past_score_modiefied_time = copy.deepcopy(
            self.current_score_modiefied_time)

        with open(_file) as f:
            lines = f.readlines()
        if len(lines) > 52:
            if len(lines[-1]) >= 170:
                try:
                    line = lines[-1]
                    data["Innings"] = int(line[0:3].strip())
                    data["Batsman"] = line[8:33].strip()
                    data["Bowler"] = line[34:59].strip()
                    data["Over"] = int(line[60:63].strip())
                    data["Ball"] = int(line[65:67].strip())
                    data["Runs"] = int(line[74])
                    data["WagonX"] = int(line[82:85].strip())
                    data["WagonY"] = int(line[88:91].strip())
                    data["Wkt?"] = line[95].strip()
                    data["Bat_LH/RH"] = line[102].strip()
                    data["Shot1"] = line[109].strip()
                    data["Shot2"] = line[110].strip()
                    data["Hgt"] = line[115].strip()
                    data["LandX"] = line[119:122].strip()
                    data["LandY"] = line[125:128].strip()
                    data["Bowl_LH/RH"] = line[129].strip()
                    data["Other_Batsman"] = line[131:156].strip()
                    data["T/Ov"] = line[157:161].strip()
                    data["6D"] = line[162:165].strip()
                    data["Shot_type"] = line[166:172].strip()
                    data["Spin_text"] = line[173:178].strip()
                    # print(data, line)
                except Exception as e:
                    print(e, "error")
                    line = ""
                    data = {}
        else:
            line = ""
        #     print("inside else")
    else:
        line = ""

    return data, line ,lines


def process_lr_nf(data, old_over, over_wide, batman_data, nf_end):

    if data:
        if "wd" in data["T/Ov"].lower() or "nb" in data["T/Ov"].lower():
            over_wide += 1

        if old_over != data["Over"]:
            over_wide = 0

        old_over = data["Over"]
        # old_ball = data["Ball"]

        hand_flag = batsman_position(data["Ball"], data["Runs"], over_wide)
        over_flag = overend(data["Ball"], over_wide)

        if hand_flag == 1:
            next_bat_hand = batman_data[1][list(
                batman_data[0]).index(data["Other_Batsman"].lower())]
        else:
            next_bat_hand = data["Bat_LH/RH"]

        if next_bat_hand == "L":
            left_handed = True
        else:
            left_handed = False
        if over_flag == True:
            nf_end = 1 if nf_end == 0 else 0
            over_flag = False

    # print("nfend: ",nf_end)

    return left_handed, nf_end, old_over, over_wide


def command_f4(self):
    message = "F4"
    self.send_tcp_message(message)


def command_f6(self):
    message = "F6"
    self.send_tcp_message(message)


def command_f7(self):
    message = "F7"
    self.send_tcp_message(message)


def command_f8(self):
    message = "F8"
    self.send_tcp_message(message)


def command_f9(self):
    message = "F9"
    self.send_tcp_message(message)


def command_f(self):
    message = "F"
    self.send_tcp_message(message)


def command_w(self):
    message = "W"
    self.send_tcp_message(message)


def command_front_end_mode(self, flag):
    if flag:
        message = "START"
        self.send_tcp_message(message)
    else:
        message = "STOP"
        self.send_tcp_message(message)


def send_tcp_message(self, message):
    try:
        serverSock_tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # self.serverSock_tcp.settimeout(1)
        serverSock_tcp.connect(
            (self.process_config.viz_tcp_ip_address, self.process_config.viz_tcp_port))
        serverSock_tcp.send(message.encode('utf8'))
        serverSock_tcp.close()
    except:
        print("Caught exception socket.error ")


def outside_circle_calc(self, x, y):
    point = Point(x, y)
    # print(point.within(in_polygon))
    return point.within(self.process_config.in_polygon)


@jit(forceobj=True)
def calculate_homography(obj, cam_matrix, lens_distortion_flag=0, newcameramtx=[], lens_dist=0):
    # homo_start_time = time.time()
    width_current = int(obj[2]) - int(obj[0])
    height_current = int(obj[3]) - int(obj[1])
    x_current = obj[2]-width_current/2
    y_current = obj[3]
    if lens_distortion_flag == 1:
        test_in = np.array([[[x_current, y_current]]], dtype=np.float32)
        xy_undistorted = cv2.undistortPoints(
            test_in, newcameramtx, lens_dist, None, newcameramtx)
        x_current = int(xy_undistorted[0][0][0])
        y_current = int(xy_undistorted[0][0][1])
    points_map = np.array([[x_current, y_current]], dtype='float32')
    points_map = np.array([points_map])
    return cv2.perspectiveTransform(points_map, cam_matrix)


def distance_between(player1, player2, m_for_pix, homo_track):
    if((player1 in homo_track) and (player2 in homo_track)):
        pt1 = homo_track[player1]
        pt2 = homo_track[player2]
        pixel_distance = math.hypot((pt1[0] - pt2[0]), (pt1[1] - pt2[1]))
        return pixel_distance*m_for_pix
    return -1


def update_player_connect(self, player_connect_line_dict):
    self.player_connect_l__dict = player_connect_line_dict


# def image_comparison(img1, img2, compare_image):

#     h, w, c = compare_image.shape
#     img1 = cv2.resize(img1, (w, h))
#     img2 = cv2.resize(img2, (w, h))

#     grayA = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
#     grayB = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
#     grayCompare_image = cv2.cvtColor(compare_image, cv2.COLOR_BGR2GRAY)

#     (score1, diff1) = compare_ssim(grayA, grayCompare_image, full=True)
#     (score2, diff2) = compare_ssim(grayB, grayCompare_image, full=True)
#     diff1 = (diff1 * 255).astype("uint8")
#     diff2 = (diff2 * 255).astype("uint8")
#     if((score1 > 0.90) and (score2 < 0.850)):
#         #print("retrns 1")
#         return 2
#     elif((score2 > 0.90) and (score1 < 0.850)):
#         #print("returns 2")
#         return 1
#     else:
#         return 0


def update_lr_automation(self, lr_automated):
    self.left_right_automated = lr_automated


def livelock_clicked(self):
    self.livelock_status = not(self.livelock_status)
    self.livelock_ids_saved = False
    self.livelock_ids = {}
    self.livelock_ids_extra = {}


def insertDb_clicked(self):
    self.insertDb_status = not(self.insertDb_status)
    print(self.insertDb_status)

def save_db_colection_to_json(self,collection,collection_name):
    cursor = collection.find()
    list_cur = list(cursor)
    json_data = dumps(list_cur, indent = 2)
    with open(f'Settings/{collection_name}.json', 'w') as file:
	    file.write(json_data)




def air_clicked(self):
    if self.air_status:
        self.air_status = False
        self.air_init = True
    else:
        self.air_status = True


"""

Method name : generate_background

With the coordinates of boxes of players , using opencv's inpaint to generate the image of ground without players

Arguments:
img - frame
bboxes - coordinates of players
frm_count - current frame count
Return:
Returns the inpainted background frame

This background frame is used for generaatig mask

"""


# def generate_background(img, bboxes, frm_count):
#     mask = np.zeros((FRAME_HT, FRAME_WD), np.uint8)
#     for bbox in bboxes:
#         # print("box",bbox)
#         for i in range(int(bbox[1]), int(bbox[1] + bbox[3])):
#             for j in range(int(bbox[0]), int(bbox[0] + bbox[2])):
#                 if (i < FRAME_HT and j < FRAME_WD):
#                     mask[i, j] = 255
#     BG_Frame = cv2.inpaint(img, mask, 3, cv2.INPAINT_NS)
#     return BG_Frame
################DB-Functions###################################

def insert_mongo(collection, push_data):
    # print(push_data)
    try:
        collection.insert_one(push_data)
    except pymongo.errors.DuplicateKeyError:
        # print("no no")
        pass


def get_mongoData(self, over_value, ball_value):
    collection = self.collection

    # print("Getting from mongo !")
    outfile = "output.txt"

    file_txt = open(outfile, 'w+')

    # try:
    print("inside int")
    over_value = int(over_value)
    ball_value = int(ball_value)

    val = collection.find({
        "$and": [
            {
                "players.player_type": 0
            },
            {
                "$and": [
                    {
                        "over": over_value
                    },
                    {
                        "ball": ball_value
                    }
                ]
            }
        ]
    }, {'_id': False})
    data = list(val)
    print("length:", len(data))
    # print(data)

    for data1 in data:
        temp = data1
        udp_string = ""
        count = 0
        for players in temp["players"]:
            if(players['player_type'] == 0):
                count += 1
                homo_track = players['homo_track']
                if(udp_string == ""):
                    udp_string += "PLAYER" + \
                        str(players['identities']) + "|"
                else:
                    udp_string += "," + "PLAYER" + \
                        str(players['identities']) + "|"

                udp_string += str(homo_track[0]) + "_"
                udp_string += str((homo_track[1]) * -1) + "_"
                udp_string += str(int(players['player_type']))

        file_txt.write(udp_string)
        # print(udp_string)
        file_txt.write("\n")
        # print(udp_string)
        # print("\n")
    # except Exception as e:
    #     print("not given a proper integer value")
    #     print(e)
    file_txt.close
    print("saved to a text file")


############################################################################################################################################

def batsmen_pos_flip(self, value):
    self.left_handed = (value == 0)


def change_player_type(self, player_id, player_type):
    self.sort_tracker.change_playertype(player_id, player_type)


def highlight_player(self, player_id):
    if(str(player_id) in self.highlights_list):
        self.highlights_list.remove(str(player_id))
    else:
        self.highlights_list.append(str(player_id))

    self.sort_tracker.highlightPlayer(player_id)


def reset_handler_flager(self):
    self.reset_handler_flag = True


def reset_Highlight_flager(self):
    self.reset_Highlight_flag = True

def each_slice(self,iterable, n=2):
    if n < 2: n = 1
    i, size = 0, len(iterable)
    while i < size-n+1:
        yield iterable[i:i+n]
        i += n

def reset_Highlight(self):
    if self.livelock_ids_saved:
        for id in self.batsmen_ids:
            if id in self.livelock_ids.keys():
                del self.livelock_ids[id]
            elif id in self.livelock_ids_extra.keys():
                del self.livelock_ids_extra[id]
    self.dummy_connect_id = -1
    self.dummy_player_cache = [-1,-1]
    self.dummy_player_falg = False
    self.player_connect_l__dict = {}
    self.highlights_list = []
    self.gap_ids = []
    self.ingap_ids= []
    self.sort_tracker.dehighlight()
    self.reset_FN_highlights()
    self.multi_gap_ids = []


def highlight_fn(self, player_id):
    # global false_negatives, highlight_fns, false_negatives_slipFielders, false_negatives_outside_frame_z, false_negatives_outside_frame_a
    if(player_id in self.false_negatives.keys()):
        if(player_id not in self.highlight_fns):
            self.highlight_fns.append(player_id)
            self.highlights_list.append(str(player_id))
        else:
            self.highlights_list.remove(str(player_id))
            self.highlight_fns.remove(player_id)
    elif(player_id in self.false_negatives_slipFielders.keys()):
        if(player_id not in self.highlight_fns):
            self.highlight_fns.append(player_id)
            self.highlights_list.append(str(player_id))
        else:
            self.highlights_list.remove(str(player_id))
            self.highlight_fns.remove(player_id)
    elif(player_id in self.false_negatives_outside_frame_z.keys()):
        if(player_id not in self.highlight_fns):
            self.highlight_fns.append(player_id)
            self.highlights_list.append(str(player_id))
        else:
            self.highlights_list.remove(str(player_id))
            self.highlight_fns.remove(player_id)
    elif(player_id in self.false_negatives_outside_frame_a.keys()):
        if(player_id not in self.highlight_fns):
            self.highlight_fns.append(player_id)
            self.highlights_list.append(str(player_id))
        else:
            self.highlights_list.remove(str(player_id))
            self.highlight_fns.remove(player_id)
    elif(player_id in self.false_negatives_outside_frame_u.keys()):
        if(player_id not in self.highlight_fns):
            self.highlight_fns.append(player_id)
            self.highlights_list.append(str(player_id))
        else:
            self.highlights_list.remove(str(player_id))
            self.highlight_fns.remove(player_id)


def highlight_player_streak(self, current_mouse_pos, img_size, frame_size):
    player_id = self.find_clicked_player_id(
        current_mouse_pos[0], current_mouse_pos[1], img_size, frame_size)
    if player_id not in [-1, None]:
        self.sort_tracker.highlightPlayerStreak(player_id)


def reset_HighlightStreak(self):
    self.sort_tracker.dehighlight_streak()


def calculate_distance(self, player1, player2):
    temp_list = [self.frm_count, player1, player2]
    self.distance_list.append(temp_list)


def remove_distance(self, index):
    # print(self.livelock_ids_extra.keys())
    # print(self.livelock_ids.keys())

    self.wk_pt = []

    if(index != -1):
        if self.distance_list != []:
            if((str(self.distance_list[index][0]) in self.batsmen_ids) and (str(self.distance_list[index][0]) in self.livelock_ids)):
                del self.livelock_ids[str(self.distance_list[index][0])]
                del self.livelock_ids_extra[str(self.distance_list[index][0])]

            elif((str(self.distance_list[index][1]) in self.batsmen_ids) and (str(self.distance_list[index][1]) in self.livelock_ids)):
                del self.livelock_ids[str(self.distance_list[index][1])]
                del self.livelock_ids_extra[str(self.distance_list[index][1])]

            self.distance_list.pop(index)
    else:
        for index in range(len(self.distance_list)):
            if((str(self.distance_list[index][0]) in self.batsmen_ids) and (str(self.distance_list[index][0]) in self.livelock_ids)):
                del self.livelock_ids[str(self.distance_list[index][0])]
                del self.livelock_ids_extra[str(self.distance_list[index][0])]
            elif((str(self.distance_list[index][1]) in self.batsmen_ids) and (str(self.distance_list[index][1]) in self.livelock_ids)):
                del self.livelock_ids[str(self.distance_list[index][1])]
                del self.livelock_ids_extra[str(self.distance_list[index][1])]
        self.distance_list = []
    # print(self.livelock_ids_extra.keys())
    self.reset_batsmen()


def activate_distance(self, player_id):
    self.activate_distance_id = player_id


def create_FN_player(self, mouse_pos, img_size, frame_size):
    x = mouse_pos[0]
    y = mouse_pos[1]
    resized_x, resized_y = convert_coordinates_image_label_size(
        x, y, img_size, frame_size)
    # ##print("RESIZING COORDS", x,y, resized_x,resized_y)
    for i in range(1, 11):
        if((i + 100) not in self.false_negatives.keys()):
            mutex.acquire()
            key = i + 100
            self.false_negatives[i + 100] = [int(resized_x - int(self.FN_WD/2)),
                                             int(resized_y - int(self.FN_HT)), self.FN_WD, self.FN_HT]
            obj = [int(self.false_negatives[key][0])-10, int(self.false_negatives[key][1])-10, int(self.false_negatives[key]
                                                                                                   [2]+self.false_negatives[key][0])+10, int(self.false_negatives[key][3]+self.false_negatives[key][1])+10]
            if self.process_config.lens_distortion_flag == 1:
                homo_pt = calculate_homography(obj=obj, cam_matrix=self.process_config.cam_matrix, lens_distortion_flag=self.process_config.lens_distortion_flag,
                                               newcameramtx=self.process_config.newcameramtx, lens_dist=self.process_config.lens_dist)
            else:
                homo_pt = calculate_homography(
                    obj=obj, cam_matrix=self.process_config.cam_matrix)

            homo_pt = (int(homo_pt[0][0][0]), int(homo_pt[0][0][1]))
            x1 = homo_pt[0]
            y1 = homo_pt[1]
            self.homo_track[key] = [x1, y1]
            # ##print("FN NEWID created", i, self.false_negatives[i])
            mutex.release()

            break


def remove_FN_player(self, player_id):
    mutex.acquire()
    del self.false_negatives[player_id]
    mutex.release()


def remove_FN_player_outside_frame_z(self, player_id):
    mutex.acquire()
    if(player_id in self.false_negatives_outside_frame_z.keys()):
        del self.false_negatives_outside_frame_z[player_id]
    mutex.release()


def create_FN_player_outside_frame_z(self, mouse_pos, img_size, frame_size):
    x = mouse_pos[0]
    y = mouse_pos[1]
    resized_x, resized_y = convert_coordinates_image_label_size(
        x, y, img_size, frame_size)

    key = 115
    if((key) not in self.false_negatives_outside_frame_z.keys()):
        mutex.acquire()

        self.false_negatives_outside_frame_z[key] = [int(resized_x - int(self.FN_WD/2)),
                                                     int(resized_y - int(self.FN_HT)), self.FN_WD, self.FN_HT]
        obj = [int(self.false_negatives_outside_frame_z[key][0])-10, int(self.false_negatives_outside_frame_z[key][1])-10, int(self.false_negatives_outside_frame_z[key]
                                                                                                                               [2]+self.false_negatives_outside_frame_z[key][0])+10, int(self.false_negatives_outside_frame_z[key][3]+self.false_negatives_outside_frame_z[key][1])+10]
        homo_pt = calculate_homography(
            obj=obj, cam_matrix=self.process_config.cam_matrix)
        homo_pt = (int(homo_pt[0][0][0]), int(homo_pt[0][0][1]))
        x1 = homo_pt[0]
        y1 = homo_pt[1]
        self.homo_track[key] = [x1, y1]
        mutex.release()


def remove_FN_player_outside_frame_a(self, player_id):
    mutex.acquire()
    if(player_id in self.false_negatives_outside_frame_a.keys()):
        del self.false_negatives_outside_frame_a[player_id]
    mutex.release()


def create_FN_player_outside_frame_a(self, mouse_pos, img_size, frame_size):
    x = mouse_pos[0]
    y = mouse_pos[1]
    resized_x, resized_y = convert_coordinates_image_label_size(
        x, y, img_size, frame_size)
    # ##print("RESIZING COORDS", x,y, resized_x,resized_y)
    key = 116

    if((key) not in self.false_negatives_outside_frame_a.keys()):
        mutex.acquire()

        self.false_negatives_outside_frame_a[key] = [
            int(resized_x - int(self.FN_WD/2)), int(resized_y - int(self.FN_HT)), self.FN_WD, self.FN_HT]
        obj = [int(self.false_negatives_outside_frame_a[key][0])-10, int(self.false_negatives_outside_frame_a[key][1])-10, int(self.false_negatives_outside_frame_a[key]
                                                                                                                               [2]+self.false_negatives_outside_frame_a[key][0])+10, int(self.false_negatives_outside_frame_a[key][3]+self.false_negatives_outside_frame_a[key][1])+10]
        homo_pt = calculate_homography(
            obj=obj, cam_matrix=self.process_config.cam_matrix)
        homo_pt = (int(homo_pt[0][0][0]), int(homo_pt[0][0][1]))
        x1 = homo_pt[0]
        y1 = homo_pt[1]
        self.homo_track[key] = [x1, y1]
        # ##print("FN NEWID created", i, false_negatives[i])
        mutex.release()


def remove_FN_player_outside_frame_u(self, player_id):
    mutex.acquire()
    if(player_id in self.false_negatives_outside_frame_u.keys()):
        del self.false_negatives_outside_frame_u[player_id]
    mutex.release()


def create_FN_player_outside_frame_u(self, mouse_pos, img_size, frame_size):
    x = mouse_pos[0]
    y = mouse_pos[1]
    resized_x, resized_y = convert_coordinates_image_label_size(
        x, y, img_size, frame_size)
    # ##print("RESIZING COORDS", x,y, resized_x,resized_y)

    # for i in range(1, 6):
    # if((150) not in self.false_negatives_outside_frame_u.keys()):
    mutex.acquire()
    key = 150
    self.false_negatives_outside_frame_u[key] = [
        int(resized_x - int(self.FN_WD/2)), int(resized_y - int(self.FN_HT)), self.FN_WD, self.FN_HT]
    obj = [int(self.false_negatives_outside_frame_u[key][0])-10, int(self.false_negatives_outside_frame_u[key][1])-10, int(self.false_negatives_outside_frame_u[key]
                                                                                                                           [2]+self.false_negatives_outside_frame_u[key][0])+10, int(self.false_negatives_outside_frame_u[key][3]+self.false_negatives_outside_frame_u[key][1])+10]
    homo_pt = calculate_homography(
        obj=obj, cam_matrix=self.process_config.cam_matrix)
    homo_pt = (int(homo_pt[0][0][0]), int(homo_pt[0][0][1]))
    x1 = homo_pt[0]
    y1 = homo_pt[1]
    self.homo_track[key] = [x1, y1]
    # ##print("FN NEWID created", i, false_negatives[i])
    mutex.release()
    # break


# def remove_batsmen_player(self, player_id):
#     mutex.acquire()

#     del self.false_batsmen[player_id]
#     if str(player_id) in self.batsmen_ids:
#         self.batsmen_ids.remove(str(player_id))
#     if str(player_id) in self.livelock_ids:
#         del self.livelock_ids[str(player_id)]
#         del self.livelock_ids_extra[str(player_id)]
#     mutex.release()

def reset_FN_flager(self):
    self.reset_FN_flag = True


def reset_FN(self):
    mutex.acquire()
    self.reset_FN_highlights()
    for i in range(len(self.distance_list)):
        for key in self.false_negatives.keys():
            if key in [self.distance_list[i][1], self.distance_list[i][2]]:
                self.distance_list.pop(i)
                break

        for key in self.false_negatives_slipFielders.keys():
            if key in [self.distance_list[i][1], self.distance_list[i][2]]:
                self.distance_list.pop(i)
                break

        for key in self.false_negatives_mark_point_p.keys():
            if key in [self.distance_list[i][1], self.distance_list[i][2]]:
                self.distance_list.pop(i)
                break

        for key in self.false_negatives_mark_point_o.keys():
            if key == self.distance_list[i][1] or key == self.distance_list[i][2]:
                self.distance_list.pop(i)
                break

        for key in self.false_negatives_outside_frame_z.keys():
            if key == self.distance_list[i][1] or key == self.distance_list[i][2]:
                self.distance_list.pop(i)
                break

        for key in self.false_negatives_outside_frame_a.keys():
            if key == self.distance_list[i][1] or key == self.distance_list[i][2]:
                self.distance_list.pop(i)
                break

        for key in self.false_negatives_outside_frame_u.keys():
            if key == self.distance_list[i][1] or key == self.distance_list[i][2]:
                self.distance_list.pop(i)
                break

    self.false_negatives = {}
    self.false_negatives_slipFielders = {}
    self.false_negatives_mark_point_p = {}
    self.false_negatives_mark_point_o = {}
    self.false_negatives_outside_frame_z = {}
    self.false_negatives_outside_frame_a = {}
    self.false_negatives_outside_frame_u = {}
    self.highlight_fns = []
    mutex.release()


def reset_FN_highlights(self):
    self.highlight_fns = []
    self.false_negatives_mark_point_o = {}
    self.dummy_connect_id = -1
    self.dummy_player_id = -1
    for i in range(len(self.distance_list)):
        if self.distance_list[i][1] == 151 or self.distance_list[i][2] == 151:
            self.distance_list.pop(i)
            break


def create_FN_player_slipFielders(self, mouse_pos, img_size, frame_size):
    x = mouse_pos[0]
    y = mouse_pos[1]
    resized_x, resized_y = convert_coordinates_image_label_size(
        x, y, img_size, frame_size)
    # ##print("RESIZING COORDS", x,y, resized_x,resized_y)
    for i in range(1, 11):
        if((i + 115) not in self.false_negatives_slipFielders.keys()):
            mutex.acquire()

            self.false_negatives_slipFielders[i + 115] = [
                int(resized_x - int(self.FN_WD/2)), int(resized_y - int(self.FN_HT)), self.FN_WD, self.FN_HT]
            key = i + 115
            obj = [int(self.false_negatives_slipFielders[key][0])-10, int(self.false_negatives_slipFielders[key][1])-10, int(self.false_negatives_slipFielders[key]
                                                                                                                             [2]+self.false_negatives_slipFielders[key][0])+10, int(self.false_negatives_slipFielders[key][3]+self.false_negatives_slipFielders[key][1])+10]
            if self.process_config.lens_distortion_flag == 1:
                homo_pt = calculate_homography(obj=obj, cam_matrix=self.process_config.cam_matrix, lens_distortion_flag=self.process_config.lens_distortion_flag,
                                               newcameramtx=self.process_config.newcameramtx, lens_dist=self.process_config.lens_dist)
            else:
                homo_pt = calculate_homography(
                    obj=obj, cam_matrix=self.process_config.cam_matrix)
            homo_pt = (int(homo_pt[0][0][0]), int(homo_pt[0][0][1]))
            x1 = homo_pt[0]
            y1 = homo_pt[1]
            self.homo_track[key] = [x1, y1]
            # ##print("FN NEWID created", i, self.false_negatives[i])
            mutex.release()
            break


def create_FN_player_mark_point_p(self, mouse_pos, img_size, frame_size):
    x = mouse_pos[0]
    y = mouse_pos[1]
    resized_x, resized_y = convert_coordinates_image_label_size(
        x, y, img_size, frame_size)
    # ##print("RESIZING COORDS", x,y, resized_x,resized_y)
    for i in range(1, 11):
        if((i + 130) not in self.false_negatives_mark_point_p.keys()):
            mutex.acquire()

            self.false_negatives_mark_point_p[i + 130] = [
                int(resized_x - int(self.FN_WD/2)), int(resized_y - int(self.FN_HT)), self.FN_WD, self.FN_HT]
            key = i + 130
            obj = [int(self.false_negatives_mark_point_p[key][0])-10, int(self.false_negatives_mark_point_p[key][1])-10, int(self.false_negatives_mark_point_p[key]
                                                                                                                             [2]+self.false_negatives_mark_point_p[key][0])+10, int(self.false_negatives_mark_point_p[key][3]+self.false_negatives_mark_point_p[key][1])+10]
            if self.process_config.lens_distortion_flag == 1:
                homo_pt = calculate_homography(obj=obj, cam_matrix=self.process_config.cam_matrix, lens_distortion_flag=self.process_config.lens_distortion_flag,
                                               newcameramtx=self.process_config.newcameramtx, lens_dist=self.process_config.lens_dist)
            else:
                homo_pt = calculate_homography(
                    obj=obj, cam_matrix=self.process_config.cam_matrix)
            homo_pt = (int(homo_pt[0][0][0]), int(homo_pt[0][0][1]))
            x1 = homo_pt[0]
            y1 = homo_pt[1]
            self.homo_track[key] = [x1, y1]
            # ##print("FN NEWID created", i, self.false_negatives[i])
            mutex.release()
            break


def create_FN_player_mark_point_o(self, mouse_pos, img_size, frame_size):
    # create a dummy point
    x = mouse_pos[0]
    y = mouse_pos[1]
    resized_x, resized_y = convert_coordinates_image_label_size(
        x, y, img_size, frame_size)
    # ##print("RESIZING COORDS", x,y, resized_x,resized_y)
    key = 151
    self.false_negatives_mark_point_p[key] = [
        int(resized_x - int(self.FN_WD/2)), int(resized_y - int(self.FN_HT)), self.FN_WD, self.FN_HT]
    obj = [int(self.false_negatives_mark_point_p[key][0])-10, int(self.false_negatives_mark_point_p[key][1])-10, int(self.false_negatives_mark_point_p[key]
                                                                                                                     [2]+self.false_negatives_mark_point_p[key][0])+10, int(self.false_negatives_mark_point_p[key][3]+self.false_negatives_mark_point_p[key][1])+10]
    if self.process_config.lens_distortion_flag == 1:
        homo_pt = calculate_homography(obj=obj, cam_matrix=self.process_config.cam_matrix, lens_distortion_flag=self.process_config.lens_distortion_flag,
                                       newcameramtx=self.process_config.newcameramtx, lens_dist=self.process_config.lens_dist)
    else:
        homo_pt = calculate_homography(
            obj=obj, cam_matrix=self.process_config.cam_matrix)
    homo_pt = (int(homo_pt[0][0][0]), int(homo_pt[0][0][1]))
    x1 = homo_pt[0]
    y1 = homo_pt[1]
    self.homo_track[key] = [x1, y1]
    # mark up the wkt pt after detecting near or far end points
    if self.flip_field_plot == 0:  # near-end
        x, y = convert_coordinates_image_label_size(
            self.process_config.near_end_stump[0], self.process_config.near_end_stump[1], frame_size, img_size)
        self.homo_track[124] = [
            self.process_config.near_end_stump[0], self.process_config.near_end_stump[1]]
    if self.flip_field_plot == 1:  # far-end
        x, y = convert_coordinates_image_label_size(
            self.process_config.far_end_stump[0], self.process_config.far_end_stump[1], frame_size, img_size)
        self.homo_track[124] = [
            self.process_config.far_end_stump[0], self.process_config.far_end_stump[1]]
    # get ui coords from real image coords
    self.wk_pt = [int(x - int(self.FN_WD/2)), int(y - int(self.FN_HT)),
                  int(x + int(self.FN_WD)), int(y + int(self.FN_HT))]
    print("wkpt", self.wk_pt)
    ###############
    # get distance between wk_pt and O


def create_FN_player_mark_point_b(self, player_id, img_size, frame_size):
    print("bbbclicked")
    # mark up the wkt pt after detecting near or far end points
    if self.flip_field_plot == 0:  # near-end
        x, y = convert_coordinates_image_label_size(
            self.process_config.near_end_stump[0], self.process_config.near_end_stump[1], frame_size, img_size)
        self.homo_track[124] = [
            self.process_config.near_end_stump[0], self.process_config.near_end_stump[1]]
    if self.flip_field_plot == 1:  # far-end
        x, y = convert_coordinates_image_label_size(
            self.process_config.far_end_stump[0], self.process_config.far_end_stump[1], frame_size, img_size)
        self.homo_track[124] = [
            self.process_config.far_end_stump[0], self.process_config.far_end_stump[1]]
    # get ui coords from real image coords
    self.wk_pt = [int(x - int(self.FN_WD/2)), int(y - int(self.FN_HT)),
                  int(x + int(self.FN_WD)), int(y + int(self.FN_HT))]
    print("wkpt", self.wk_pt)
    ###############
    # get distance between wk_pt and O


def remove_FN_player_markpoint_o(self, player_id):

    mutex.acquire()
    if(player_id in self.false_negatives_mark_point_o.keys()):
        del self.false_negatives_mark_point_o[player_id]
        self.wk_pt = []
    mutex.release()


def remove_FN_player_markpoint_p(self, player_id):
    mutex.acquire()
    if(player_id in self.false_negatives_mark_point_p.keys()):
        del self.false_negatives_mark_point_p[player_id]
    mutex.release()


def remove_FN_player_slipFielders(self, player_id):
    mutex.acquire()
    if(player_id in self.false_negatives_slipFielders.keys()):
        del self.false_negatives_slipFielders[player_id]
    mutex.release()


def save_frame(self):
    self.save_frame_flag = True


def reset_batsmen(self):
    self.batsmen_ids = []
    self.false_batsmen = {}
    self.batsman_no = 0


def reset_bowler_flager(self):
    self.reset_bowler_flag = True


def reset_bowler(self):
    if self.bowler_id != -1:
        np_ids = np.array(self.identities)
        if self.bowler_id not in [-1, None] and self.bowler_id in np_ids:
            p_idx = np.where(np_ids == self.bowler_id)[0][0]
            if self.player_types[p_idx] == 9:
                self.change_player_type(self.bowler_id, 0)
        self.bowler_id = -1


def activate_count_outside_player(self):
    self.outside_circle = not(self.outside_circle)


def set_gap_ids(self, player_id):
    if len(self.gap_ids) >= 2:
        self.gap_ids = []
    self.gap_ids.append(player_id)
    
def set_multi_gap_ids(self, player_id):
    self.multi_gap_ids.append(player_id)

def set_ingap_ids(self, player_id):
    if len(self.ingap_ids) >= 3:
        self.ingap_ids = []
    self.ingap_ids.append(player_id)

def set_fielder_name_TA(self, f_name):
    self.fielder_dict[str(self.detect_fielder_id_TA)] = f_name
    self.detect_fielder_id_TA = -1


def set_fielder_name_TB(self, f_name):
    self.fielder_dict[str(self.detect_fielder_id_TB)] = f_name
    self.detect_fielder_id_TB = -1


def set_fielder_name_PO(self, f_name):
    # print("position name:",f_name)
    self.fielder_dict_PO[str(self.detect_fielder_id_PO)] = f_name
    self.detect_fielder_id_PO = -1


def set_clicked_fielder_TA(self, player_id):
    if self.detect_fielder_id_TA == str(player_id):
        self.detect_fielder_id_TA = -1
    else:
        self.detect_fielder_id_TA = str(player_id)
        self.detect_fielder_id_TB = -1
        # self.detect_fielder_id_PO = -1
    return self.detect_fielder_id_TA


def set_clicked_fielder_TB(self, player_id):
    if self.detect_fielder_id_TB == str(player_id):
        self.detect_fielder_id_TB = -1
    else:
        self.detect_fielder_id_TB = str(player_id)
        self.detect_fielder_id_TA = -1
        # self.detect_fielder_id_PO = -1
    return self.detect_fielder_id_TB


def set_clicked_fielder_PO(self, player_id):
    if self.detect_fielder_id_PO == str(player_id):
        self.detect_fielder_id_PO = -1
    else:
        self.detect_fielder_id_PO = str(player_id)
        # self.detect_fielder_id_TA = -1
        # self.detect_fielder_id_TB = -1
    return self.detect_fielder_id_PO


def set_umpire_id(self, player_id):
    if((len(self.umpire_id) > 0) and player_id == self.umpire_id[0]):
        del self.fielder_dict_PO[str(player_id)]
        self.umpire_id.remove(player_id)
        self.sort_tracker.umpire_id = self.umpire_id
    elif((len(self.umpire_id) > 1) and player_id == self.umpire_id[1]):
        del self.fielder_dict_PO[str(player_id)]
        self.umpire_id.remove(player_id)
        self.sort_tracker.umpire_id = self.umpire_id
    else:
        if(len(self.umpire_id) == 2):
            del self.fielder_dict_PO[str(self.umpire_id[0])]
            del self.fielder_dict_PO[str(self.umpire_id[1])]
            self.umpire_id = []
        self.umpire_id.append(player_id)
        self.sort_tracker.change_playertype(player_id, 2)
        self.sort_tracker.umpire_id = self.umpire_id

def set_batsmen_id(self, player_id):
    if((len(self.batsmen_ids_automated) > 0) and player_id == self.batsmen_ids_automated[0]):
        del self.fielder_dict_PO[str(player_id)]
        self.batsmen_ids_automated.remove(player_id)
        self.sort_tracker.batsmen_ids = self.batsmen_ids_automated 
    elif((len(self.batsmen_ids_automated) > 1) and player_id == self.batsmen_ids_automated[1]):
        del self.fielder_dict_PO[str(player_id)]
        self.batsmen_ids_automated.remove(player_id)
        self.sort_tracker.batsmen_ids = self.batsmen_ids_automated 
    else:
        if(len(self.batsmen_ids_automated) == 2):
            del self.fielder_dict_PO[str(self.batsmen_ids_automated[0])]
            del self.fielder_dict_PO[str(self.batsmen_ids_automated[1])]
            self.batsmen_ids_automated = []
        self.batsmen_ids_automated.append(int(player_id))
        self.sort_tracker.change_playertype(player_id, 1)
        self.sort_tracker.batsmen_ids = self.batsmen_ids_automated
        
def reset_umpire_flager(self):
    self.reset_umpire_flag = True


def reset_umpire(self):
    self.umpire_id = []
    self.sort_tracker.umpire_id = self.umpire_id


def reset_naming_flager(self):
    self.reset_naming_flag = True


def reset_naming(self):
    self.fielder_dict = {}
    self.detect_fielder_id_TA = -1
    self.detect_fielder_id_TB = -1
    self.detect_fielder_id_PO = -1


def clear_naming_coloring(self):
    self.detect_fielder_id_TA = -1
    self.detect_fielder_id_TB = -1
    self.detect_fielder_id_PO = -1


def find_clicked_player_id(self, pointx, pointy, img_size, frame_size):
    resized_x, resized_y = convert_coordinates_image_label_size(
        pointx, pointy, img_size, frame_size)

    for key in self.current_coords.keys():
        if key not in self.false_negatives.keys() and key not in self.false_negatives_slipFielders.keys():
            [x1, y1, x2, y2] = self.current_coords[key]
            if(((resized_x >= x1) and (resized_x <= x2)) and ((resized_y >= y1) and (resized_y <= y2))):
                return key
    return -1


# def find_clicked_batsman_player_id(self, pointx, pointy, img_size, frame_size):

#     resized_x, resized_y = convert_coordinates_image_label_size(
#         pointx, pointy, img_size, frame_size)

#     for key in self.false_batsmen.keys():
#         [x1, y1, w, h] = self.false_batsmen[key]
#         if(((resized_x >= x1) and (resized_x <= (x1+w))) and ((resized_y >= y1) and (resized_y <= (y1+h)))):
#             return key
#     return -1


# def create_batsmen_player(self, mouse_pos, img_size, frame_size):
#     # sourcery skip: extract-method
#     x = mouse_pos[0]
#     y = mouse_pos[1]
#     resized_x, resized_y = convert_coordinates_image_label_size(
#         x, y, img_size, frame_size)
#     if((self.batsman_no + 123) not in self.false_batsmen.keys()):
#         self.batsman_no += 1
#         mutex.acquire()
#         key = self.batsman_no + 123
#         self.false_batsmen[key] = [int(resized_x - int(self.FN_WD/2)),
#                                    int(resized_y - int(self.FN_HT/2)), self.FN_WD, self.FN_HT]
#         obj = [int(self.false_batsmen[key][0])+10, int(self.false_batsmen[key][1])+10, int(self.false_batsmen[key]
#                                                                                            [2]+self.false_batsmen[key][0])-10, int(self.false_batsmen[key][3]+self.false_batsmen[key][1])-10]
#         if self.process_config.lens_distortion_flag == 1:
#             homo_pt = calculate_homography(obj=obj, cam_matrix=self.process_config.cam_matrix, lens_distortion_flag=self.process_config.lens_distortion_flag,
#                                            newcameramtx=self.process_config.newcameramtx, lens_dist=self.process_config.lens_dist)
#         else:
#             homo_pt = calculate_homography(
#                 obj=obj, cam_matrix=self.process_config.cam_matrix)
#         homo_pt = (int(homo_pt[0][0][0]), int(homo_pt[0][0][1]))
#         x1 = homo_pt[0]
#         y1 = homo_pt[1]
#         self.homo_track[key] = [x1, y1]
#         self.batsmen_ids.append(str(key))
#         batsman_ids_copy = copy.deepcopy(self.false_batsmen)
#         mutex.release()


def find_clicked_FN_player_id(self, pointx, pointy, img_size, frame_size):
    resized_x, resized_y = convert_coordinates_image_label_size(
        pointx, pointy, img_size, frame_size)

    for key in self.false_negatives.keys():
        [x1, y1, w, h] = self.false_negatives[key]
        if(((resized_x >= x1) and (resized_x <= (x1+w))) and ((resized_y >= y1) and (resized_y <= (y1+h)))):
            return key
    return -1


def find_clicked_FN_player_id_slipFielders(self, pointx, pointy, img_size, frame_size):

    resized_x, resized_y = convert_coordinates_image_label_size(
        pointx, pointy, img_size, frame_size)
    for key in self.false_negatives_slipFielders.keys():
        [x1, y1, w, h] = self.false_negatives_slipFielders[key]
        if(((resized_x >= x1) and (resized_x <= (x1+w))) and ((resized_y >= y1) and (resized_y <= (y1+h)))):
            return key
    return -1


def find_clicked_FN_player_id_outside_frame_z(self, pointx, pointy, img_size, frame_size):
    resized_x, resized_y = convert_coordinates_image_label_size(
        pointx, pointy, img_size, frame_size)
    for key in self.false_negatives_outside_frame_z.keys():
        [x1, y1, w, h] = self.false_negatives_outside_frame_z[key]
        if(((resized_x >= x1) and (resized_x <= (x1+w))) and ((resized_y >= y1) and (resized_y <= (y1+h)))):
            return key
    return -1


def find_clicked_FN_player_id_outside_frame_a(self, pointx, pointy, img_size, frame_size):
    resized_x, resized_y = convert_coordinates_image_label_size(
        pointx, pointy, img_size, frame_size)
    for key in self.false_negatives_outside_frame_a.keys():
        [x1, y1, w, h] = self.false_negatives_outside_frame_a[key]
        if(((resized_x >= x1) and (resized_x <= (x1+w))) and ((resized_y >= y1) and (resized_y <= (y1+h)))):
            return key
    return -1


def find_clicked_FN_player_id_outside_frame_u(self, pointx, pointy, img_size, frame_size):
    resized_x, resized_y = convert_coordinates_image_label_size(
        pointx, pointy, img_size, frame_size)
    for key in self.false_negatives_outside_frame_u.keys():
        [x1, y1, w, h] = self.false_negatives_outside_frame_u[key]
        if(((resized_x >= x1) and (resized_x <= (x1+w))) and ((resized_y >= y1) and (resized_y <= (y1+h)))):
            return key
    return -1


def find_clicked_FN_player_id_markpoint_p(self, pointx, pointy, img_size, frame_size):

    resized_x, resized_y = convert_coordinates_image_label_size(
        pointx, pointy, img_size, frame_size)
    for key in self.false_negatives_mark_point_p.keys():
        [x1, y1, w, h] = self.false_negatives_mark_point_p[key]
        if(((resized_x >= x1) and (resized_x <= (x1+w))) and ((resized_y >= y1) and (resized_y <= (y1+h)))):
            return key
    return -1


def find_clicked_FN_player_id_markpoint_o(self, pointx, pointy, img_size, frame_size):

    resized_x, resized_y = convert_coordinates_image_label_size(
        pointx, pointy, img_size, frame_size)
    for key in self.false_negatives_mark_point_p.keys():
        [x1, y1, w, h] = self.false_negatives_mark_point_p[key]
        if(((resized_x >= x1) and (resized_x <= (x1+w))) and ((resized_y >= y1) and (resized_y <= (y1+h)))):
            return key
    return -1


def convert_coordinates_image_label_size(x, y, img_size, frame_size):
    resized_x = int((x * frame_size[0])/img_size[0])
    resized_y = int((y * frame_size[1])/img_size[1])
    # print("resized",resized_x,resized_y)
    return resized_x, resized_y


def convert_coordinates_image_resized(x, y, z, img_size, frame_size):
    resized_x = int((x * img_size[0])/frame_size[0])
    resized_y = int((y * img_size[1])/frame_size[1])
    resized_z = int((z * img_size[1])/frame_size[1])
    return resized_x, resized_y, resized_z

def get_relative_location(detected_id,flip_field_plot,left_handed,far_end_stump,near_end_stump,homo_track):
    if flip_field_plot == 1:
        local_centre = far_end_stump
        opposite_end = near_end_stump
        if left_handed:
            P1_X = -(homo_track[detected_id][0] - local_centre[0])  
            P1_Y =  (-local_centre[1] +  homo_track[detected_id][1])

            P2_X = -(homo_track[detected_id][0] - opposite_end[0])  
            P2_Y =  (-opposite_end[1] +  homo_track[detected_id][1])
        else:
            P1_X = -(local_centre[0] - homo_track[detected_id][0])  
            P1_Y =  -local_centre[1] +  homo_track[detected_id][1]

            P2_X = -(opposite_end[0] - homo_track[detected_id][0])  
            P2_Y =  -opposite_end[1] +  homo_track[detected_id][1]


    else:
        local_centre = near_end_stump
        opposite_end = far_end_stump
        if left_handed:
            P1_X = homo_track[detected_id][0] - local_centre[0]  
            P1_Y =  -(-local_centre[1] +  homo_track[detected_id][1])

            P2_X = homo_track[detected_id][0] - opposite_end[0]  
            P2_Y =  -(-opposite_end[1] +  homo_track[detected_id][1])
        else:
            P1_X = local_centre[0] - homo_track[detected_id][0]  
            P1_Y =  -(-local_centre[1] +  homo_track[detected_id][1])

            P2_X = opposite_end[0] - homo_track[detected_id][0]  
            P2_Y =  -(-opposite_end[1] +  homo_track[detected_id][1])
    
    return P1_X,P1_Y,P2_X,P2_Y
    
def get_polar_coordinates(x,y):
    r_ = math.sqrt(x**2+y**2)
    tan_0 = math.atan2(y,x)
    tan_ = math.degrees(tan_0)
    if tan_ < 0:
        tan_ = tan_ + 360
    return r_,tan_
