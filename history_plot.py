import json
import os 
import cv2
import time
from cv2 import flip
import pandas as pd
import numpy as np
import copy
from pymongo import MongoClient

from Utils.detect_utils import get_wt_scorefile_data, get_ae_scorefile_data


class History():
    
    get_wt_scorefile_data = get_wt_scorefile_data
    get_ae_scorefile_data = get_ae_scorefile_data
    def __init__(self):
        self.score_file_data = {}
        self.old_score_line = ""
        self.old_flip = ""
        self.score_line = ""
        self.data={}
        self.json_val = ""
        self.player_dets = {}
        self.master_history = ""
        self.past_score_modiefied_time = 0
        
        with open("Settings/config.json", 'r') as _file:
            _data = json.load(_file)
            self.score_file_path = _data["score_file_path"]
            # self.score_file_path = rfilename2 = r"E:\Internship\Quidich\restructured_code\QT_history_plot\qt_history_backend\QT\score_file"
            self.innings = _data["innings"]
            self.db_name = _data["db_name"]
            self.json_file_path = _data["history_data_path"]
            self.bowl_direction = [_data["near_bowl_dir"],_data["far_bowl_dir"]]
            
            stump = _data["stump"]
            self.near_end_stump = stump["near_end"]
            self.far_end_stump = stump["far_end"]
            self.clear_multiple_entry_flag = _data["clear_multiple_entry_flag"]
            self.score_file_mode = _data["score_file_mode"]
            
            
        self.conn = MongoClient()
        
        db = self.conn.quidich_db_v3
        self.score_collection = db[self.db_name +'_score_data']
        self.livelock_collection = db[self.db_name +'_livelock_data']
        self.history_collection = db[self.db_name + '_history_data']

    def driver(self):
        while True :
            check_flag = self.check_file()
            if check_flag:
                print("Gonna sleep for 2 secs --> New line found")
                time.sleep(2)
                # print("over: ", self.data["over"])
                # print("ball: ",self.data["ball"])        
                ret_val = self.generate_history_plot()
                if ret_val ==0:
                    self.append_json_data()
            

    def check_file(self):

        if self.score_file_mode == "wt":
            self.score_file_data, self.score_line = self.get_wt_scorefile_data(
                        self.score_file_data, self.score_line,
                        self.score_file_path,
                        self.innings
                )
        elif self.score_file_mode == "ae":
            self.score_file_data, self.score_line = self.get_ae_scorefile_data(
                        self.score_file_data, self.score_line,
                        self.score_file_path,
                        self.innings
                )
        

        if  self.score_line != self.old_score_line:
            if self.score_line:
                self.old_score_line = self.score_line
                # print("New Line\n")
                # print("Line :",self.score_line)
                self.data =  {
                    "over":self.score_file_data["Over"],
                    "ball":self.score_file_data["Ball"]
                    }
                return True
            else:
                self.data ={}
                return False
        else:
            self.data ={}
            return False
        
    def append_json_data(self):
        master_history_df ={
            "innings_1":{},
            "innings_2":{},
            "innings_3":{},
            "innings_4":{}
        }
        

        if not os.path.exists(self.json_file_path):
            print("Not present so creating one")
            with open(self.json_file_path, "w+") as outfile:
                json.dump(master_history_df, outfile)

        his = open(self.json_file_path)
        self.master_history = json.load(his)
        over_no = str(int(self.data["over"])-1)
        if over_no not in list(self.master_history["innings_"+str(self.innings)].keys()):
            self.master_history["innings_"+str(self.innings)][over_no] = []

        self.master_history["innings_"+str(self.innings)][over_no].append(self.player_dets)

        if self.clear_multiple_entry_flag:
            self.master_history["innings_"+str(self.innings)][over_no] = list({c['ball']:c for c in self.master_history["innings_"+str(self.innings)][over_no]}.values())


        with open(self.json_file_path, "w") as outfile:
            json.dump(self.master_history, outfile) 


    # def search_binary(self,arr,thresh):
    #     low = 0
    #     high = len(arr) - 1
    #     mid = 0

    #     while low <= high:
    #         mid = (high + low) // 2

    #         if low == high:
    #             return mid

    #         if arr[mid] -arr[mid-1] >thresh:
    #             return mid
    #         elif arr[len(arr)-1] - arr[mid] == arr[mid] - arr[0]:
    #             if arr[len(arr)-1] - arr[0] > 800:
    #                 return mid
    #             else:
    #                 return 0
    #         elif arr[len(arr)-1] - arr[mid] < arr[mid] - arr[0]:
    #             high = mid-1
    #         elif arr[mid] - arr[0] < arr[len(arr)-1]-arr[mid]:
    #             low = mid + 1

    #     return -1

    
        
    def generate_history_plot(self):
        over_no = int(self.data["over"])
        ball_no = int(self.data["ball"])

        
        print(f"over: {str(over_no-1 + ball_no/10)}, innings: {self.innings}")
        over_ball = over_no + ball_no/10
        # print(f"over_ball: {over_ball}")
        
        ball_dict = {}
        master_fr = {}
        green_len  = 0
        ump_id =""
        bowl_id =""
        thresh =10
        
        ball_data_mongo = self.score_collection.find(
            {
                "Over": 
                    {
                    '$gte': int(over_no)-1,
                    '$lte':int(over_no)
                },
                "Innings":{
                    "$eq": int(self.innings)
                }
            }
        ) # need to mention innings
        
        for ball in ball_data_mongo:
            # print("ball:",ball)
            key = str(ball['Over'])+'.'+str(ball['Ball'])
            value = ball['_id']
            if key in ball_dict:
                ball_dict[key].append(value)
            else: 
                ball_dict.setdefault(key, [value])
        # print(ball_dict)

        if str(over_ball) not in ball_dict.keys():
            print(f"Data for {str(over_no-1 + ball_no/10)} is not present for this ball ")
            # self.player_dets["_id"] = data["_id"]
            return -1

        
        # Fetch essential details of the ball 
        ball_details = list(self.score_collection.find(
            {
                "Over": 
                    {
                    '$eq':over_no
                },
                    
                "Ball": 
                    {
                    '$eq':ball_no
                }
            }
        ))

        if len(ball_details)>0:
            ball = ball_details[-1]
            self.player_dets["onstrike_batsman"] = ball["Batsman"]
            self.player_dets["offstrike_batsman"] = ball["Other_Batsman"]
            self.player_dets["bowler"] = ball["Bowler"]
            self.player_dets["over"] = ball["Over"]-1
            self.player_dets["ball"] = ball["Ball"]
            self.player_dets["Bat_type"] = ball["Bat_LH/RH"]
            result = ball['T/Ov']
            if ball['Wkt?'] == 'Y':
                result = 'w'
            elif 'wd' in ball['T/Ov']:
                result = 'wd'
            elif 'lb' in ball['T/Ov']:
                result = 'lb'
            elif 'nb' in ball['T/Ov']:
                result = 'nb'
            self.player_dets["result"] = result
            
        print("\n\nball_dict",ball_dict)
        ball_dict_keys = list(ball_dict.keys())
        ball2_no_tstamp_index = ball_dict_keys.index(str(over_ball))
        if (ball2_no_tstamp_index != 0):
            ball1_no_tstamp_index = ball2_no_tstamp_index -1
            ball1_no = ball_dict_keys[ball1_no_tstamp_index]
            ball1_time_stamp = ball_dict[ball1_no][0]
            
        else:
            print(ball_dict[ball_dict_keys[ball2_no_tstamp_index]][0])

            first_ball = self.livelock_collection.find(
                {
                "_id":{
                    "$lte": ball_dict[ball_dict_keys[ball2_no_tstamp_index]][0]
                    }   
                }
            )
            # for i in first_ball:
            #     print("i",i)

            l_first_ball = list(first_ball)
            if len(l_first_ball)>0:
                ball1_time_stamp = l_first_ball[0]["_id"] 
                
        ball2_time_stamp = ball_dict[str(over_ball)][0]
        if len(ball1_time_stamp) == 0:
            print("Data not present for this ball maybe its the first ball and livelock was not pressed on time")
            
            self.player_dets["players"] = []
            
            return -1
        temp = list(
            self.livelock_collection.find(
                {'_id': {
                        '$gte': ball1_time_stamp,
                        '$lte': ball2_time_stamp
                        }
                    }
                )
            )
        if len(temp) == 0:
            print("Maybe, Livelock was not found hence data not present")
            return -1
        
        ump_lis = temp[-1]["umpire_id"]
        flip_val = temp[-1]["flip"]

        # print("temp",temp[-1])

        if ball_no >= 6:
            if self.old_flip != temp[-1]["flip"] and self.old_flip !="":
                flip_val = copy.deepcopy(self.old_flip)
        
        self.old_flip = copy.deepcopy(flip_val)

        
        print(f"flip_val: {flip_val}")
        print(f"total_frames: {len(temp)}")
        print(f"ump_lis: {ump_lis}")

        
        data= list(self.history_collection.find({
            "_id": {
                '$gte': ball1_time_stamp, 
                '$lt':ball2_time_stamp
            }
        },{'_id': False}
            )
        )

        if len(data) == 0:
            print("Maybe, Livelock was not pressed hence no data found in history collection")
            return -1

        df = pd.DataFrame(data)
        # filling all nan values with -100 list
        for cols in list(df.columns):
            for row in df.loc[df[cols].isnull(), cols].index:
                df.at[row, cols] = [-100,-100,-100,-100]
        
        # flaattening the data from 3d to 2d
        dc = list(df.columns)
        
        if "frame_count" in dc:
            dc.remove("frame_count")

        master_df = pd.DataFrame()
        for i in dc:
            df2 = pd.DataFrame(df[i].to_list(),columns=[f"{i}_x",f"{i}_y",f"{i}_pt",f"{i}_d"])
            master_df = pd.concat([master_df, df2], axis=1)
        
        master_df.insert(loc=0, column='frame_no', value=df["frame_count"])
        
        
        box_cor_f = [self.far_end_stump[0]-60,self.far_end_stump[1],self.far_end_stump[0]+60,self.far_end_stump[1]+70]
        box_cor_n = [self.near_end_stump[0]-60,self.near_end_stump[1]-70,self.near_end_stump[0]+60,self.near_end_stump[1]]
        box_cor = [box_cor_f,box_cor_n]
        wicket_corr = [self.far_end_stump,self.near_end_stump]
        
        # print(f"master_df:{master_df.shape}")

        df_cols = master_df.columns
        # print("df cols",df_cols)
        
        # print("flip vval", flip_val)
        
        x1,y1,x2,y2 = box_cor[flip_val]

        ump_ix = []
        for up in ump_lis :
            ump_ix.append(list(df_cols).index(str(up)+"_x"))
        
        for i in ump_ix:
            # print(df_cols[i+2])
            df_id_x = master_df[df_cols[i]]
            df_id_y = master_df[df_cols[i+1]]
            df_id_pt = master_df[df_cols[i+2]]
            # df_id_dir = master_df[df_cols[i+3]]
            
            fr_ump_x = df_id_x[df_id_x<x2][df_id_x[df_id_x<x2] > x1]
            fr_ump_y = df_id_y[df_id_y<y2][df_id_y[df_id_y<y2] > y1]
            fr_ump_c = list(set(fr_ump_x.index).intersection(set(fr_ump_y.index)))
            # print(f"len(fr_ump_c): {len(fr_ump_c)}")
            if len(fr_ump_c)>0:
                id = str(df_cols[i+2])[:-3]
                master_fr[id] = fr_ump_c
                ump_id = id
                # print("ump_id is ",ump_id)
                break
        
        if len(fr_ump_c) != 0:
            for i in range(1,len(df_cols)-4,4):
                # print(df_cols[i+2])
                df_id_x = master_df[df_cols[i]]
                df_id_y = master_df[df_cols[i+1]]
                df_id_pt = master_df[df_cols[i+2]]
                df_id_dir = master_df[df_cols[i+3]]
                
                fr_x = df_id_x[df_id_x<x2][df_id_x[df_id_x<x2] > x1]
                fr_y = df_id_y[df_id_y<y2][df_id_y[df_id_y<y2] > y1]
                fr_dir = df_id_dir[df_id_dir == self.bowl_direction[flip_val][0]]
                # fr_dir = df_id_dir[df_id_dir == 0]
                # print(fr_dir)
                # print(len(fr_x),len(fr_y))
                fr_c = list(set(fr_x.index).intersection(set(fr_y.index))) #get all frames when x in box and y in box
                fr_ump_bowl = list(set(fr_ump_c).intersection(set(fr_c))) #get all frames when ump in box and green id in box
                # print(fr_ump_bowl)
                fr_b_u = list(set(fr_ump_bowl).intersection(set(fr_dir.index))) #get all frames when green id in box with bowling direction and above condition

                # print("fr_b_u",len(fr_b_u))

                if len(fr_b_u)>0:

                    id = str(df_cols[i+2])[:-3]
                    if df_id_pt[fr_b_u[0]] == 0 and int(id) not in ump_lis:
                        master_fr[id] = fr_b_u
                        if len(fr_b_u) > green_len:
                            green_len = len(fr_b_u)
                            bowl_y = df_id_y
                            bowl_id = id

            if bowl_id == "" and len(self.bowl_direction[flip_val])==2:
                print("trying to find bowler by different direction")
                for i in range(1,len(df_cols)-4,4):
                # print(df_cols[i+2])
                    df_id_x = master_df[df_cols[i]]
                    df_id_y = master_df[df_cols[i+1]]
                    df_id_pt = master_df[df_cols[i+2]]
                    df_id_dir = master_df[df_cols[i+3]]
                    
                    fr_x = df_id_x[df_id_x<x2][df_id_x[df_id_x<x2] > x1]
                    fr_y = df_id_y[df_id_y<y2][df_id_y[df_id_y<y2] > y1]
                    fr_dir = df_id_dir[df_id_dir == self.bowl_direction[flip_val][1]]
                    # fr_dir = df_id_dir[df_id_dir == 0]
                    # print(fr_dir)
                    # print(len(fr_x),len(fr_y))
                    fr_c = list(set(fr_x.index).intersection(set(fr_y.index))) #get all frames when x in box and y in box
                    fr_ump_bowl = list(set(fr_ump_c).intersection(set(fr_c))) #get all frames when ump in box and green id in box
                    # print(fr_ump_bowl)
                    fr_b_u = list(set(fr_ump_bowl).intersection(set(fr_dir.index))) #get all frames when green id in box with bowling direction and above condition

                    # print("fr_b_u",len(fr_b_u))

                    if len(fr_b_u)>0:

                        id = str(df_cols[i+2])[:-3]
                        if df_id_pt[fr_b_u[0]] == 0 and int(id) not in ump_lis:
                            master_fr[id] = fr_b_u
                            if len(fr_b_u) > green_len:
                                green_len = len(fr_b_u)
                                bowl_y = df_id_y
                                bowl_id = id
               
        print("ump_id ",ump_id)
        print("bowl_id ",bowl_id)
        print("ids ",list(master_fr.keys()))

        if ump_id == "" or bowl_id == "":
            print("Data not present for this ball as either umpire is not there or bowler not there")
            self.player_dets["flip"] = flip_val
            # self.player_dets["_id"] = data["_id"]
            self.player_dets["players"] = []
            return 0
        
        tot_frames = list(set(master_fr[bowl_id]).intersection(master_fr[ump_id]))
        tot_frames.sort()

        print("\ntot_frame ",tot_frames)
        # print("bowl_y[tot_frames] ",bowl_y[tot_frames])

        bowl_wic_dif = list(bowl_y[tot_frames] - wicket_corr[flip_val][1])
        if flip_val ==0:
            final_frame = tot_frames[bowl_wic_dif.index(min(bowl_wic_dif))]
        else:
            final_frame = tot_frames[bowl_wic_dif.index(max(bowl_wic_dif))]

        print("bowl_wic_dif ",bowl_wic_dif)
        print("final_frame ",final_frame)


        
        
        # final_frame = int(np.median(tot_frames))

        # mid = self.search_binary(tot_frames,thresh)
        # # print("mid :", mid, "arr[mid] :",tot_frames[mid])
        # if mid!= -1: 
        #     final_frame = tot_frames[mid]

        # print(f"Frame number :{final_frame} is where ball was bowled")
        
        frm_no = int(master_df['frame_no'][final_frame])
        print("Actual frame number :",frm_no)

        livelock_data_mongo = self.livelock_collection.find(
            {
                "frame_count":
                    {
                        '$lte':frm_no
                    },
                '_id': {
                        '$gte': ball1_time_stamp,
                        '$lte': ball2_time_stamp
                        }
                    
                }
        )
        
        data ={}
        for d in livelock_data_mongo:
            data= d
            
        
        del data["frame_count"]
        del data["umpire_id"]
        del data["flip"]

        # print("data",data)
        
        '''

            
        # dst = cv2.imread(r"Settings/dst.jpg")

        
        # for i in range(1,len(list(data.keys()))):
        #     x_homo = abs(data[list(data.keys())[i]]['homo_track'][0])
        #     y_homo = abs(data[list(data.keys())[i]]['homo_track'][1])
        #     dst = cv2.circle(dst, (int(x_homo),int(y_homo)), 3, (0,0,255), 3)
        
        # print("wicket", wicket_corr)

        # dst = cv2.circle(dst, tuple(wicket_corr[0]), 3, (255,0,0), 3)
        # dst = cv2.circle(dst, tuple(wicket_corr[1]), 3, (255,0,0), 3)
        # dst = cv2.rectangle(dst, (box_cor_f[0],box_cor_f[1]), (box_cor_f[2],box_cor_f[3]), (0, 0, 0), 2)
        # dst = cv2.rectangle(dst, (box_cor_n[0],box_cor_n[1]), (box_cor_n[2],box_cor_n[3]), (0, 0, 0), 2)

        # dst = cv2.resize(dst, (1080,720))
        # dst = cv2.rotate(dst, cv2.ROTATE_180)
        # dst = cv2.flip(dst, 0)
        # dst = cv2.resize(dst, (1080,720))
        
        # cv2.imshow("window_name", dst)
        # cv2.waitKey(0)

        # cv2.imwrite(f"history_plot/images/output_{self.innings}_{over_no}_{ball_no}.png",dst)
        '''
        
        # print(data)
        self.player_dets["flip"] = flip_val
        self.player_dets["_id"] = data["_id"]
        del data["_id"]
        self.player_dets["players"] = list(data.values())

        return 0
        

if __name__ == "__main__":
    app = History()
    app.driver()