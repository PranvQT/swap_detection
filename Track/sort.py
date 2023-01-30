import pandas as pd
import numpy as np
from numba import jit
from scipy.optimize import linear_sum_assignment
from Utils.detect_utils import calculate_homography
from Utils.sort_track_utils import convert_x_to_bbox, reset_swap_flags
from Utils.ui_utils import Config
import pickle
import cv2
import os
from PIL import Image
from matplotlib import cm

# import tensorflow as tf
import torch
import torch.nn as nn
from torchvision import datasets, models

import torchvision.transforms as transforms
from torch.autograd import Variable
from Track.model import ft_net

from Track.track import KalmanBoxTracker

class LSTMClassifier(nn.Module):
    """Very simple implementation of LSTM-based time-series classifier."""
    
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim,bidirectional = True,dropout =0.2,batch_first=True)
        self.fc = nn.Linear(hidden_dim*2, output_dim)
        self.batch_size = None
        self.hidden = None
    
    def forward(self, x):
        # print(self.layer_dim)
        h0, c0 = self.init_hidden(x)
        out, (hn, cn) = self.lstm(x, (h0, c0))
        # print(out.shape)
        out = self.fc(out[:, -1, :])
        return out
    
    def init_hidden(self, x):
        h0 = torch.zeros(self.layer_dim*2, x.size(0), self.hidden_dim)
        c0 = torch.zeros(self.layer_dim*2, x.size(0), self.hidden_dim)
        return [t for t in (h0, c0)]

class Sort(object):
    def __init__(self, max_age=800, min_hits=3, iou_threshold=0.3):
        """
        Initializes the class variables for Sort

        Parameters: 
        iou_threshold: It is the threshold to identify the iou
        min_hits:
        max_age: The time since last update is checked with this threshold

        Returns: None, as this initialises the class

        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = {}
        self.trks = []
        self.frame_count = 0
        self.umpire_id = []
        self.batsmen_ids_automated = []
        self.occlusion_dict = {}
        self.verification_dict = {}
        self.occlusion_ids = []
        self.close_id_threshold = 50
        self.swap_dict = {
            "flag_inswap": [],
            "flag_normal": [],
            "temp_fr": [],
            "cent_cor_x": [],
            "cent_cor_y": [],
            "past_dist": [],
            "pot_swap": [],
            "green_id": [],
            "red_id": [],
            "oc_frame": [],
            "in_pitch_box": []
        }
        self.rem_list = []
        self.swap_targets = []
        self.count_id = 0
        self.tagging_info ={}
        self.sort_config = Config()
        self.sort_config.sort_config()
        self.ump_tuple = {}
        self.bastmen_tuple = {}
        self.main_ump_tuple = {}
        self.complete_actionable_data = []
        self.num_frame = 50
        self.num_features = 12
        self.features_df = pd.DataFrame()
        self.occlusion_dictionary = {}
        # self.rf_model = LSTMClassifier(self.num_features, 64, 8, 1)
        # self.x1_model = LSTMClassifier(9, 256, 4, 1)
        # self.y1_model = LSTMClassifier(9, 256, 4, 1)
        # self.x2_model = LSTMClassifier(9, 256, 4, 1)
        # self.y2_model = LSTMClassifier(9, 256, 4, 1)

        # self.rf_model.load_state_dict(torch.load('Track/12_features50frames_best_scaled_data.pth'))
        # self.x1_model.load_state_dict(torch.load('Track/best_x1.pth'))
        # self.y1_model.load_state_dict(torch.load('Track/best_y1.pth'))
        # self.x2_model.load_state_dict(torch.load('Track/best_x2.pth'))
        # self.y2_model.load_state_dict(torch.load('Track/best_y2.pth'))


        model_structure = ft_net(3, stride = 2, ibn = False, linear_num=512)
        model_structure.load_state_dict(torch.load('Track/models/ft_ResNet50/net_last.pth'))
        self.tr_model = model_structure
        self.tr_model.cuda()
        self.cnn = model_structure.eval()
        # self.cnn_tr = 
        self.feature_extractor_cnn_block = torch.nn.Sequential(*list(self.cnn.model.children())[:-1]).cuda()
        self.feature_extractor_add_block = self.cnn.classifier.add_block.cuda()
        self.clf = pickle.load(open('Track/models/classification_model', 'rb'))

        self.data_transforms = transforms.Compose([
                        transforms.Resize((256, 128), interpolation=3),
                        transforms.ToTensor(),
                        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                        ])
        
        
        self.process_config = Config()
        self.process_config.process_config()
        # self.reg_model = tf.keras.models.load_model('my_model.h5')
        self.player_Data = {}
        self.one_hot_encoding = {
                                'UMPIRE' : [0,1,0,0],
                                'BOWLER' : [0,0,1,0],
                                'BATSMAN' : [1,0,0,0], 
                                'fielder' : [0,0,0,1],
                                }
    reset_swap_flags = reset_swap_flags

    def reswap(self, id1, id2):
        """
        Returns the current bounding box estimate and updates the class variables

        Parameters:
        id1: id of the player from the frame that needs to be reswapped
        id2: the id that will be reswaped with

        Returns: None, but updates the trackers with the new values
        """
        # print("here 1",id1,id2)
        temp_track = self.trackers[str(id1)]
        temp_track2 = self.trackers[str(id2)]

        t1_player_type = temp_track.player_type
        t1_direction = temp_track.direction
        t1_highlight = temp_track.highlight
        t1_highlight_streak = temp_track.highlight_streak
        t1_tracklets = temp_track.tracklets

        # print(t1_player_type,t1_direction,t1_highlight,t1_highlight_streak,t1_tracklets)
        t2_player_type = temp_track2.player_type
        t2_direction = temp_track2.direction
        t2_highlight = temp_track2.highlight
        t2_highlight_streak = temp_track2.highlight_streak
        t2_tracklets = temp_track2.tracklets
        # print(t2_player_type,t2_direction,t2_highlight,t2_highlight_streak,t2_tracklets)
        
        self.trackers[str(id1)] = temp_track2
        self.trackers[str(id1)].player_type = t1_player_type
        self.trackers[str(id1)].direction = t1_direction
        self.trackers[str(id1)].highlight = t1_highlight
        self.trackers[str(id1)].highlight_streak = t1_highlight_streak
        self.trackers[str(id1)].tracklets = t1_tracklets
        self.trackers[str(id1)].id = id1

        self.trackers[str(id2)] = temp_track
        self.trackers[str(id2)].player_type = t2_player_type
        self.trackers[str(id2)].direction = t2_direction
        self.trackers[str(id2)].highlight = t2_highlight
        self.trackers[str(id2)].highlight_streak = t2_highlight_streak
        self.trackers[str(id2)].tracklets = t2_tracklets
        self.trackers[str(id2)].id = id2
        # print("here 2")
        
    def get_iou(self, bb1, bb2):
        """
        Calculate the Intersection over Union (IoU) of two bounding boxes.

        Parameters
        bb1 : dict
            Keys: {'x1', 'x2', 'y1', 'y2'}
            The (x1, y1) position is at the top left corner,
            the (x2, y2) position is at the bottom right corner
        bb2 : dict
            Keys: {'x1', 'x2', 'y1', 'y2'}
            The (x, y) position is at the top left corner,
            the (x2, y2) position is at the bottom right corner

        Returns
        float
            in [0, 1]
        """
        assert bb1['x1'] < bb1['x2']
        assert bb1['y1'] < bb1['y2']
        assert bb2['x1'] < bb2['x2']
        assert bb2['y1'] < bb2['y2']

        # determine the coordinates of the intersection rectangle
        x_left = max(bb1['x1'], bb2['x1'])
        y_top = max(bb1['y1'], bb2['y1'])
        x_right = min(bb1['x2'], bb2['x2'])
        y_bottom = min(bb1['y2'], bb2['y2'])

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        # The intersection of two axis-aligned bounding boxes is always an
        # axis-aligned bounding box
        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # compute the area of both AABBs
        bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
        bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = intersection_area / \
            float(bb1_area + bb2_area - intersection_area)
        assert iou >= 0.0
        assert iou <= 1.0
        return iou

    def highlightPlayer(self, player_id):
        """
        changes the highlight parameter of the highlighted player

        Parameters:
        player_id: the id of the player that has been highlighted on the UI

        Returns: None, updates the corresponding class variables
        """
        for trk in self.trackers.values():
            if trk.id == (player_id - 1):
                trk.highlight = 1 if trk.highlight == 0 else 0

    def dehighlight(self):
        """
        changes the highlight parameter of the dehighlighted player

        Parameters:
        player_id: the id of the player that has been dehighlighted on the UI

        Returns: None, updates the corresponding class variables
        """
        for trk in self.trackers.values():
            trk.highlight = 0

    def highlightPlayerStreak(self, player_id):
        """
        updates the highlight streak parameter of the class to 1

        Parameters:
        player_id: the id of the player that has been added to highlight streak on the UI

        Returns: None, updates the corresponding class variables as 1 or 0
        """
        for trk in self.trackers.values():
            if trk.id == (player_id - 1):
                trk.highlight_streak = 1 if (trk.highlight_streak == 0) else 0

    def dehighlight_streak(self):
        """
        updates the highlight streak parameter of the class to 0

        Parameters:
        player_id: the id of the player that has been removed from the highlight streak on the UI

        Returns: None, updates the corresponding class variables as 1 or 0
        """
        for trk in self.trackers.values():
            trk.highlight_streak = 0

    def change_playertype(self, player_id, player_type):
        """
        updates the player type parameter of a player

        Parameters:
        player_id: the id of the player that has been highlighted on the UI
        player_type: the player type of the player 

        Returns: None, updates the corresponding class variables of the players with the new player types
        """
        for trk in self.trackers.values():
            if trk.id == (player_id - 1):
                trk.player_type = player_type

    def get_distance(self, p1_corr, p2_corr):
        """
        calculates the distance between 2 players

        Parameters:
        p1_corr: player 1 coordinate
        p2_corr: player 2 coordinate

        Returns: returns the euclidean distance of two players
        """
        p1_cor_x = float((p1_corr[2]+p1_corr[0])/2)
        p1_cor_y = float((p1_corr[3]+p1_corr[1])/2)

        p2_cor_x = float((p2_corr[2]+p2_corr[0])/2)
        p2_cor_y = float((p2_corr[3]+p2_corr[1])/2)

        return np.sqrt((p2_cor_x-p1_cor_x)**2 + (p2_cor_y-p1_cor_y)**2)

    def check_for_swap(self, val_swap, frm_count, identities,homo_track):
        # sourcery no-metrics
        """
        this function checks if there is any need of reswap and performs the same if required

        Parameters:
        val_swap: this is the index of the id that has been identified for a potential swap situation
        frm_count: current frame count for reference
        identities: list of identities which are present on frame
        bbox_xyxy: the bounding boxes of the identities

        Returns: None, all the siituations are considered and the ids are reswaped if a swap occurs and it is udpated in the class variable
        """

        green_id = self.swap_dict["green_id"][val_swap]
        red_id = self.swap_dict["red_id"][val_swap]
        flag_inswap = self.swap_dict['flag_inswap'][val_swap]
        flag_normal = self.swap_dict['flag_normal'][val_swap]
        pot_swap = self.swap_dict['pot_swap'][val_swap]
        if pot_swap is True:

            temp_fr = self.swap_dict['temp_fr'][val_swap]
            cent_cor_x = self.swap_dict['cent_cor_x'][val_swap]
            cent_cor_y = self.swap_dict['cent_cor_y'][val_swap]
            oc_frame = self.swap_dict['oc_frame'][val_swap]
            in_pitch_box = self.swap_dict["in_pitch_box"][val_swap]

            if oc_frame != -1 and frm_count >= oc_frame+50 and flag_inswap:
                print("Reset flags")
                # print("frm_count",frm_count,oc_frame)
                self.swap_dict['pot_swap'][val_swap] = False
                self.swap_dict['flag_inswap'][val_swap] = False
                self.swap_dict['oc_frame'][val_swap] = -1
                if val_swap not in self.rem_list:
                    self.rem_list.append(val_swap)
            if (green_id in identities) and (red_id in identities) and flag_inswap:
                if temp_fr == 15:
                    self.swap_dict['pot_swap'][val_swap] = False
                    self.swap_dict['oc_frame'][val_swap] = -1
                    if val_swap not in self.rem_list:
                        self.rem_list.append(val_swap)

                    p1_new_corr = homo_track[red_id]
                    p2_new_corr = homo_track[green_id]

                    p1_new_cen_x = p1_new_corr[0]
                    p1_new_cen_y = p1_new_corr[1]

                    p2_new_cen_x = p2_new_corr[0]
                    p2_new_cen_y = p2_new_corr[1]

                    dist1 = np.sqrt((p1_new_cen_x-cent_cor_x) **
                                    2 + (p1_new_cen_y-cent_cor_y)**2)
                    dist2 = np.sqrt((p2_new_cen_x-cent_cor_x) **
                                    2 + (p2_new_cen_y-cent_cor_y)**2)

                    temp_lis = [dist1, dist2]
                    # print(temp_lis)
                    min_val = temp_lis.index(min(temp_lis))
                    if in_pitch_box is True:
                        pass
                        # print("Inside the pitch it is !! ")
                        if min_val == 0:
                            pass
                            # self.reswap(int(green_id - 1), int(red_id - 1))
                            # print("reswap is needed ")

                        else:
                            pass
                            # print("No Reswap needed")
                    elif min_val == 0:
                        # print("No reswap is needed ")
                        pass
                    else:
                        # self.reswap(int(green_id - 1), int(red_id - 1))
                        # print("Reswap needed")
                        pass

                    # print("\n")
                    self.swap_dict['flag_inswap'][val_swap] = False
                    self.swap_dict['temp_fr'][val_swap] = 0

                else:
                    self.swap_dict['temp_fr'][val_swap] += 1

            elif green_id in identities and red_id in identities:

                p1_corr = homo_track[red_id]
                p2_corr = homo_track[green_id]

                p1_cen_x = p1_corr[0]
                p1_cen_y = p1_corr[1]

                p2_cen_x = p2_corr[0]
                p2_cen_y = p2_corr[1]

                self.swap_dict['past_dist'][val_swap] = np.sqrt(
                    (p1_cen_x-p2_cen_x)**2 + (p1_cen_y-p2_cen_y)**2)
                # print("here",self.swap_dict['past_dist'][val_swap])
                if self.swap_dict['past_dist'][val_swap] > 30:
                    self.swap_dict['pot_swap'][val_swap] = False
                    self.swap_dict['oc_frame'][val_swap] = -1
                    if val_swap not in self.rem_list:
                        # print("appending value coz too far (30)")
                        self.rem_list.append(val_swap)

                if in_pitch_box:
                    self.swap_dict['cent_cor_x'][val_swap] = p2_cen_x
                    self.swap_dict['cent_cor_y'][val_swap] = p2_cen_y
                else:
                    self.swap_dict['cent_cor_x'][val_swap] = p1_cen_x
                    self.swap_dict['cent_cor_y'][val_swap] = p1_cen_y

                self.swap_dict['flag_normal'][val_swap] = True

            elif flag_normal:
                # print("Occlusion Occured")
                self.swap_dict['oc_frame'][val_swap] = frm_count
                # print("occlusion frm number :",self.swap_dict['oc_frame'][val_swap])

                self.swap_dict['flag_normal'][val_swap] = False
                self.swap_dict['flag_inswap'][val_swap] = True
    
    

    def linear_assignment(self, cost_matrix):
        """
        Computes the value for linear sum assignment problem

        The linear sum assignment problem is also known as minimum weight matching 
        in bipartite graphs. A problem instance is described by a matrix C, where each C[i,j] 
        is the cost of matching vertex i of the first partite set (a "worker") and vertex j of 
        the second set (a "job"). The goal is to find a complete assignment of workers to jobs of 
        minimal cost.

        Parameters: 
        cost_matrix: it is the iou matrix of the frame 

        Returns: the x and y values of the computed linear sum assignment
        """

        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))

    # @njit(fastmath=True)
    def iou_batch(self):
        """
        Computes IOU between two boxes in the form [x1,y1,x2,y2]

        Parameters: None

        Returns: None, but updates the class variables with the new values
        """
        bb_gt = np.expand_dims(self.trks, 0)
        bb_test = np.expand_dims(self.detections, 1)

        xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
        yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
        xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
        yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
        w = np.maximum(0., xx2 - xx1)
        h = np.maximum(0., yy2 - yy1)
        wh = w * h
        return wh / (
            (bb_test[..., 2] - bb_test[..., 0])
            * (bb_test[..., 3] - bb_test[..., 1])
            + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1])
            - wh
        )

    @jit(forceobj=True)  # forceobj=True, parallel=True)
    def associate_detections_to_trackers(self):
        """
        Assigns detections to tracked object (both represented as bounding boxes)
        Returns 3 lists of 
        1. matches,
        2. unmatched_detections
        3. unmatched_trackers

        Parameters: None

        Returns: None, but updates the class variables with the new values


        """
        if len(self.trks) == 0:

            return np.empty((0, 2), dtype=int), np.arange(len(self.detections)), np.empty((0, 5), dtype=int)

        iou_matrix = self.iou_batch()
        if min(iou_matrix.shape) > 0:
            a = (iou_matrix > self.iou_threshold).astype(np.int32)
            if a.sum(1).max() == 1 and a.sum(0).max() == 1:
                matched_indices = np.stack(np.where(a), axis=1)
            else:
                matched_indices = self.linear_assignment(-iou_matrix)
        else:
            matched_indices = np.empty(shape=(0, 2))

        unmatched_detections = []
        for d, det in enumerate(self.detections):
            if(d not in matched_indices[:, 0]):
                unmatched_detections.append(d)

        unmatched_trackers = []
        for t, trk in enumerate(self.trks):
            if(t not in matched_indices[:, 1]):
                unmatched_trackers.append(t)

        # filter out matched with low IOU
        matches = []
        for m in matched_indices:
            if(iou_matrix[m[0], m[1]] < self.iou_threshold):
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
            else:
                matches.append(m.reshape(1, 2))

        if len(matches) == 0:
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)

        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

    

    # def get_features(self):
    #     data_dir = 'save_data'
    #     image_datasets = datasets.ImageFolder('save_data/gallery/',self.data_transforms)
                
    #     dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=2,
    #                                          shuffle=False, num_workers=16)
    #     complete_df = pd.DataFrame()
    #     count =0
    #     for iter, data in enumerate(dataloaders):
    #         # print(data)
    #         img, label = data
    #         n, c, h, w = img.size()
    #         count += n
    #         # print(count)
    #         # ff = torch.FloatTensor(n,opt.linear_num).zero_().cuda()
            
    #         input_img = Variable(img.cuda())
    #         outputs = self.tr_model(input_img)
    #         df = pd.DataFrame(outputs.cpu())
    #         df['class']=label
    #         complete_df = pd.concat([complete_df, df], axis=0)
    #     complete_df.to_csv("save_data/featuers.csv",index=False)
    #     print(count)
    
    # def train_on_featuers(self):
    #     df = pd.read_csv('save_data/featuers.csv')
        
    #     #Create a svm Classifier
    #     clf = svm.SVC(kernel='linear') # Linear Kernel
    #     X_train = df.loc[:, df.columns != 'class']
    #     y_train = df.loc[:,df.columns == 'class']
    #     clf.fit(X_train, y_train)
    #     pickle.dump(clf, open('Track/models/classification_model', 'wb'))
    
    def check_occlusion_end(self,occlusion_ids,identities,bounding_box):
        flag = 0 
        # print(occlusion_ids)
        for i in range(len(occlusion_ids)-1):
            id_1 = occlusion_ids[i]
            for j in range(i+1,len(occlusion_ids)):
                id_2 = occlusion_ids[j]
                idx_1 = np.where(identities == id_1)[0]
                idx_2 = np.where(identities == id_2)[0]
                if id_1 == id_2:
                    return 1
                # print(idx_1,idx_2)
                try:
                    new_distance = self.get_distance(bounding_box[idx_1][0],bounding_box[idx_2][0])
                except:
                    new_distance = 3
                # print(new_distance,flag)
                if new_distance > 50:
                    flag = 1
                else:
                    return 0
                # print()
        return flag
    
    def prediction_for_swap_flag(self,f_in,identities,bounding_box,use_verifiction_dict=0):
        # print(f_in,identities,bounding_box,use_verifiction_dict)
        swap_flag = 0
        predicted_clsses = []
        if use_verifiction_dict ==1:
            oc_dict = {}
            oc_dict = self.verification_dict
        else:
            oc_dict = {}
            oc_dict = self.occlusion_dict
        
        
        for i,id in enumerate(oc_dict[f_in][0]):
            clss = int(oc_dict[f_in][1][i])
            idx = np.where(identities == id)[0]
            try:
                x11, y11, x12, y12 = [int(i) for i in bounding_box[idx][0]]
            except:
                return swap_flag,[]
            
            per = self.img[y11:y12,x11:x12]
            cv2.imwrite("Track/per"+str(int(len(os.listdir('Track'))))+".jpg",per)
            pil_image = Image.fromarray(cv2.cvtColor(per, cv2.COLOR_BGR2RGB))
            img_tensor = self.data_transforms(pil_image)
            input_img = Variable(img_tensor.float().cuda())
            output = self.feature_extractor_cnn_block(input_img.unsqueeze(0))
            per_features = pd.DataFrame(self.feature_extractor_add_block(output.reshape(1, 2048)).detach().cpu())
            predicted_class = int(self.clf.predict(per_features)[0])
            predicted_clsses.append(predicted_class)
            if clss != predicted_class:
                swap_flag = 1
        
        return swap_flag,predicted_clsses

    def take_decision(self,frm_count,identities,bounding_box):
        if len(self.verification_dict) > 0:
            for frm_out in list(self.verification_dict.keys()):
                
                # print(frm_count - frm_out)
                if frm_count - frm_out  >100:
                    del self.verification_dict[frm_out]
                    continue
                elif frm_count - frm_out  >50:
                    # print(frm_out,self.verification_dict)
                    
                    verification_flag, verification_clsses = self.prediction_for_swap_flag(frm_out,identities,bounding_box,use_verifiction_dict=int(1))
                    print(self.verification_dict[frm_out],verification_clsses,frm_count)

                    # print(self.verification_dict[frm_out][3],"old flag")
                    # print(verification_flag,"new flag")
                    if len(verification_clsses)==0:
                        continue
                    if verification_flag == 1 and self.verification_dict[frm_out][3]==1:
                        if len(verification_clsses)==2:
                            # print(int(self.verification_dict[frm_out][0][0]),int(self.verification_dict[frm_out][0][1]),"len(verification_clsses)==2")
                            self.reswap(int(self.verification_dict[frm_out][0][0])-1,int(self.verification_dict[frm_out][0][1])-1)
                            # print("\n\n\n")
                            del self.verification_dict[frm_out]
                            continue
                        else:
                            indices = np.where(np.array(verification_clsses) != np.array(self.verification_dict[frm_out][1]))[0]
                            print(indices)
                            if len(indices) ==2:
                                # print(self.verification_dict[frm_out])
                                # print(int(self.verification_dict[frm_out][0][indices[0]]),int(self.verification_dict[frm_out][0][indices[1]]),"len(verification_clsses)>2 swapped only 2")
                                self.reswap(int(self.verification_dict[frm_out][0][indices[0]])-1,int(self.verification_dict[frm_out][0][indices[1]])-1)
                                # print("\n\n\n")
                                del self.verification_dict[frm_out]
                                continue
                            elif len(indices) ==1:
                                # print(self.verification_dict[frm_out])
                                # print("\n\n\n")
                                del self.verification_dict[frm_out]
                                continue
                            elif len(indices) ==3 and len(self.verification_dict[frm_out][1]) == 3:
                                # print(self.verification_dict[frm_out])
                                ori_cls_1 = self.verification_dict[frm_out][1][0]
                                swap_index_1 = np.where(np.array(verification_clsses) == ori_cls_1)
                                ori_cls_2 = self.verification_dict[frm_out][1][1]
                                swap_index_2 = np.where(np.array(verification_clsses) == ori_cls_2)
                                # print(int(self.verification_dict[frm_out][0][0]),int(self.verification_dict[frm_out][0][swap_index_1]))
                                # print(int(self.verification_dict[frm_out][0][0]),int(self.verification_dict[frm_out][0][swap_index_1]))

                                self.reswap(int(self.verification_dict[frm_out][0][0])-1,int(self.verification_dict[frm_out][0][swap_index_1])-1)
                                self.reswap(int(self.verification_dict[frm_out][0][1])-1,int(self.verification_dict[frm_out][0][swap_index_2])-1)
                                # print("\n\n\n")
                                del self.verification_dict[frm_out]

                                continue
                    elif verification_flag == 0 and self.verification_dict[frm_out][3]==0:
                        del self.verification_dict[frm_out]
                        # print("\n\n\n")
                        continue
                    else:
                        pass

                    # print("\n\n\n")
                
                else:
                    pass

    def check(self,bounding_box,identities,frm_count):
        self.take_decision(frm_count,identities,bounding_box)
        if len(self.occlusion_dict.values()) > 0:
            for f_in in list(self.occlusion_dict.keys()):
                # print(self.check_occlusion_end(self.occlusion_dict[f_in][0],identities,bounding_box))
                if self.check_occlusion_end(self.occlusion_dict[f_in][0],identities,bounding_box):
                    # f_in = list(self.occlusion_dict.keys())[i]
                    swap_flag,predicted_clsses = self.prediction_for_swap_flag(f_in,identities,bounding_box,use_verifiction_dict=int(0))
                    
                    # print(self.occlusion_dict)
                    # print(self.occlusion_ids)
                    
                    if len(self.verification_dict)==0:
                        self.verification_dict[frm_count] = [self.occlusion_dict[f_in][0],self.occlusion_dict[f_in][1],predicted_clsses,swap_flag,f_in]
                    else:
                        for i,fo in enumerate(list(self.verification_dict.keys())):
                            if f_in ==  self.verification_dict[fo][4]:
                                pass
                            else:
                                self.verification_dict[frm_count] = [self.occlusion_dict[f_in][0],self.occlusion_dict[f_in][1],predicted_clsses,swap_flag,f_in]
                    self.occlusion_ids = [x for x in self.occlusion_ids if x not in self.occlusion_dict[f_in][0]]            
                    del self.occlusion_dict[f_in]

                    
                    # id_1 = int(self.occlusion_dict[f_in][0][0])
                    # orignal_class_1 = int(self.occlusion_dict[f_in][1][0]) 
                    
                    # id_2 = int(self.occlusion_dict[f_in][0][1])
                    # orignal_class_2 = int(self.occlusion_dict[f_in][1][1])

                    # if id_1 == id_2:
                    #     del self.occlusion_dict[f_in]
                    #     break
                    
                    # f_out =0
                    # idx_1 = np.where(identities == id_1)[0]
                    # idx_2 = np.where(identities == id_2)[0]
                    # try:
                    #     new_distance = self.get_distance(bounding_box[idx_1][0],bounding_box[idx_2][0])
                    # except:
                    #     new_distance = 3
                    # # print(tup,new_distance)
                    # # print(id_1,id_2)
                    # # print(identities)
                    # # print(bounding_box)
                    # # print(bounding_box[idx_1][0])
                    # # print(bounding_box[idx_2][0])
                    # try:
                    #     x11, y11, x12, y12 = [int(i) for i in bounding_box[idx_1][0]]
                    #     x21, y21, x22, y22 = [int(i) for i in bounding_box[idx_2][0]]
                    # # print(x11, y11, x12, y12)
                    # except:
                    #     break
                    # if (new_distance > 50):
                    #     # print('true')                    
                    #     # per_1 = self.get_pil(self.img[y11:y12,x11:x12])
                    #     # per_2 = self.get_pil(self.img[y21:y22,x21:x22])
                    #     # print(per_2.shape)
                    #     per_1 = self.img[y11:y12,x11:x12]
                    #     per_2 = self.img[y21:y22,x21:x22]
                    #     pil_image_1 = Image.fromarray(cv2.cvtColor(per_1, cv2.COLOR_BGR2RGB))
                    #     pil_image_2 = Image.fromarray(cv2.cvtColor(per_2, cv2.COLOR_BGR2RGB))

                        
                    #     img_tensor_1 = self.data_transforms(pil_image_1)
                    #     input_img_1 = Variable(img_tensor_1.float().cuda())
                        
                        
                    #     img_tensor_2 = self.data_transforms(pil_image_2)
                    #     input_img_2 = Variable(img_tensor_2.float().cuda())
                        
                        
                    #     # print(input_img_2)
                    #     output_1 = self.feature_extractor_cnn_block(input_img_1.unsqueeze(0))
                    #     per_1_features = pd.DataFrame(self.feature_extractor_add_block(output_1.reshape(1, 2048)).detach().cpu())
                    #     predicted_class_1 = int(self.clf.predict(per_1_features)[0])

                    #     output_2 = self.feature_extractor_cnn_block(input_img_2.unsqueeze(0))
                    #     per_2_features = pd.DataFrame(self.feature_extractor_add_block(output_2.reshape(1, 2048)).detach().cpu())
                    #     predicted_class_2 = int(self.clf.predict(per_2_features)[0])
                        
                    #     print(self.occlusion_dict[f_in],predicted_class_1,self.clf.predict_proba(per_1_features)[0][predicted_class_1],predicted_class_2,self.clf.predict_proba(per_2_features)[0][predicted_class_2])

                    #     # if self.clf.predict_proba(per_1_features)[0][predicted_class_1] > 0.90:
                    #         # if self.clf.predict_proba(per_2_features)[0][predicted_class_2] > 0.90:
                    #     if orignal_class_1 == predicted_class_1:
                    #         if orignal_class_2 == predicted_class_2:
                    #             print("NO reswap required")
                    #             self.occlusion_ids.remove(id_1)
                    #             self.occlusion_ids.remove(id_2)
                    #             del self.occlusion_dict[f_in]
                    #             break
                    #         elif orignal_class_2 == predicted_class_1:
                    #             print("soft reswap")
                    #             self.occlusion_ids.remove(id_1)
                    #             self.occlusion_ids.remove(id_2)
                    #             del self.occlusion_dict[f_in]
                    #             break
                    #         else:
                    #             self.occlusion_ids.remove(id_1)
                    #             self.occlusion_ids.remove(id_2)
                    #             del self.occlusion_dict[f_in]
                    #             break
                
                    #     elif orignal_class_1 == predicted_class_2:
                    #         if orignal_class_2 == predicted_class_1:
                    #             self.reswap(int(id_1)-1,int(id_2)-1)
                    #             self.occlusion_ids.remove(id_1)
                    #             self.occlusion_ids.remove(id_2)
                    #             print("reswaped")
                    #             del self.occlusion_dict[f_in]
                    #             break
                    #         elif orignal_class_2 == predicted_class_2:
                    #             print("soft reswap")
                    #             self.occlusion_ids.remove(id_1)
                    #             self.occlusion_ids.remove(id_2)
                    #             # self.reswap(int(id_1)-1,int(id_2)-1)
                    #             del self.occlusion_dict[f_in]
                    #             break
                    #         else:
                    #             self.occlusion_ids.remove(id_1)
                    #             self.occlusion_ids.remove(id_2)
                    #             del self.occlusion_dict[f_in]
                    #             break
                        
    def get_features(self,bbox):
        """
        This function is used to extract features from a given bounding box.
        
        Parameters:
        bbox (list): a list of bounding box coordinates in the format [x1, y1, x2, y2]
        
        Returns:
        per_features (DataFrame): a DataFrame containing the extracted features
        """
        x11, y11, x12, y12 = [int(i) for i in bbox]
        # extract the portion of image corresponding to the bounding box
        per = self.img[y11:y12,x11:x12]
        
        # convert the image to PIL image
        pil_image = Image.fromarray(cv2.cvtColor(per, cv2.COLOR_BGR2RGB))
        
        # apply data transformations
        img_tensor = self.data_transforms(pil_image)
        
        # convert the image to a PyTorch variable
        input_img = Variable(img_tensor.float().cuda())
        
        # pass the image through the CNN block
        output = self.feature_extractor_cnn_block(input_img.unsqueeze(0))
        
        # pass the output through the additional block
        per_features = pd.DataFrame(self.feature_extractor_add_block(output.reshape(1, 2048)).detach().cpu())

        return per_features
    
    def higher_frame(self,i,j):
        higher =  lambda int(i), int(j): int(i) if int(i) > int(j) else int(j)
        return higher
        
    def check_occlusions(self,ids,bbox,frm_count,np_ptype):
        """
        This function is used to check for occlusions among the objects in the current frame, 
        and update the occlusion_dictionary accordingly.

        Parameters:
        ids (list): a list of object ids in the current frame
        bbox (list): a list of bounding boxes corresponding to the objects in the current frame
        frm_count (int): the current frame count
        
        """
        
        ids = list(ids)
        if len(self.occlusion_dictionary) ==0:
            current_id = ids
            current_bbox = bbox
        else:
            for i,key in enumerate(self.occlusion_dictionary.keys()):
                deletion_ids = self.occlusion_dictionary[key][0]
                deletion_bbox = self.occlusion_dictionary[key][1]

                current_id = [j for j in ids if j not in deletion_ids]
                current_id.append(str(key))
                
                for id in deletion_ids:
                    if id in ids:
                        index = ids.index(id)
                        del current_bbox[index]
                result = (np.array(deletion_bbox).sum(axis=0))/2
                current_bbox.append(result)

        for i in range(len(current_id)):
            if np_ptype[i] == 3:
                continue
            for j in range(i+1,len(current_id)):
                if np_ptype[j] == 3:
                    continue
                # calculate the distance between current_bboxes[i] and current_bboxes[j]
                try:
                    distance = self.get_distance(current_bbox[i],current_bbox[j])
                except:
                    continue
                # check if the distance is less than 50 and greater than 30
                if distance < 50 and distance >30:
                    if type(i) != str and type(j) != str:
                        self.occlusion_dictionary[frm_count] = [
                                    [
                                        current_id[i],
                                        current_id[j]
                                        ],
                                    [
                                        current_bbox[i],
                                        current_bbox[j]
                                        ],
                                    [   
                                        self.get_features(current_bbox[i]),
                                        self.get_features(current_bbox[j])
                                        ]
                                    ]
                    elif type(i) == str:
                        self.occlusion_dictionary[int(i)][0].append(j)
                        self.occlusion_dictionary[int(i)][1].append(current_bbox[j])
                        self.occlusion_dictionary[int(i)][2].append(self.get_features(current_bbox[j]))
                    elif type(j) == str:
                        self.occlusion_dictionary[int(j)][0].append(i)
                        self.occlusion_dictionary[int(j)][1].append(current_bbox[i])
                        self.occlusion_dictionary[int(j)][2].append(self.get_features(current_bbox[i]))
                    else:
                        new_key = self.higher_frame(i,j)
                        self.occlusion_dictionary[new_key] = [
                                    self.occlusion_dictionary[int(i)][0]+
                                    self.occlusion_dictionary[int(j)][0]
                                    ,
                                    self.occlusion_dictionary[int(i)][1]+
                                    self.occlusion_dictionary[int(j)][1]
                                    ,
                                    self.occlusion_dictionary[int(i)][2]+
                                    self.occlusion_dictionary[int(j)][2]
                                    
                                    ]


    def update(self,homo_track,livelock_ids,fielder_dict_PO,img,dets=np.empty((0, 6)), frm_count=0, FRAME_HT=2160, FRAME_WD=3840):
        # sourcery no-metrics
        push_data = {}
        push_data["players"] = []
        downstream_data = {}
        downstream_data["players"] = {}
        self.detections = dets
        self.fielder_dict_PO = fielder_dict_PO 
        self.img = img

        """
        update function is used to update all the corresponding frame ids and its parameters depending on
        the previous values

        Parameters:
        dets - a numpy array of detection in the format [[x1, y1, x2, y2, score], [x1,y1,x2,y2,score],...]

        Ensure to call this method even frame has no detections. (pass np.empty((0,5)))

        Returns: a similar array, where the last column is object ID (replacing confidence score)

        NOTE: The number of objects returned may differ from the number of objects provided.
        """
        self.frame_count += 1
        params = []
        # Get predicted locations from existing trackers
        trks = np.zeros((len(self.trackers), 6))
        to_del = []
        ret = []
        for t, trk in enumerate(list(self.trackers.keys())):
            # ##print(t,trk)
            pos = self.trackers[str(trk)].predict()[0]
            trks[t][:] = [pos[0], pos[1], pos[2], pos[3], 0, 0]
            if np.any(np.isnan(pos)):
                to_del.append(trk)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        self.trks = trks

        matched, unmatched_dets, unmatched_trks = self.associate_detections_to_trackers()

        # print(unmatched_dets)

       # REASSIGN

        for detection_idx in unmatched_dets:
            if len(unmatched_trks) > 0:
                # reallocate old ids if within Euclidean dist thresh else initiate new track
                tlwh_track = []
                tlwh_det = dets[detection_idx]
                dst_boxes = []
                track_list = []
                track_list_unmached = []
                for track_idx in unmatched_trks:
                    # try:
                    temp_id = list(self.trackers.keys())[track_idx]
                    tlwh_track = convert_x_to_bbox(
                        self.trackers[temp_id].kf.x[:4])
                    dst = np.sqrt(
                        (tlwh_track[0][0]-tlwh_det[0])**2 + (tlwh_track[0][1]-tlwh_det[1])**2)
                    dst_boxes.append(dst)
                    track_list.append(int(temp_id))
                    track_list_unmached.append(track_idx)
                    # except:
                    #     pass
                idx = np.argmin(dst_boxes)

                if dst_boxes[idx] < self.sort_config.reassign_dist_thresh:
                    track_idx_unmatched = track_list_unmached[idx]
                    track_idx = track_list[idx]

                    unmatched_trks = np.delete(unmatched_trks, np.argwhere(
                        unmatched_trks == track_idx_unmatched))
                    unmatched_dets = np.delete(
                        unmatched_dets, np.argwhere(unmatched_dets == detection_idx))

                    detection = dets[detection_idx]
                    trk = KalmanBoxTracker(
                        np.hstack((dets[detection_idx, :], np.array([0]))), 1000)

                    trk.id = int(track_idx)
                    trk.direction = self.trackers[str(track_idx)].direction
                    trk.player_type = self.trackers[str(track_idx)].player_type
                    trk.highlight = self.trackers[str(track_idx)].highlight
                    trk.highlight_streak = self.trackers[str(
                        track_idx)].highlight_streak
                    trk.tracklets = self.trackers[str(track_idx)].tracklets
                    trk.temp_close_ids = self.trackers[str(
                        track_idx)].temp_close_ids
                    trk.is_merged = self.trackers[str(track_idx)].is_merged
                    trk.time_since_update = 0
                    trk.history = []
                    trk.hits += 1
                    trk.hit_streak += 1
                    self.trackers[str(track_idx)] = trk

        for m in matched:
            # try:
            temp_id = list(self.trackers.keys())[m[1]]
            self.trackers[temp_id].update(dets[m[0], :])
            # except:
            #     pass
        # Create and initialize new trackers for unmatched detections
        for i in unmatched_dets:
            # # Automating Reset 
            id_list = [trk.id for trk in reversed(list(self.trackers.values()))]
            for idd in range(0,50):
                # print(self.trackers.keys())
                if str(idd) not in self.trackers.keys():
                    if idd not in id_list:
                        if str(idd+1) not in livelock_ids.keys():
                            self.count_id = idd
                            break
    
            trk = KalmanBoxTracker(
                np.hstack((dets[i, :], np.array([0]))), self.count_id)
            self.trackers[str(self.count_id)] = trk
            self.count_id += 1

        for trk_id in unmatched_trks:
            temp_id = (list(self.trackers.keys())[trk_id])
            if self.trackers[temp_id].close_ids != []:

                self.trackers[temp_id].is_merged = True
                self.trackers[temp_id].temp_close_ids = self.trackers[temp_id].close_ids

        i = len(self.trackers)
        highlights = []
        highlight_streaks = []
        player_type = []
        directions = []
        to_del = []
        for i, swap in enumerate(self.swap_targets):
            id1 = swap[1]
            id2 = swap[4]
            if(frm_count == swap[0]) and (str(id1) in self.trackers.keys()) and (str(id2) in self.trackers.keys()):
                id1 = swap[1]
                id2 = swap[4]
                direction1 = swap[2]
                direction2 = swap[5]
                bbox1 = swap[3]
                bbox2 = swap[6]
                track_box1 = self.trackers[str(id1)].get_state()[0][0][:4]
                track_box2 = self.trackers[str(id2)].get_state()[0][0][:4]

                centroid1 = (int(
                    int(track_box1[0] + track_box1[2])/2), int(int(track_box1[1] + track_box1[3])/2))
                centroid2 = (
                    int(int(bbox1[0] + bbox1[2])/2), int(int(bbox1[1] + bbox1[3])/2))

                x_diff = track_box1[0] - bbox1[0]
                y_diff = track_box1[1] - bbox1[1]
                if(((x_diff <= self.sort_config.direction_threshold) and (x_diff >= (-1 * self.sort_config.direction_threshold))) and ((y_diff <= self.sort_config.direction_threshold) and (y_diff >= (-1 * self.sort_config.direction_threshold)))):
                    direction_track1 = 0
                elif((x_diff >= self.sort_config.direction_threshold) and ((y_diff <= self.sort_config.direction_threshold) and (y_diff >= (-1 * self.sort_config.direction_threshold)))):
                    direction_track1 = 1
                elif((x_diff >= self.sort_config.direction_threshold) and (y_diff >= self.sort_config.direction_threshold)):
                    direction_track1 = 2
                elif(((x_diff <= self.sort_config.direction_threshold) and (x_diff >= (-1 * self.sort_config.direction_threshold))) and (y_diff >= self.sort_config.direction_threshold)):
                    direction_track1 = 3
                elif((x_diff <= (-1 * self.sort_config.direction_threshold)) and (y_diff >= self.sort_config.direction_threshold)):
                    direction_track1 = 4
                elif((x_diff <= (-1 * self.sort_config.direction_threshold)) and ((y_diff <= self.sort_config.direction_threshold) and (y_diff >= (-1 * self.sort_config.direction_threshold)))):
                    direction_track1 = 5
                elif((x_diff <= (-1 * self.sort_config.direction_threshold)) and (y_diff <= (-1 * self.sort_config.direction_threshold))):
                    direction_track1 = 6
                elif(((x_diff <= self.sort_config.direction_threshold) and (x_diff >= (-1 * self.sort_config.direction_threshold))) and (y_diff <= (-1 * self.sort_config.direction_threshold))):
                    direction_track1 = 7
                elif((x_diff >= self.sort_config.direction_threshold) and (y_diff <= (-1 * self.sort_config.direction_threshold))):
                    direction_track1 = 8
                else:
                    direction_track1 = -1

                x_diff = track_box2[0] - bbox2[0]
                y_diff = track_box2[1] - bbox2[1]
                if(((x_diff <= self.sort_config.direction_threshold) and (x_diff >= (-1 * self.sort_config.direction_threshold))) and ((y_diff <= self.sort_config.direction_threshold) and (y_diff >= (-1 * self.sort_config.direction_threshold)))):
                    direction_track2 = 0
                elif((x_diff >= self.sort_config.direction_threshold) and ((y_diff <= self.sort_config.direction_threshold) and (y_diff >= (-1 * self.sort_config.direction_threshold)))):
                    direction_track2 = 1
                elif((x_diff >= self.sort_config.direction_threshold) and (y_diff >= self.sort_config.direction_threshold)):
                    direction_track2 = 2
                elif(((x_diff <= self.sort_config.direction_threshold) and (x_diff >= (-1 * self.sort_config.direction_threshold))) and (y_diff >= self.sort_config.direction_threshold)):
                    direction_track2 = 3
                elif((x_diff <= (-1 * self.sort_config.direction_threshold)) and (y_diff >= self.sort_config.direction_threshold)):
                    direction_track2 = 4
                elif((x_diff <= (-1 * self.sort_config.direction_threshold)) and ((y_diff <= self.sort_config.direction_threshold) and (y_diff >= (-1 * self.sort_config.direction_threshold)))):
                    direction_track2 = 5
                elif((x_diff <= (-1 * self.sort_config.direction_threshold)) and (y_diff <= (-1 * self.sort_config.direction_threshold))):
                    direction_track2 = 6
                elif(((x_diff <= self.sort_config.direction_threshold) and (x_diff >= (-1 * self.sort_config.direction_threshold))) and (y_diff <= (-1 * self.sort_config.direction_threshold))):
                    direction_track2 = 7
                elif((x_diff >= self.sort_config.direction_threshold) and (y_diff <= (-1 * self.sort_config.direction_threshold))):
                    direction_track2 = 8
                else:
                    direction_track2 = -1

        for i, swap in enumerate(self.swap_targets):
            if frm_count >= (swap[0] + 500):
                to_del.append(i)
        for i, ele in enumerate(sorted(to_del, reverse=True)):
            if len(self.swap_targets) > i:
                del self.swap_targets[i]
        close_tracks_list = []
        for trk in reversed(list(self.trackers.values())):
            d, playerType, highlight = trk.get_state()
            if trk.time_since_update < 1:
                ret.append(np.concatenate((d[0], [trk.id+1])).reshape(1, -1))
                if trk.close_ids != []:
                    close_tracks_list.append(str(trk.id))
                # direction vector
                centroid = (int(int(d[0][0] + d[0][2])/2),
                            int(int(d[0][1] + d[0][3])/2))

                if len(trk.tracklets) >= 40:
                    trk.tracklets.pop(0)
                trk.tracklets.append(centroid)
                if len(trk.tracklets) == 40:
                    x_diff = trk.tracklets[39][0] - trk.tracklets[0][0]
                    y_diff = trk.tracklets[39][1] - trk.tracklets[0][1]
                    if(((x_diff < self.sort_config.direction_threshold) and (x_diff > (-1 * self.sort_config.direction_threshold))) and ((y_diff < self.sort_config.direction_threshold) and (y_diff > (-1 * self.sort_config.direction_threshold)))):
                        trk.direction = 0
                    elif((x_diff > self.sort_config.direction_threshold) and ((y_diff < self.sort_config.direction_threshold) and (y_diff > (-1 * self.sort_config.direction_threshold)))):
                        trk.direction = 1
                    elif((x_diff > self.sort_config.direction_threshold) and (y_diff > self.sort_config.direction_threshold)):
                        trk.direction = 2
                    elif(((x_diff < self.sort_config.direction_threshold) and (x_diff > (-1 * self.sort_config.direction_threshold))) and (y_diff > self.sort_config.direction_threshold)):
                        trk.direction = 3
                    elif((x_diff < (-1 * self.sort_config.direction_threshold)) and (y_diff > self.sort_config.direction_threshold)):
                        trk.direction = 4
                    elif((x_diff < (-1 * self.sort_config.direction_threshold)) and ((y_diff < self.sort_config.direction_threshold) and (y_diff > (-1 * self.sort_config.direction_threshold)))):
                        trk.direction = 5
                    elif((x_diff < (-1 * self.sort_config.direction_threshold)) and (y_diff < (-1 * self.sort_config.direction_threshold))):
                        trk.direction = 6
                    elif(((x_diff < self.sort_config.direction_threshold) and (x_diff > (-1 * self.sort_config.direction_threshold))) and (y_diff < (-1 * self.sort_config.direction_threshold))):
                        trk.direction = 7
                    elif((x_diff > self.sort_config.direction_threshold) and (y_diff < (-1 * self.sort_config.direction_threshold))):
                        trk.direction = 8
                else:
                    trk.direction = -1

                highlights.append(highlight)
                player_type.append(playerType)
                directions.append(trk.direction)
                highlight_streaks.append(trk.highlight_streak)

            i -= 1
            # remove dead tracklet
            if trk.time_since_update > self.max_age:
                # if str(trk.id) in self.trackers.keys():
                del self.trackers[str(trk.id)]
                if trk.id in unmatched_trks:
                    itemindex = np.where(unmatched_trks == trk.id)
                    unmatched_trks = np.delete(unmatched_trks, itemindex)
                if trk.id in matched:
                    itemindex = list(np.where(matched == trk.id)[0])[0]
                    matched = np.delete(matched, itemindex, 0)

        # close ids
        for trk in self.trackers.values():
            if trk.time_since_update < 1:
                trk.close_ids = []
                for close_trks in self.trackers.values():
                    if(trk.id != close_trks.id and close_trks.time_since_update < 1):
                        if(len(trk.tracklets) > 1 and len(close_trks.tracklets) > 1):
                            centroid1 = trk.tracklets[len(trk.tracklets) - 1]
                            centroid2 = close_trks.tracklets[len(
                                close_trks.tracklets) - 1]
                            if((abs(centroid2[0] - centroid1[0]) < self.close_id_threshold) and (abs(centroid2[1] - centroid1[1]) < self.close_id_threshold)):
                                trk.close_ids.append(close_trks.id)
                if trk.is_merged:
                    trk.is_merged = False

                    for close_id in trk.temp_close_ids:
                        if close_id in trk.close_ids:
                            trk.temp_close_ids = []
                            bbox1 = trk.get_state()[0][0][:4]
                            bbox2 = self.trackers[str(close_id)].get_state()[
                                0][0][:4]
                            temp_list = []
                            temp_list.append(
                                frm_count+self.sort_config.mv_threshold)
                            temp_list.append(trk.id)
                            temp_list.append(trk.direction)
                            temp_list.append(bbox1)
                            temp_list.append(close_id)
                            temp_list.append(
                                self.trackers[str(close_id)].direction)
                            temp_list.append(bbox2)
                            self.swap_targets.append(temp_list)

        if len(ret) > 1:
            rets = np.concatenate(ret)

            bbox_xyxy = rets[:, :4]
            identities = rets[:, 8]
            player_types = player_type.copy()

            # homography
            for i, bbox in enumerate(bbox_xyxy):
                id = int(identities[i]) if identities is not None else 0
                obj = [int(bbox[0])+10, int(bbox[1])+10,
                       int(bbox[2])-10, int(bbox[3])-10]
                if self.sort_config.lens_distortion_flag == 1:
                    homo_pt = calculate_homography(obj=obj, cam_matrix=self.sort_config.cam_matrix, lens_distortion_flag=self.sort_config.lens_distortion_flag,
                                                   newcameramtx=self.sort_config.newcameramtx, lens_dist=self.sort_config.lens_dist)
                else:
                    homo_pt = calculate_homography(
                        obj=obj, cam_matrix=self.sort_config.cam_matrix)
                homo_pt = (int(homo_pt[0][0][0]),
                           int(homo_pt[0][0][1]))
                x1 = homo_pt[0]
                y1 = homo_pt[1]

                if id not in homo_track.keys():
                    homo_track[id] = [x1, y1]
                else:
                    if abs(homo_track[id][0] - x1) < 7:
                        x1 = homo_track[id][0]
                    else:
                        homo_track[id][0] = x1
                    if abs(homo_track[id][1] - y1) < 7:
                        y1 = homo_track[id][1]
                    else:
                        homo_track[id][1] = y1
                    push_data["players"].append({
                        "identities": id,
                        "bbox": (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])),
                        "player_type": player_types[i],
                        "direction": directions[i],
                        "homo_track": (x1, y1)
                    }
                    )
                    downstream_data["players"][str(id)] = {
                        "identities": id,
                        "bbox": (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])),
                        "player_type": player_types[i],
                        "direction": directions[i],
                        "homo_track": (x1, y1)
                    }

            np_ids = np.array(identities)
            np_ptype = np.array(player_types)

            # umpire_index = np.where(np_ids == self.umpire_id)
            # umpire_index = list(umpire_index[0])

            umpire_index = []
            for id in self.umpire_id:
                if(id in np_ids):
                    umpire_index.append(np.where(np_ids == id)[0][0])
            
            
            
            # print(player_types)
            green_index = np.where(np.logical_or(np_ptype == 0 , np_ptype == 9))
            red_index = list(np.where(np_ptype == 3)[0])
            green_index = list(green_index[0])
            

            # red_index = list(np.where(np_ptype == 3)[0])
            # green_index = list(green_index[0])
            copy_of_green_index = green_index.copy()
            
            pink_index = list(np.where(np_ptype == 1)[0])
            # df= pd.DataFrame()
            # for i in range(len(np_ptype)):
            #     if np_ptype[i] == 0:
            #         print("here",np_ptype)
            #         for j in range(i+1,len(np_ptype)):
            #             if np_ptype[j] == 3:
            #                 continue
            #             if self.get_distance(bbox_xyxy[np_ptype[i]], np_ptype[j]) < 30:
            #                 if len(self.occl_ids) ==0:
            #                     if np_ptype[i] == np_ptype[j]:
            #                         pass
            #                     else:
            #                         self.occl_ids[frm_count] = tuple([int(identities[i]),int(np_ptype[i]),int(identities[j]),int(np_ptype[j])])
            #                 else:
            #                     for frm in self.occl_ids.keys():
            #                         ids = self.occl_ids[frm][0]
            #                         clsses = self.occl_ids[frm][1]
            #                         if identities[i] in ids:
            #                             self.occl_ids[frm][0].append(int(identities[j]))
            #                             self.occl_ids[frm][1].append(int(np_ptype[j]))
            #                         elif identities[j] in ids:
            #                             self.occl_ids[frm][0].append(int(identities[i]))
            #                             self.occl_ids[frm][1].append(int(np_ptype[i]))
            #                         else:
            #                             pass
            #     elif np_ptype[i] == 1:
            #         for j in range(len(np_ptype)):
            #             if np_ptype[j] ==2:
            #                 if self.get_distance(bbox_xyxy[np_ptype[i]], bbox_xyxy[np_ptype[j]]) < 30:
            #                     if len(self.occl_ids) ==0:
            #                         self.occl_ids[frm_count] = tuple([int(identities[i]),int(np_ptype[i]),int(identities[j]),int(np_ptype[j])])
            #                     else:
            #                         for frm in self.occl_ids.keys():
            #                             ids = self.occl_ids[frm][0]
            #                             clsses = self.occl_ids[frm][1]
            #                             if identities[i] in ids:
            #                                 self.occl_ids[frm][0].append(int(identities[j]))
            #                                 self.occl_ids[frm][1].append(int(np_ptype[j]))
            #                             elif identities[j] in ids:
            #                                 self.occl_ids[frm][0].append(int(identities[i]))
            #                                 self.occl_ids[frm][1].append(int(np_ptype[i]))
            #                             else:
            #                                 pass

            # print(self.occl_ids)
            # blue_index = list(np.where(np_ptype == 2)[0]) 

            ########################
            # remove 1 id - blue id
            ########################
            self.check_occlusions(identities,bbox_xyxy,frm_count,np_ptype)

            # for i in range(len(green_index)):
            #     green_corr = bbox_xyxy[green_index[i]]
            #     green_id = identities[green_index[i]]
                
            #     x11, y11, x12, y12 = [int(i) for i in green_corr]

            #     per = self.img[y11:y12,x11:x12]
            #     # cv2.imwrite("Track/per"+str(int(len(os.listdir('Track'))))+".jpg",per)
            #     pil_image = Image.fromarray(cv2.cvtColor(per, cv2.COLOR_BGR2RGB))
            #     img_tensor = self.data_transforms(pil_image)
            #     input_img = Variable(img_tensor.float().cuda())
            #     output = self.feature_extractor_cnn_block(input_img.unsqueeze(0))
            #     per_features = pd.DataFrame(self.feature_extractor_add_block(output.reshape(1, 2048)).detach().cpu())
            #     per_features['id'] = green_id
            #     self.features_df = pd.concat([self.features_df,per_features])
            #     self.features_df.to_csv
            #     print(self.features_df.shape[0]%1000)
            #     if self.features_df.shape[0]%1000 ==0:
            #         self.features_df.to_csv('Track/features.csv',index=False)    

            #     for itr, index in enumerate(umpire_index):
            #         umpire_corr = bbox_xyxy[index]
            #         green_ump_dist = self.get_distance(green_corr, umpire_corr)
            #         # print(green_ump_dist,self.umpire_id[itr],green_id)
                    
            #         if green_ump_dist < 30:
            #             # print(green_ump_dist,green_id,self.umpire_id[itr])
            #             self.swap_dict["green_id"].append(green_id)
            #             self.swap_dict["red_id"].append(
            #                 self.umpire_id[itr])
            #             self.swap_dict['flag_inswap'].append(False)
            #             self.swap_dict['flag_normal'].append(False)
            #             self.swap_dict['pot_swap'].append(True)
            #             self.swap_dict['temp_fr'].append(0)
            #             self.swap_dict['cent_cor_x'].append(0)
            #             self.swap_dict['cent_cor_y'].append(0)
            #             self.swap_dict['past_dist'].append(0)
            #             self.swap_dict['oc_frame'].append(0)
            #             self.swap_dict['in_pitch_box'].append(False)
            #             # print("Potential umpire occlusion ids: ",
            #                     # green_id, self.umpire_id[itr])
            #             try:
            #                 # print(tuple([int(green_id), int(self.umpire_id[itr])]) not in self.occlusion_dict.values())
                            
            #                 if (int(self.umpire_id[itr]) in self.occlusion_ids) and (int(green_id) not in self.occlusion_ids): 
            #                     # print("here")
            #                     for frm in self.occlusion_dict.keys():
            #                         present_occ_ids = self.occlusion_dict[frm][0]
            #                         if self.umpire_id[itr] in present_occ_ids:
            #                             self.occlusion_dict[frm][0].append(int(green_id))
            #                             self.occlusion_dict[frm][1].append(int(0))
            #                             self.occlusion_ids.append(int(green_id))
            #                 else:
            #                     # print((int(self.umpire_id[itr]) not in self.occlusion_ids) and (int(green_id) not in self.occlusion_ids))
            #                     if (int(self.umpire_id[itr]) not in self.occlusion_ids) and (int(green_id) not in self.occlusion_ids):
            #                         if tuple([[int(green_id),int(self.umpire_id[itr])],[int(0),int(2)]]) not in self.occlusion_dict.values():
            #                             self.occlusion_dict[frm_count] = tuple([[int(green_id),int(self.umpire_id[itr])],[int(0),int(2)]])
            #                             self.occlusion_ids.append(int(self.umpire_id[itr]))
            #                             self.occlusion_ids.append(int(green_id))     
            #             except:
            #                 pass
                        
                 
                
            #     for itr_b, index_b in enumerate(pink_index):
                    
            #         bat_corr = bbox_xyxy[index_b]
            #         green_bat_dist = self.get_distance(green_corr, bat_corr)
                    
            #         if green_bat_dist < 30:
            #             # print(green_bat_dist,green_id,self.batsmen_ids[itr_b])
            #             self.swap_dict["green_id"].append(green_id)
            #             self.swap_dict["red_id"].append(
            #                 identities[index_b])
            #             self.swap_dict['flag_inswap'].append(False)
            #             self.swap_dict['flag_normal'].append(False)
            #             self.swap_dict['pot_swap'].append(True)
            #             self.swap_dict['temp_fr'].append(0)
            #             self.swap_dict['cent_cor_x'].append(0)
            #             self.swap_dict['cent_cor_y'].append(0)
            #             self.swap_dict['past_dist'].append(0)
            #             self.swap_dict['oc_frame'].append(0)
            #             self.swap_dict['in_pitch_box'].append(False)
            #             # print("Potential batsmen occlusion ids: ",
            #                     # green_id, identities[index_b])
            #             try:

            #                 if (int(identities[index_b]) in self.occlusion_ids) and (int(green_id) not in self.occlusion_ids): 
            #                     for frm in self.occlusion_dict.keys():
            #                         present_occ_ids = self.occlusion_dict[frm][0]
            #                         if int(identities[index_b]) in present_occ_ids:
            #                             self.occlusion_dict[frm][0].append(int(green_id))
            #                             self.occlusion_dict[frm][1].append(int(0))
            #                             self.occlusion_ids.append(int(green_id))
            #                 else:
            #                     if (int(identities[index_b]) not in self.occlusion_ids) and (int(green_id) not in self.occlusion_ids):
            #                         if tuple([[int(green_id),int(identities[index_b])],[int(0),int(1)]]) not in self.occlusion_dict.values():    
            #                             self.occlusion_dict[frm_count] = tuple([[int(green_id),int(identities[index_b])],[int(0),int(1)]])
            #                             self.occlusion_ids.append(int(identities[index_b]))
            #                             self.occlusion_ids.append(int(green_id))
            #             except:
            #                 pass
            #             # if len(self.occlusion_dict) ==0:
            #             #     self.occlusion_dict[frm_count] = [[int(green_id),int(self.umpire_id[itr]],[int(0),int(2)]]
            #             # else:
            #             #     if [[int(green_id),int(self.umpire_id[itr]],[int(0),int(2)]] not in self.occlusion_dict[frm_count].values():
            #             #         if int(self.umpire_id[itr] in self.occlusion_dict[frm_count].values()[0]:
            #             #             self.occlusion_dict[frm_count][0].append(int(green_id))
            #             #             self.occlusion_dict[frm_count][1].append(int(0))
                
            # for itr_b, index_b in enumerate(pink_index):
            #     green_corr = bbox_xyxy[index_b]
            #     green_id = identities[index_b]
            #     for itr, index in enumerate(umpire_index):
            #         umpire_corr = bbox_xyxy[index]
            #         green_ump_dist = self.get_distance(green_corr, umpire_corr)
            #         if green_ump_dist < 30:
            #             # print(green_ump_dist,green_id,self.umpire_id[itr])
            #             self.swap_dict["green_id"].append(green_id)
            #             self.swap_dict["red_id"].append(
            #                 self.umpire_id[itr])
            #             self.swap_dict['flag_inswap'].append(False)
            #             self.swap_dict['flag_normal'].append(False)
            #             self.swap_dict['pot_swap'].append(True)
            #             self.swap_dict['temp_fr'].append(0)
            #             self.swap_dict['cent_cor_x'].append(0)
            #             self.swap_dict['cent_cor_y'].append(0)
            #             self.swap_dict['past_dist'].append(0)
            #             self.swap_dict['oc_frame'].append(0)
            #             self.swap_dict['in_pitch_box'].append(False)
            #             # print("Potential umpire occlusion ids: ",
            #             #         green_id, self.umpire_id[itr])
            #             try:
            #                 # print(tuple([int(green_id), int(self.umpire_id[itr])]) not in self.occlusion_dict.values())
            #                 if (int(self.umpire_id[itr]) not in self.occlusion_ids) and (int(green_id) not in self.occlusion_ids):
            #                     if tuple([[int(green_id),int(self.umpire_id[itr])],[int(1),int(2)]]) not in self.occlusion_dict.values():
            #                         self.occlusion_dict[frm_count] = tuple([[int(green_id),int(self.umpire_id[itr])],[int(1),int(2)]])
            #                         self.occlusion_ids.append(int(self.umpire_id[itr]))
            #                         self.occlusion_ids.append(int(green_id))
            #             except:
            #                 pass



                # if len(copy_of_green_index) > 1:
                    
                #     copy_of_green_index.remove(green_index[i])
                    
                #     for itr_g, index_g in enumerate(copy_of_green_index):
                #         other_green_corr = bbox_xyxy[index_g]
                #         green_bat_dist = self.get_distance(green_corr, other_green_corr)
                #         if green_id not in self.swap_dict["green_id"]:
                #             if green_bat_dist < 30:
                #                 # print(green_bat_dist,green_id,identities[index_g])
                #                 self.swap_dict["green_id"].append(green_id)
                #                 self.swap_dict["red_id"].append(
                #                     identities[index_g])
                #                 self.swap_dict['flag_inswap'].append(False)
                #                 self.swap_dict['flag_normal'].append(False)
                #                 self.swap_dict['pot_swap'].append(True)
                #                 self.swap_dict['temp_fr'].append(0)
                #                 self.swap_dict['cent_cor_x'].append(0)
                #                 self.swap_dict['cent_cor_y'].append(0)
                #                 self.swap_dict['past_dist'].append(0)
                #                 self.swap_dict['oc_frame'].append(0)
                #                 self.swap_dict['in_pitch_box'].append(False)
                #                 # print("Potential green occlusion ids: ",
                #                 #     green_id, identities[index_g])
                #                 # print([tuple(int(green_id), int(identities[index_g])]),self.occlusion_dict.values())
                #                 try:
                #                     # print(tuple(int(green_id), int(identities[index_g])),self.occlusion_dict.values())
                #                     if tuple([int(green_id), int(identities[index_g])]) not in self.occlusion_dict.values():    
                #                         self.occlusion_dict[frm_count] = tuple([int(green_id), int(identities[index_g])])
                #                 except:
                #                     pass
                            
                # for j in range(len(red_index)):
                #     red_corr = bbox_xyxy[red_index[j]]
                #     red_green_dist = self.get_distance(green_corr, red_corr)

                #     if red_green_dist < 30:
                #         red_id = identities[red_index[j]]
                #         if green_id not in self.swap_dict["green_id"]:
                #             temp_homo_pt = homo_track[green_id]
                #             if((temp_homo_pt[0] > self.sort_config.crop_u1_x1) and (temp_homo_pt[0] < self.sort_config.crop_u1_x2) and (temp_homo_pt[1] > self.sort_config.crop_u1_y1) and (temp_homo_pt[1] < self.sort_config.crop_u1_y2)):

                #                 self.swap_dict["green_id"].append(green_id)
                #                 self.swap_dict["red_id"].append(red_id)
                #                 self.swap_dict['flag_inswap'].append(False)
                #                 self.swap_dict['flag_normal'].append(False)
                #                 self.swap_dict['pot_swap'].append(True)
                #                 self.swap_dict['temp_fr'].append(0)
                #                 self.swap_dict['cent_cor_x'].append(0)
                #                 self.swap_dict['cent_cor_y'].append(0)
                #                 self.swap_dict['past_dist'].append(0)
                #                 self.swap_dict['oc_frame'].append(0)
                #                 # print("inside the pitch region")

                #                 self.swap_dict['in_pitch_box'].append(True)
                #                 # print("Potential occlusion ids: ",
                #                 #         green_id, red_id)
                #                 try:
                #                     if tuple(int(green_id), int(red_id)) not in self.occlusion_dict.values():    
                #                         self.occlusion_dict[frm_count] = tuple([int(green_id), int(red_id)])
                #                 except:
                #                     pass  
            
            # self.check(bbox_xyxy,identities,frm_count)

        params.append(highlights)
        params.append(player_type)
        params.append(directions)
        params.append(highlight_streaks)
        params.append(close_tracks_list)

        if len(ret) > 0:
            return np.concatenate(ret), params, homo_track, push_data, downstream_data
        return np.empty((0, 6)), params, homo_track, push_data, downstream_data