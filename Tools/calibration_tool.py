###########################################################################################################################################
"""
The calibration Ui currently has 5 modules :
1. Crop module  : Crops out the region of intrest (cuts out stadium and other parts such that maximum of the image is filled with ground only). 
                  Saves up these coordinates for the main app to use.
2. Umpire region module : Marking region of the primary umpire
3. Segmentation module : After cropping(or without cropping), selecting points along the boundary to draw out an ellipse masking out areas 
                         outside the ellipse(boundary)
# 4. OCR Module : For marking out regions containing batsmen names, overs and selecting on strike batsman.
#                 Processes the regions with OCR and display raw results and post processed results
5. Hyperparameter Tuning : Set values for Yolo and SORT parameters, option to reset to defaults included

"""
############################################### IMPORTS  #####################################################################################

import json
# General imports
import sys
import time
import tkinter
from pickle import FALSE, NONE, TRUE
from threading import Lock
import os 

import cv2
import numpy as np
from PyQt5 import QtCore, QtGui
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QDoubleValidator, QFont, QIntValidator, QPixmap
# imports related to UI
from PyQt5.QtWidgets import (QApplication, QComboBox, QHBoxLayout, QLabel,
                             QLineEdit, QMessageBox, QPushButton, QSizePolicy,QRadioButton,
                             QVBoxLayout, QWidget)

with open("../Settings/config.json", 'r') as f:
    config_dict = json.load(f)
    cam_model = config_dict['camera_model']
    # db_name = config_dict['db_name']
    lens_distortion_flag = config_dict["lens_distortion"]

if lens_distortion_flag == 1:
    mtx1 = [[1760.5563, 0, 2104.0234], [0, 1772.3534, 1059.7439], [0, 0, 1]]
    dist1 = [-0.2906, 0.1071, 0.0011, 0.0008, -0.0206]
    lens_mtx = np.array(mtx1)
    lens_dist = np.array(dist1)
    w, h = 3840, 2160
    newcameramtx, lens_roi = cv2.getOptimalNewCameraMatrix(
        lens_mtx, lens_dist, (w, h), 0, (w, h))


with open("../Settings/cam_params.json", 'r') as json_file:
    camera_params = json.load(json_file)
    cam_matrix = np.array(camera_params['CAM'])


def convert_coordinates(x, y, img_size, frame_size):
    """
    To convert coordinates obtained from mouse click thats in the scale of image displayed to original image's size 
    """
    resized_x = int((x * frame_size[0])/img_size[0])
    resized_y = int((y * frame_size[1])/img_size[1])
    return resized_x, resized_y


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    """
    image labels's image display based on module it currently is in and drawing rectangles to indicate cropped areas
    """

    def __init__(self):
        super(VideoThread, self).__init__()

        self.field_frame_name = "../Settings/frame.jpg"
        self.src_img_original = cv2.imread(self.field_frame_name, -1)
        self.src_img = self.src_img_original.copy()
        self.src_umpire_region = cv2.imread("../Settings/dst.jpg")
        self.src_image_current = self.src_img.copy()

        self.crop_image_module = True
        self.pitch_region_marking = False
        self.pitch_image = cv2.imread("../Settings/dst.jpg", -1)
        self.stream_area_width = 1510
        self.stream_area_height = 1080
        self.img_size = [1510, 1080]
        self.gap_module = False
        self.gap_points = []
        self.Segmentation_module = False
        self.segmentation_done = False
        self.i = 0
        self.key_drawEllipse = False
        ht, wd, c = cv2.imread(self.field_frame_name).shape
        self.frame_size = [wd, ht]
        self.drawRectCoords = []
        self.drawRect = False
        self.segmentation_points_list = []
        self.save_crop_coordinates = False
        self.ocr_module = False
        self.key_ellipse = False
        self.save_segmentation = False

    def run(self):  # sourcery no-metrics
        while(True):
            """
            1. crop image module is to draw a rectangle of region aof intrest ad save the coords
            2. umpire region is to mark the region where umpire ll be
            src_image_current : The image to be displayed n Ui
            """
            if(self.crop_image_module == True):
                if(self.pitch_region_marking):
                    cv_img = self.pitch_image.copy()
                else:
                    cv_img = self.src_image_current.copy()

                if((self.drawRect == True) and (self.drawRectCoords != []) and (self.save_crop_coordinates == False)):
                    try:
                        cv_img = cv2.resize(
                            cv_img, (self.stream_area_width, self.stream_area_height))
                        cv2.rectangle(cv_img, (self.drawRectCoords[0], self.drawRectCoords[1]), (
                            self.drawRectCoords[2], self.drawRectCoords[3]), (0, 255, 0), 3)
                        self.change_pixmap_signal.emit(cv_img)
                    except:
                        print("Due to thread issue this error is being skipped!")
                        pass

                elif((self.drawRect == True) and (self.drawRectCoords != []) and (self.save_crop_coordinates == True)):
                    self.save_crop_coordinates = False
                    # convert coords from display size to atual image size
                    x1, y1 = convert_coordinates(self.drawRectCoords[0], self.drawRectCoords[1], self.img_size, [
                                                 cv_img.shape[1], cv_img.shape[0]])
                    x2, y2 = convert_coordinates(self.drawRectCoords[2], self.drawRectCoords[3], self.img_size, [
                                                 cv_img.shape[1], cv_img.shape[0]])
                    # crop the image and change the src_image_current so that it is the image to be displayed
                    # as well as used in processing for segmentation
                    image = cv_img[y1:y2, x1:x2, :]
                    cv2.imwrite("../Settings/src.jpg",image)
                    self.frame_size = [image.shape[1], image.shape[0]]
                    self.src_image_current = image
                    self.drawRect = False
                    self.drawRectCoords = []
                    cv_img = cv2.resize(
                        image, (self.stream_area_width, self.stream_area_height))
                    self.change_pixmap_signal.emit(cv_img)
                else:
                    cv_img = cv2.resize(
                        cv_img, (self.stream_area_width, self.stream_area_height))
                    self.change_pixmap_signal.emit(cv_img)

            elif(self.gap_module == True):
                """
                Marking both the wicket points
                """
                cv_img = self.src_image_current.copy()
                h, w, c = cv_img.shape
                if(len(self.gap_points) > 0):
                    for point in self.gap_points:
                        x, y = point
                        cv2.circle(cv_img, (x, y), 5, (255, 0, 0), -1)
                cv_img = cv2.resize(
                    cv_img, (self.stream_area_width, self.stream_area_height))
                self.change_pixmap_signal.emit(cv_img)

            elif(self.Segmentation_module == True):
                """
                Marks points along boundary and draws an ellipse masking ot the boundary

                """

                cv_img = self.src_image_current.copy()
                h, w, c = cv_img.shape
                frame_size = [w, h]

                if self.key_ellipse == True and self.segmentation_done == False:
                    self.key_ellipse = False
                    self.segmentation_points_list = []
                    self.i += 1
                    print(
                        "Select the {i}th point for ellipse".format(i=self.i))
                    for pts in self.segmentation_points_list:
                        [x, y] = pts
                        cv2.circle(cv_img, (x, y), 5, (0, 0, 255), -1)
                    cv_img = cv2.resize(
                        cv_img, (self.stream_area_width, self.stream_area_height))
                    self.change_pixmap_signal.emit(cv_img)

                elif self.key_drawEllipse == True:
                    el_array = np.asarray(self.segmentation_points_list)
                    if(len(el_array) >= 7):
                        ellipse = cv2.fitEllipse(el_array)

                        mask = np.zeros_like(cv_img)
                        mask = cv2.ellipse(mask, ellipse, (255, 255, 255), -1)
                        cv_img = cv2.bitwise_and(cv_img, mask)
                        if(self.save_segmentation == True):
                            self.save_segmentation = False
                            cv2.imwrite(
                                "../Settings/segmentation_mask.jpg", mask)

                    cv_img = cv2.resize(
                        cv_img, (self.stream_area_width, self.stream_area_height))
                    self.change_pixmap_signal.emit(cv_img)

                else:
                    if(self.segmentation_done == False):
                        for pts in self.segmentation_points_list:
                            [x, y] = pts
                            cv2.circle(cv_img, (x, y), 8, (0, 0, 255), -1)
                    cv_img = cv2.resize(
                        cv_img, (self.stream_area_width, self.stream_area_height))
                    self.change_pixmap_signal.emit(cv_img)


class App(QWidget):

    def __init__(self):
        super().__init__()
        # create the video capture thread
        self.thread = VideoThread()
        # global ocr_crop_data
        # createJsonFile()

        self.showMaximized()
        self.setWindowState(QtCore.Qt.WindowMaximized)

        root = tkinter.Tk()
        self.screen_resolution_width = root.winfo_screenwidth()
        self.screen_resolution_height = root.winfo_screenheight()
        self.font_size = 40
        self.mutex = Lock()
        self.drawing = False
        self.initial_x, self.initial_y = -1, -1
        self.InitialiseScreenSettings()

        ### Window Settings ###
        self.setWindowTitle("Quidich Tracker : Calibration tool")
        self.disply_width = self.screen_resolution_width
        self.display_height = self.screen_resolution_height
        self.setStyleSheet("QWidget {background-color: #151515;}")

        ### Create the label that holds the image ###
        self.image_label = QLabel(self)
        self.image_label.resize(self.disply_width, self.display_height)
        self.image_label.mousePressEvent = self.mousePressEvent
        self.image_label.mouseMoveEvent = self.mouseMoveEvent
        self.image_label.mouseReleaseEvent = self.mouseReleaseEvent

        ### PANEL LAYOUT ###

        ### CROP MODULE ###

        self.button_crop = QPushButton('Crop Footage', self)
        self.button_crop.setStyleSheet(
            "background-color: #2234a8;color:black;")
        self.button_crop.setFont(QFont('Arial', self.font_size))
        ht = self.button_crop.frameGeometry().height()
        self.button_crop.setMinimumHeight(int(1.4*ht))
        self.button_crop.clicked.connect(
            lambda: self.button_cropimage_clicked())

        self.button_savecropcoord = QPushButton('Save Crop Coordinates', self)
        self.button_savecropcoord.setStyleSheet(
            "background-color: #2234a8;color:black;")
        self.button_savecropcoord.setFont(QFont('Arial', self.font_size))
        ht = self.button_savecropcoord.frameGeometry().height()
        self.button_savecropcoord.setMinimumHeight(int(1.2*ht))
        self.button_savecropcoord.clicked.connect(
            lambda: self.button_savecropcoord_clicked())

        ### Umpire region selection ###

        self.button_pitchcoords = QPushButton('Pitch Coordinates', self)
        self.button_pitchcoords.setStyleSheet(
            "background-color: #2234a8;color:black;")
        self.button_pitchcoords.setFont(QFont('Arial', self.font_size))
        ht = self.button_pitchcoords.frameGeometry().height()
        self.button_pitchcoords.setMinimumHeight(int(1.2*ht))
        self.button_pitchcoords.clicked.connect(
            lambda: self.button_pitchcoords_clicked())

        self.button_savecropcoord_pitch = QPushButton(
            'Save Pitch Coordinates', self)
        self.button_savecropcoord_pitch.setStyleSheet(
            "background-color: #2234a8;color:black;")
        self.button_savecropcoord_pitch.setFont(QFont('Arial', self.font_size))
        ht = self.button_savecropcoord_pitch.frameGeometry().height()
        self.button_savecropcoord_pitch.setMinimumHeight(int(1.2*ht))
        self.button_savecropcoord_pitch.clicked.connect(
            lambda: self.button_savecropcoord_pitch_clicked())

        ### GAP MODULE ###

        self.button_gap_points = QPushButton('Gap module', self)
        self.button_gap_points.setStyleSheet(
            "background-color: #2234a8;color:black;")
        self.button_gap_points.setFont(QFont('Arial', self.font_size))
        ht = self.button_gap_points.frameGeometry().height()
        self.button_gap_points.setMinimumHeight(int(1.2*ht))
        self.button_gap_points.clicked.connect(
            lambda: self.button_gap_points_clicked())

        self.button_save_gap_points = QPushButton(
            'Save Crop Coordinates', self)
        self.button_save_gap_points.setStyleSheet(
            "background-color: #2234a8;color:black;")
        self.button_save_gap_points.setFont(QFont('Arial', self.font_size))
        ht = self.button_save_gap_points.frameGeometry().height()
        self.button_save_gap_points.setMinimumHeight(int(1.2*ht))
        self.button_save_gap_points.clicked.connect(
            lambda: self.button_save_gap_points_clicked())

        ### SEGMENTATION MODULE ###

        self.button_segmentation = QPushButton('Segmentation', self)
        self.button_segmentation.setStyleSheet(
            "background-color: #2234a8;color:black")
        self.button_segmentation.setFont(QFont('Arial', self.font_size))
        ht = self.button_segmentation.frameGeometry().height()
        self.button_segmentation.setMinimumHeight(int(1.4*ht))
        self.button_segmentation.clicked.connect(
            lambda: self.button_segmentation_clicked())

        hl_segmentation = QHBoxLayout()
        self.button_startselect = QPushButton('Select Points', self)
        self.button_startselect.setStyleSheet(
            "background-color: #2234a8;color:black")
        self.button_startselect.setFont(QFont('Arial', self.font_size))
        ht = self.button_startselect.frameGeometry().height()
        self.button_startselect.setMinimumHeight(int(1.2*ht))
        self.button_startselect.clicked.connect(
            lambda: self.button_startselect_clicked())

        self.button_displaysegimg = QPushButton('View Mask', self)
        self.button_displaysegimg.setStyleSheet(
            "background-color: #2234a8;color:black")
        self.button_displaysegimg.setFont(QFont('Arial', self.font_size))
        ht = self.button_displaysegimg.frameGeometry().height()
        self.button_displaysegimg.setMinimumHeight(int(1.2*ht))
        self.button_displaysegimg.clicked.connect(
            lambda: self.button_displaysegimg_clicked())

        hl_segmentation.addWidget(self.button_startselect)
        hl_segmentation.addWidget(self.button_displaysegimg)

        ### HYPERPARAMETER TUNING ###
        with open("../Settings/hyperparms.json", "r") as f:
            param_dict = json.load(f)




        self.weights = QComboBox()
        self.weights.setStyleSheet("background-color: #2234a8;color:black;selection-color: black;selection-background-color: white;")
        self.weights.setFont(QFont('Arial', self.font_size))
        self.weights.setMinimumHeight(int(1.4*ht))
        self.weights.addItem("Select Weight")
        self.weight_list = self.folder_list_with_extension()
        for item in self.weight_list:
            self.weights.addItem(item)    
        self.weights.currentIndexChanged.connect(self.weights_change)

        hl_weight = QHBoxLayout()
        hl_weight.addWidget(self.weights)


        self.lens = QRadioButton('Lens Distortion')
        self.lens.setStyleSheet("background-color: #2234a8;color:black;selection-color: black;selection-background-color: white;")
        self.lens.setFont(QFont('Arial', self.font_size*1.2))
        self.lens.setMinimumHeight(int(1.4*ht))
        self.lens.clicked.connect(lambda: self.lensButtonClicked())


        hl_lens = QHBoxLayout()
        hl_lens.addWidget(self.lens)

        with open("../Settings/config.json", 'r') as f:
            config_dict = json.load(f)
        lens_distortion = config_dict['lens_distortion'] 
        if lens_distortion ==0:
            self.lens.setChecked(False)
        elif lens_distortion ==1:
            self.lens.setChecked(True)



        # Setting Yolo confidence value

        hl_conf = QHBoxLayout()
        self.input_conf = QLineEdit(self)
        dv = QDoubleValidator(0.0, 5.0, 2)
        self.input_conf.setValidator(dv)
        self.input_conf.setStyleSheet("background-color:white;color:black")
        self.input_conf.setMinimumHeight(int(1.3*ht))

        self.input_conf_display = QLineEdit(self)
        self.input_conf_display.setReadOnly(True)
        val = param_dict['confidence']
        self.input_conf_display.setText(str(val))
        self.input_conf_display.setStyleSheet(
            "background-color:white;color:black")
        self.input_conf_display.setMinimumHeight(int(1.3*ht))

        self.change_conf = QPushButton('CONFIDENCE', self)
        self.change_conf.setStyleSheet("background-color: #2234a8;color:black")
        self.change_conf.setFont(QFont('Arial', self.font_size))
        ht = self.change_conf.frameGeometry().height()
        self.change_conf.setMinimumHeight(int(1.3*ht))
        self.change_conf.clicked.connect(
            lambda: self.button_change_conf_threshold_clicked())

        hl_conf.addWidget(self.input_conf_display, 33)
        hl_conf.addWidget(self.input_conf, 33)
        hl_conf.addWidget(self.change_conf, 33)

        # Setting Yolo IOU value

        hl_iou = QHBoxLayout()
        self.input_iou = QLineEdit(self)
        dv = QDoubleValidator(0.0, 5.0, 2)
        self.input_iou.setValidator(dv)
        self.input_iou.setStyleSheet("background-color:white;color:black")
        self.input_iou.setMinimumHeight(int(1.3*ht))

        self.input_iou_display = QLineEdit(self)
        self.input_iou_display.setReadOnly(True)
        val = param_dict['iou']
        self.input_iou_display.setText(str(val))
        self.input_iou_display.setStyleSheet(
            "background-color:white;color:black")
        self.input_iou_display.setMinimumHeight(int(1.3*ht))

        self.change_iou = QPushButton('    IOU    ', self)
        self.change_iou.setStyleSheet("background-color: #2234a8;color:black")
        self.change_iou.setFont(QFont('Arial', self.font_size))
        ht = self.change_iou.frameGeometry().height()
        self.change_iou.setMinimumHeight(int(1.3*ht))
        self.change_iou.clicked.connect(
            lambda: self.button_change_iou_threshold_clicked())

        hl_iou.addWidget(self.input_iou_display, 33)
        hl_iou.addWidget(self.input_iou, 33)
        hl_iou.addWidget(self.change_iou, 33)

        # Setting Reassignmet threshold for SORT

        hl_reassign = QHBoxLayout()
        self.input_reassign = QLineEdit(self)
        self.onlyInt = QIntValidator()
        self.input_reassign.setValidator(self.onlyInt)
        self.input_reassign.setStyleSheet("background-color:white;color:black")
        self.input_reassign.setMinimumHeight(int(1.3*ht))

        self.input_reassign_display = QLineEdit(self)
        self.input_reassign_display.setReadOnly(True)
        val = param_dict['reassign']
        self.input_reassign_display.setText(str(val))
        self.input_reassign_display.setStyleSheet(
            "background-color:white;color:black")
        self.input_reassign_display.setMinimumHeight(int(1.3*ht))

        self.change_reassign = QPushButton('  REASSIGN  ', self)
        self.change_reassign.setStyleSheet(
            "background-color: #2234a8;color:black")
        self.change_reassign.setFont(QFont('Arial', self.font_size))
        ht = self.change_reassign.frameGeometry().height()
        self.change_reassign.setMinimumHeight(int(1.3*ht))
        self.change_reassign.clicked.connect(
            lambda: self.button_change_reassign_threshold_clicked())

        hl_reassign.addWidget(self.input_reassign_display, 33)
        hl_reassign.addWidget(self.input_reassign, 33)
        hl_reassign.addWidget(self.change_reassign, 33)

        # Setting Motion vector distance threshold for SORT

        hl_mv_distance = QHBoxLayout()
        self.input_mvdistane = QLineEdit(self)
        self.onlyInt = QIntValidator()
        self.input_mvdistane.setValidator(self.onlyInt)
        self.input_mvdistane.setStyleSheet(
            "background-color:white;color:black")
        self.input_mvdistane.setMinimumHeight(int(1.3*ht))

        self.input_mvdistane_display = QLineEdit(self)
        self.input_mvdistane_display.setReadOnly(True)
        val = param_dict['mv_distance']
        self.input_mvdistane_display.setText(str(val))
        self.input_mvdistane_display.setMinimumHeight(int(1.3*ht))
        self.input_mvdistane_display.setStyleSheet(
            "background-color:white;color:black")

        self.change_mv_distane = QPushButton('MV_DISTANCE ', self)
        self.change_mv_distane.setStyleSheet(
            "background-color: #2234a8;color:black")
        self.change_mv_distane.setFont(QFont('Arial', self.font_size))
        ht = self.change_mv_distane.frameGeometry().height()
        self.change_mv_distane.setMinimumHeight(int(1.3*ht))
        self.change_mv_distane.clicked.connect(
            lambda: self.button_change_mv_disatnce_threshold_clicked())

        hl_mv_distance.addWidget(self.input_mvdistane_display, 33)
        hl_mv_distance.addWidget(self.input_mvdistane, 33)
        hl_mv_distance.addWidget(self.change_mv_distane, 33)

        # Setting Motion vector number of frames to use for direction calculation in  SORT

        hl_mv_frameskip = QHBoxLayout()
        self.input_mv_frameskip = QLineEdit(self)
        self.onlyInt = QIntValidator()
        self.input_mv_frameskip.setValidator(self.onlyInt)
        self.input_mv_frameskip.setStyleSheet(
            "background-color:white;color:black")
        self.input_mv_frameskip.setMinimumHeight(int(1.3*ht))

        self.input_mv_frameskip_display = QLineEdit(self)
        self.input_mv_frameskip_display.setReadOnly(True)
        val = param_dict['mv_frameskip']
        self.input_mv_frameskip_display.setText(str(val))
        self.input_mv_frameskip_display.setStyleSheet(
            "background-color:white;color:black")
        self.input_mv_frameskip_display.setMinimumHeight(int(1.3*ht))

        self.change_mv_frameskip = QPushButton('MV_FRAMESKIP', self)
        self.change_mv_frameskip.setStyleSheet(
            "background-color: #2234a8;color:black")
        self.change_mv_frameskip.setFont(QFont('Arial', self.font_size))
        ht = self.change_mv_frameskip.frameGeometry().height()
        self.change_mv_frameskip.setMinimumHeight(int(1.3*ht))
        self.change_mv_frameskip.clicked.connect(
            lambda: self.button_change_mv_frameskip_threshold_clicked())

        hl_mv_frameskip.addWidget(self.input_mv_frameskip_display, 33)
        hl_mv_frameskip.addWidget(self.input_mv_frameskip, 33)
        hl_mv_frameskip.addWidget(self.change_mv_frameskip, 33)

        # Reset Defaults
        self.button_reset_defaults = QPushButton('RESET TO DEFAULTS', self)
        self.button_reset_defaults.setStyleSheet(
            "background-color: #2234a8;color:black")
        self.button_reset_defaults.setFont(QFont('Arial', self.font_size))
        ht = self.button_reset_defaults.frameGeometry().height()
        self.button_reset_defaults.setMinimumHeight(int(1.1*ht))
        self.button_reset_defaults.clicked.connect(
            lambda: self.button_reset_defaults_clicked())

        ###  HEADER  ###

        self.header = QPushButton('CALIBRATION TOOL', self)  # For spacing
        self.header.setStyleSheet("background-color: red;color:black")
        ht = self.header.frameGeometry().height()
        self.header.setMinimumHeight(int(ht))

        hbox = QHBoxLayout()  # add two vertical layouts 1. has stream, 2. panel

        # Stream layout
        vbox1 = QVBoxLayout()
        vbox1.addWidget(self.image_label)

        # PAnel layout
        vbox_panel = QVBoxLayout()
        vbox_panel.addWidget(self.header)

        # vbox_panel.addSpacing(0)
        
        vbox_panel.addWidget(self.button_crop)
        vbox_panel.addWidget(self.button_savecropcoord)

        # vbox_panel.addStretch()
        vbox_panel.addWidget(self.button_segmentation)
        vbox_panel.addLayout(hl_segmentation)

        # vbox_panel.addStretch()
        vbox_panel.addWidget(self.button_gap_points)
        vbox_panel.addWidget(self.button_save_gap_points)
        # vbox_panel.addStretch()

        # vbox_panel.addStretch()
        vbox_panel.addWidget(self.button_pitchcoords)
        vbox_panel.addWidget(self.button_savecropcoord_pitch)
        vbox_panel.addLayout(hl_lens)

        # comboBox = QtGui.QComboBox(self)
        # comboBox.addItem("motif")
        # comboBox.addItem("Windows")
        # comboBox.addItem("cde")
        # comboBox.addItem("Plastique")
        # comboBox.addItem("Cleanlooks")
        # comboBox.addItem("windowsvista")
        # comboBox.move(50, 250)

        vbox_panel.addLayout(hl_conf)
        vbox_panel.addLayout(hl_iou)
        vbox_panel.addLayout(hl_reassign)

        vbox_panel.addLayout(hl_weight)
    

        # vbox_panel.addLayout(hl_mv_distance)
        # vbox_panel.addLayout(hl_mv_frameskip)
        vbox_panel.addWidget(self.button_reset_defaults)
        # vbox_panel.addStretch()

        hbox.addLayout(vbox1)
        hbox.addLayout(vbox_panel)
        self.setLayout(hbox)

        # self.showMaximized()

        # connect its signal to the update_image slot
        self.thread.change_pixmap_signal.connect(self.update_image)

        # start the thread
        self.thread.start()

    def folder_list_with_extension(self,img_folder_path="../Settings/detection_weights/", img_extension =".pt"):
        index_to_remove = []
        file_names = os.listdir(img_folder_path)
        for index, file_name in enumerate(file_names):
            if (not file_name.endswith(img_extension)):
                index_to_remove.append(index)
        count = 0
        for i in index_to_remove:
            file_names.pop(i-count)
            count += 1
        file_names.sort()
        return file_names


    def calculate_homography(self, obj, cam_matrix):
        x_current, y_current = obj
        if lens_distortion_flag == 1:
            test_in = np.array([[[x_current, y_current]]], dtype=np.float32)
            xy_undistorted = cv2.undistortPoints(
                test_in, newcameramtx, lens_dist, None, newcameramtx)
            x_current = int(xy_undistorted[0][0][0])
            y_current = int(xy_undistorted[0][0][1])

        points_map = np.array([[x_current, y_current]], dtype='float32')
        points_map = np.array([points_map])
        return cv2.perspectiveTransform(points_map, cam_matrix)

    def InitialiseScreenSettings(self):
        # global screen_resolution_width, screen_resolution_height, stream_area_width, stream_area_height, panel_area_width, panel_area_height, font_size
        # global img_size

        self.thread.stream_area_width = int(
            (78.6 * self.screen_resolution_width) / 100)
        self.thread.stream_area_height = self.screen_resolution_height

        panel_area_width = self.screen_resolution_width - self.thread.stream_area_width
        panel_area_height = self.screen_resolution_height

        self.thread.img_size = [
            self.thread.stream_area_width, self.thread.stream_area_height]

        self.font_size = int((0.52 * self.screen_resolution_width) / 100)
        print("RES:", self.thread.stream_area_width, self.thread.stream_area_height,
              panel_area_width, panel_area_height)

    ### MOUSE EVENTS ###

    def mousePressEvent(self, event):
        # global drawRect, drawRectCoords
        # global initial_x, initial_y, drawing,  crop_image_module, Segmentation_module, ocr_module, segmentation_points_list, gap_points, gap_module
        self.mutex.acquire()
        pointx = event.pos().x()
        pointy = event.pos().y()
        self.mutex.release()
        if(self.thread.gap_module == True):
            print("Gapmodule")

            x, y = convert_coordinates(
                pointx, pointy, self.thread.img_size, self.thread.frame_size)
            if(len(self.thread.gap_points) >= 2):
                self.thread.gap_points = []
            self.thread.gap_points.append([x, y])
            print("gap points", self.thread.gap_points)
        if(self.thread.Segmentation_module == True):
            print("Entered Segmentation Module")
            if(self.thread.segmentation_done == False):
                pointx, pointy = convert_coordinates(
                    pointx, pointy, self.thread.img_size, self.thread.frame_size)
                self.thread.segmentation_points_list.append([pointx, pointy])
        if(self.thread.crop_image_module == True or self.thread.ocr_module == True):
            if(event.buttons() == QtCore.Qt.LeftButton):
                print("Crop module")
                self.thread.drawRect = False
                self.drawing = True
                self.initial_x, self.initial_y = pointx, pointy

    # Mouse move and release eevnts are mainly used for drawing rectangle for crop module
    # or OCR module such that they rectangle drawn is dynamically displayed on Ui

    def mouseMoveEvent(self, event):
        # global drawRect, drawRectCoords, initial_x, initial_y, drawing
        self.mutex.acquire()
        pointx = event.pos().x()
        pointy = event.pos().y()
        self.thread.drawRect = False
        if self.drawing == True:
            self.thread.drawRect = True
            self.thread.drawRectCoords = [
                self.initial_x, self.initial_y, pointx, pointy]
            a = pointx
            b = pointy
            if a != pointx | b != pointy:
                self.thread.drawRect = True
                self.thread.drawRectCoords = [
                    self.initial_x, self.initial_y, pointx, pointy]
        self.mutex.release()

    def mouseReleaseEvent(self, event):
        # global self.thread.drawRect, drawRectCoords, initial_x, initial_y, self.drawing
        self.mutex.acquire()
        pointx = event.pos().x()
        pointy = event.pos().y()
        self.mutex.release()
        if((self.thread.crop_image_module == True) or (self.thread.ocr_module == True)):
            if(self.drawing == True):
                self.drawing = False
                self.thread.drawRect = True
                self.thread.drawRectCoords = [
                    self.initial_x, self.initial_y, pointx, pointy]

    ### CROP MODULE ###

    # for changing color of buttons , to show the current button clicked in red, and rest all n default blue
    def set_stylesheets_(self, crop_clicked, save_crop_clicked, OCR_clicked, segmentation_clicked, start_select_clicked, display_seg_clicked, umpire_coords_clicked, gap_module_clicked, save_gap_points):
        self.button_crop.setStyleSheet(
            "background-color: red;color:black") if crop_clicked else self.button_crop.setStyleSheet("background-color: #2234a8;color:black")
        self.button_savecropcoord.setStyleSheet(
            "background-color: red;color:black") if save_crop_clicked else self.button_savecropcoord.setStyleSheet("background-color: #2234a8;color:black")
        self.button_segmentation.setStyleSheet(
            "background-color: red;color:black") if segmentation_clicked else self.button_segmentation.setStyleSheet("background-color: #2234a8;color:black")
        self.button_startselect.setStyleSheet(
            "background-color: red;color:black") if start_select_clicked else self.button_startselect.setStyleSheet("background-color: #2234a8;color:black")
        self.button_displaysegimg.setStyleSheet(
            "background-color: red;color:black") if display_seg_clicked else self.button_displaysegimg.setStyleSheet("background-color: #2234a8;color:black")
        self.button_pitchcoords.setStyleSheet(
            "background-color: red;color:black") if umpire_coords_clicked else self.button_pitchcoords.setStyleSheet("background-color: #2234a8;color:black")
        self.button_gap_points.setStyleSheet(
            "background-color: red;color:black") if gap_module_clicked else self.button_gap_points.setStyleSheet("background-color: #2234a8;color:black")
        self.button_save_gap_points.setStyleSheet(
            "background-color: red;color:black") if save_gap_points else self.button_save_gap_points.setStyleSheet("background-color: #2234a8;color:black")

    def button_cropimage_clicked(self):
        """
        1. Deactivates crop in config file(activates only after the coordinates are saved) , acts as a reset button
        2. Deactiavtes all other modules and activates crop module
        3. resets the frame to the frame read from Settings file 

        """
        # global drawRectCoords, drawRectCoords, self.thread.drawRect, field_frame_name, src_image_current, frame_size
        # global ocr_module, crop_image_module, Segmentation_module, gap_module, pitch_region_marking
        self.thread.drawRectCoords = []
        self.thread.crop_image_module = True
        self.thread.ocr_module = False
        self.thread.gap_module = False
        self.thread.Segmentation_module = False
        self.thread.pitch_region_marking = False
        self.thread.drawRect = True
        with open("../Settings/config.json", 'r') as f:
            config_dict = json.load(f)
        config_dict['activate_crop'] = 0
        with open("../Settings/config.json", 'w') as f:
            json.dump(config_dict, f, indent=4)
        self.thread.src_image_current = cv2.imread(
            self.thread.field_frame_name)
        ht, wd, c = self.thread.src_image_current.shape
        self.thread.frame_size = [wd, ht]
        self.set_stylesheets_(True, False, False, False,
                              False, False, False, False, False)

    def button_savecropcoord_clicked(self):
        """
        1. converts the coordinates such that they are with respect to actual image size
        2. Saves them in crop_coordinates.json
        3. Activates crop in config file
        """
        # global save_crop_coordinates, drawRectCoords, self.thread.img_size, frame_size, self.thread.drawRect
        if(self.thread.drawRectCoords != []):
            x1, y1 = convert_coordinates(
                self.thread.drawRectCoords[0], self.thread.drawRectCoords[1], self.thread.img_size, self.thread.frame_size)
            x2, y2 = convert_coordinates(
                self.thread.drawRectCoords[2], self.thread.drawRectCoords[3], self.thread.img_size, self.thread.frame_size)
            print(x1, x2, y1, y2,
                  self.thread.drawRectCoords[0], self.thread.drawRectCoords[1], self.thread.drawRectCoords[2], self.thread.drawRectCoords[3])
            # saving crop coordinates to json file
            data = {}
            data['x1'] = x1
            data['y1'] = y1
            data['x2'] = x2
            data['y2'] = y2
            with open('../Settings/crop_coordinates.json', 'w') as f:
                json.dump(data, f, indent=4)
            self.thread.save_crop_coordinates = True
            with open("../Settings/config.json", 'r') as f:
                config_dict = json.load(f)
            config_dict['activate_crop'] = 1
            with open("../Settings/config.json", 'w') as f:
                json.dump(config_dict, f, indent=4)
            self.set_stylesheets_(False, True, False, False,
                                  False, False, False, False, False)

    ### Umpire region selection ###

    def button_pitchcoords_clicked(self):
        """
        Other Modules r reset and enables drawRect to allow box drawing functionality
        """
        # global pitch_region_marking, drawRectCoords, crop_image_module, ocr_module, Segmentation_module, drawRect, gap_module
        self.thread.drawRectCoords = []
        self.thread.crop_image_module = True
        self.thread.ocr_module = False
        self.thread.gap_module = False
        self.thread.Segmentation_module = False
        self.thread.pitch_region_marking = True
        self.thread.drawRect = True
        self.set_stylesheets_(False, False, False, False,
                              False, False, True, False, False)

    def button_savecropcoord_pitch_clicked(self):
        """
        1. Converts coordinates and saves them in crop_coordinates_ump1.json for umpire 1 's position
        """
        # global drawRectCoords, self.thread.img_size, frame_size, theta
        if(self.thread.drawRectCoords != []):
            x1, y1 = convert_coordinates(
                self.thread.drawRectCoords[0], self.thread.drawRectCoords[1], self.thread.img_size, self.thread.frame_size)
            x2, y2 = convert_coordinates(
                self.thread.drawRectCoords[2], self.thread.drawRectCoords[3], self.thread.img_size, self.thread.frame_size)
            data = {}
            data['x1'] = x1
            data['y1'] = y1
            data['x2'] = x2
            data['y2'] = y2
            with open('../Settings/crop_coordinates_pitch.json', 'w') as f:
                json.dump(data, f, indent=4)
            self.button_pitchcoords_clicked()

    ### GAP module ###

    def button_gap_points_clicked(self):
        # global gap_module, crop_image_module, segmentation_module, ocr_module, self.thread.gap_points
        self.thread.gap_module = True
        self.thread.crop_image_module = False
        self.thread.ocr_module = False
        # segmentation_module = False
        self.thread.gap_points = []
        self.button_gap_points.setStyleSheet(
            "background-color: red;color:black")
        self.button_save_gap_points.setStyleSheet(
            "background-color: #2234a8;color:black")
        self.set_stylesheets_(False, False, False, False,
                              False, False, False, True, False)

    def button_save_gap_points_clicked(self):
        # global gap_points, cam_matrix
        # do homography and save to file
        if len(self.thread.gap_points) == 2:
            stump_data = {}
            if self.thread.gap_points[0] is not None and self.thread.gap_points[0] != []:
                homo_pt = self.calculate_homography(
                    self.thread.gap_points[0], cam_matrix)
                x1, y1 = (int(homo_pt[0][0][0]), int(homo_pt[0][0][1]))
                stump_data["near_end"] = [x1, y1]

            if self.thread.gap_points[1] is not None and self.thread.gap_points[1] != []:
                homo_pt = self.calculate_homography(
                    self.thread.gap_points[1], cam_matrix)
                x2, y2 = (int(homo_pt[0][0][0]), int(homo_pt[0][0][1]))
                stump_data["far_end"] = [x2, y2]
            with open("../Settings/config.json", 'r') as f:
                config_dict = json.load(f)
                config_dict['stump'] = stump_data
            with open("../Settings/config.json", 'w') as f:
                json.dump(config_dict, f, indent=4)
        self.thread.gap_points = []
        self.set_stylesheets_(False, False, False, False,
                              False, False, False, False, True)

    ### SEGMENTATION MODULE ###

    def button_segmentation_clicked(self):
        """
        Resets other modules and resets the points collected for drawing the ellipse
        """
        # global self.thread.drawRect, self.thread.Segmentation_module, save_crop_coordinates, drawRectCoords, crop_image_module, ocr_module
        # global key_ellipse, key_drawEllipse, segmentation_points_list, segmentation_done, gap_module
        self.thread.Segmentation_module = True
        self.thread.segmentation_done = False
        self.thread.save_crop_coordinates = False
        self.thread.crop_image_module = False
        self.thread.gap_module = False
        self.thread.ocr_module = False
        self.thread.drawRectCoords = []
        self.thread.key_ellipse = False
        self.thread.key_drawEllipse = False
        self.thread.drawRect = False
        self.thread.segmentation_points_list = []
        self.set_stylesheets_(False, False, False, True,
                              False, False, False, False, False)

    def button_startselect_clicked(self):
        """
        To start the point selection
        """

        self.thread.segmentation_done = False
        self.thread.key_ellipse = True
        self.set_stylesheets_(False, False, False, False,
                              True, False, False, False, False)

    def button_displaysegimg_clicked(self):
        """
        Draws image, saves the mask and displays the maskedimage
        """
        # global key_drawEllipse, segmentation_done, save_segmentation
        self.thread.key_drawEllipse = True
        self.thread.save_segmentation = True
        self.thread.segmentation_done = True
        self.set_stylesheets_(False, False, False, False,
                              False, True, False, False, False)

    ### HYPERPARAMS TUNNINg METHODS ###

    def set_stylesheets_hyperparams(self, conf_clicked, iou_clicked, reassign_clicked, mv_dist_clicked, mv_frameskip_clicked):
        self.change_conf.setStyleSheet(
            "background-color: red;color:black") if conf_clicked else self.change_conf.setStyleSheet("background-color: #2234a8;color:black")
        self.change_iou.setStyleSheet(
            "background-color: red;color:black") if iou_clicked else self.change_iou.setStyleSheet("background-color: #2234a8;color:black")
        self.change_reassign.setStyleSheet(
            "background-color: red;color:black") if reassign_clicked else self.change_reassign.setStyleSheet("background-color: #2234a8;color:black")
        self.change_mv_distane.setStyleSheet(
            "background-color: red;color:black") if mv_dist_clicked else self.change_mv_distane.setStyleSheet("background-color: #2234a8;color:black")
        self.change_mv_frameskip.setStyleSheet(
            "background-color: red;color:black") if mv_frameskip_clicked else self.change_mv_frameskip.setStyleSheet("background-color: #2234a8;color:black")

    def button_change_conf_threshold_clicked(self):
        """
        Replaces current value of hyperparameter in json file and updates display
        """

        text = self.input_conf.text()
        if text != "":
            val = float(text)
            self.input_conf.clear()
            with open("../Settings/hyperparms.json", "r") as f:
                param_dict = json.load(f)
            param_dict['confidence'] = val
            with open("../Settings/hyperparms.json", "w") as f:
                json.dump(param_dict, f, indent=4)
            self.input_conf_display.setText(str(val))
            self.set_stylesheets_hyperparams(True, False, False, False, False)

    def button_change_iou_threshold_clicked(self):
        text = self.input_iou.text()
        if text != "":
            val = float(text)
            self.input_iou.clear()
            with open("../Settings/hyperparms.json", "r") as f:
                param_dict = json.load(f)
            param_dict['iou'] = val
            with open("../Settings/hyperparms.json", "w") as f:
                json.dump(param_dict, f, indent=4)
            self.input_iou_display.setText(str(val))
            self.set_stylesheets_hyperparams(False, True, False, False, False)

    def button_change_reassign_threshold_clicked(self):
        text = self.input_reassign.text()
        if text != "":
            val = int(text)
            self.input_reassign.clear()
            with open("../Settings/hyperparms.json", "r") as f:
                param_dict = json.load(f)
            param_dict['reassign'] = val
            with open("../Settings/hyperparms.json", "w") as f:
                json.dump(param_dict, f, indent=4)
            self.input_reassign_display.setText(str(val))
            self.set_stylesheets_hyperparams(False, False, True, False, False)

    def button_change_mv_disatnce_threshold_clicked(self):
        text = self.input_mvdistane.text()
        val = int(text)
        self.input_mvdistane.clear()
        with open("../Settings/hyperparms.json", "r") as f:
            param_dict = json.load(f)
        param_dict['mv_distance'] = val
        with open("../Settings/hyperparms.json", "w") as f:
            json.dump(param_dict, f, indent=4)
        self.input_mvdistane_display.setText(str(val))
        self.set_stylesheets_hyperparams(False, False, False, True, False)

    def button_change_mv_frameskip_threshold_clicked(self):
        text = self.input_mv_frameskip.text()
        val = int(text)
        self.input_mv_frameskip.clear()
        with open("../Settings/hyperparms.json", "r") as f:
            param_dict = json.load(f)
        param_dict['mv_frameskip'] = val
        with open("../Settings/hyperparms.json", "w") as f:
            json.dump(param_dict, f, indent=4)
        self.input_mv_frameskip_display.setText(str(val))
        self.set_stylesheets_hyperparams(False, False, False, False, True)

    def button_reset_defaults_clicked(self):
        with open("../Settings/hyperparms.json", "r") as f:
            param_dict = json.load(f)

        param_dict['confidence'] = 0.55
        self.input_conf_display.setText(str(0.55))
        param_dict['iou'] = 0.33
        self.input_iou_display.setText(str(0.33))
        param_dict['reassign'] = 250
        self.input_reassign_display.setText(str(250))
        param_dict['mv_distance'] = 15
        self.input_mvdistane_display.setText(str(15))
        param_dict['mv_frameskip'] = 20
        self.input_mv_frameskip_display.setText(str(20))

        with open("../Settings/hyperparms.json", "w") as f:
            json.dump(param_dict, f, indent=4)


    def weights_change(self):
        i = self.weights.currentIndex()
        weight_name = self.weight_list[i-1]
        with open("../Settings/config.json", 'r') as f:
            config_dict = json.load(f)
        config_dict['detection_weight'] = "Settings/detection_weights/"+weight_name
        with open("../Settings/config.json", 'w') as f:
            json.dump(config_dict, f, indent=4)
    
    def lensButtonClicked(self):
        with open("../Settings/config.json", 'r') as f:
            config_dict = json.load(f)
        lens_distortion = config_dict['lens_distortion'] 
        if lens_distortion ==0:
            config_dict['lens_distortion'] =1
            self.lens.setChecked(True)
        else:
            config_dict['lens_distortion'] =0
            self.lens.setChecked(False)
        with open("../Settings/config.json", 'w') as f:
            json.dump(config_dict, f, indent=4)


    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        """Updates the image_label with a new opencv image"""
        # global batsmen_name_list, overs, ratio_list, max_val_ind

        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)



    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(
            rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(
            self.disply_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    a = App()
    a.show()
    sys.exit(app.exec_())