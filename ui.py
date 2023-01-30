import sys

import cv2
import numpy as np
from PyQt5 import QtCore, QtGui
from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtCore import Qt, pyqtSlot, QSortFilterProxyModel
from PyQt5.QtGui import QFont, QIntValidator, QKeySequence, QPixmap, QStandardItemModel, QStandardItem
from PyQt5.QtWidgets import (QComboBox, QApplication, QDesktopWidget, QHBoxLayout, QLabel,
                             QLineEdit, QPushButton, QShortcut, QVBoxLayout, QTableView,
                             QHeaderView, QAbstractItemView,
                             QWidget, QCheckBox)

from process import VideoThread
from Utils.ui_utils import Config
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class App(QWidget):

    def __init__(self):
        super().__init__()
        self.thread = VideoThread()
        self.capture_frame = False
        self.outside_player_active = False
        self.lr_automated = False
        self.player_type_change_flag = False
        self.current_mouse_pos = ()
        self.img_size = []
        self.frame_size = []
        self.screen_resolution_width = 0
        self.screen_resolution_height = 0
        self.font_size = 10
        self.dist_ids = []
        self.player_connect_count = 1
        self.player_connect_dict = {}
        self.dist_listt = []
        self.stream_area_width = 0
        self.stream_area_height = 0
        self.player_type_change_id = -1
        self.b_clicked = -1
        self.center()
        self.ui_config = Config()
        self.ui_config.ui_config()
        self.marked_id_TA = -1
        self.marked_id_TB = -1
        self.marked_id_PO = -1
        self.panel_val = 0

        self.InitialiseScreenSettings()

        QShortcut(QKeySequence(Qt.Key_Q), self,
                  activated=self.thread.livelock_clicked)
        QShortcut(QKeySequence(Qt.Key_1), self,
                  activated=self.connect_players_line1)
        # QShortcut(QKeySequence(Qt.Key_B), self, activated=self.set_batsmen_id)
        # QShortcut(QKeySequence(Qt.Key_V), self, activated=self.set_bowler_id)

        QShortcut(QKeySequence(Qt.Key_R), self, activated=self.make_all_red)
        QShortcut(QKeySequence(Qt.Key_Space), self, activated=self.assign)
        QShortcut(QKeySequence(Qt.Key_S), self, activated=self.reswap)
        QShortcut(QKeySequence(Qt.Key_V), self, activated=self.record)
        QShortcut(QKeySequence(Qt.Key_T), self, activated=self.set_batsmen_id)
        QShortcut(QKeySequence(Qt.Key_Y), self, activated=self.set_wk_id)
        
        
        QShortcut(QKeySequence(Qt.Key_0), self,
                  activated=self.set_dummy_connect)
        QShortcut(QKeySequence(Qt.Key_9), self,
                  activated=self.set_dummy_player)
        
        QShortcut(QKeySequence(Qt.Key_K), self,
                  activated=self.add_multi_gap_ids)
        QShortcut(QKeySequence(Qt.Key_G), self, activated=self.add_gap_ids)
        QShortcut(QKeySequence(Qt.Key_H), self, activated=self.add_ingap_ids)
        QShortcut(QKeySequence(Qt.Key_U), self, activated=self.set_umpire_id)
        QShortcut(QKeySequence(Qt.Key_N), self,
                  activated=self.false_negative_clicked)
        QShortcut(QKeySequence(Qt.Key_P), self,
                  activated=self.mark_point_clicked)
        QShortcut(QKeySequence(Qt.Key_D), self,
                  activated=self.calculate_distance_clicked)
        QShortcut(QKeySequence(Qt.Key_L), self,
                  activated=self.false_negative_slip_fielders_clicked)
        # QShortcut(QKeySequence(Qt.Key_S), self,
        #           activated=self.thread.switch_camera)
        QShortcut(QKeySequence(Qt.Key_O), self,
                  activated=self.set_dist_pt_to_wkt)
        QShortcut(QKeySequence(Qt.Key_B), self,
                  activated=self.set_dist_fielder_to_wkt)
        QShortcut(QKeySequence(Qt.Key_E), self,
                  activated=self.copy_scorefile_data)
        QShortcut(QKeySequence(Qt.Key_M), self,
                  activated=self.send_config_data)
        QShortcut(QKeySequence(Qt.Key_J), self,
                  activated=self.send_player_speed)
        QShortcut(QKeySequence(Qt.Key_Z), self,
                  activated=self.false_negative_outside_frame_z)
        QShortcut(QKeySequence(Qt.Key_A), self,
                  activated=self.false_negative_outside_frame_a)
        QShortcut(QKeySequence(Qt.Key_C), self,
                  activated=self.false_negative_outside_frame_u)
        # Team A
        QShortcut(QKeySequence(Qt.Key_2), self,
                  activated=lambda: self.set_id_name(1))
        # Team B
        QShortcut(QKeySequence(Qt.Key_3), self,
                  activated=lambda: self.set_id_name(2))
        # Player Postions
        QShortcut(QKeySequence(Qt.Key_4), self,
                  activated=lambda: self.set_id_name(3))
        # Back key
        QShortcut(QKeySequence(Qt.Key_5), self,
                  activated=lambda: self.clear_vbox(0))
        # adding name and position
        # QShortcut(QKeySequence(Qt.Key_H), self, activated=self.set_id_name)
        QShortcut(QKeySequence(Qt.Key_F4), self,
                  activated=self.thread.command_f4)
        QShortcut(QKeySequence(Qt.Key_F6), self,
                  activated=self.thread.command_f6)
        QShortcut(QKeySequence(Qt.Key_F7), self,
                  activated=self.thread.command_f7)
        QShortcut(QKeySequence(Qt.Key_F8), self,
                  activated=self.thread.command_f8)
        QShortcut(QKeySequence(Qt.Key_F9), self,
                  activated=self.thread.command_f9)
        # QShortcut(QKeySequence(Qt.Key_F), self,
        #           activated=self.thread.command_f)
        QShortcut(QKeySequence(Qt.Key_W), self,
                  activated=self.thread.command_w)

        ### Window Settings ###
        self.setWindowFlag(Qt.FramelessWindowHint)
        self.setWindowTitle("Quidich Tracker")
        self.setWindowIcon(QtGui.QIcon("Settings/Logos/qt_logo.png"))
        self.disply_width = self.screen_resolution_width
        self.display_height = self.screen_resolution_height
        self.setStyleSheet("QWidget {background-color: #151515;}")

        ### create the label that holds the image ###
        self.image_label = QLabel(self)
        self.image_label.resize(self.disply_width, self.display_height)
        self.image_label.mousePressEvent = self.get_pos
        self.image_label.setMouseTracking(True)
        self.image_label.mouseMoveEvent = self.handleMouseHover

        ####################
        ### PANEL LAYOUT ###

        # 1. Player IDS of Umpires and batsmen

        self.tl_fielder = QLabel('FIELDERS : ')
        self.tl_fielder.setFont(QFont('Arial', 1.4*self.font_size))
        self.tl_fielder.setAlignment(Qt.AlignCenter)
        self.tl_fielder.setStyleSheet(
            "QLabel {background-color: white; color : green;}")

        self.tl_fn = QLabel('FALSE NEG : ')
        self.tl_fn.setFont(QFont('Arial', 1.4*self.font_size))
        self.tl_fn.setAlignment(Qt.AlignCenter)
        self.tl_fn.setStyleSheet(
            "QLabel {background-color: white; color : black;}")

        self.tl_others = QLabel('OTHERS : ')
        self.tl_others.setFont(QFont('Arial', 1.4*self.font_size))
        self.tl_others.setAlignment(Qt.AlignCenter)
        self.tl_others.setStyleSheet(
            "QLabel {background-color: white; color : red;}")

        self.tl_total = QLabel('')
        self.tl_total.setFont(QFont('Arial', 2.5 * self.font_size))
        self.tl_total.setAlignment(Qt.AlignCenter)
        self.tl_total.setStyleSheet(
            "QLabel {background-color: green; color : white;}")

        self.tl_maxid = QLabel('')
        self.tl_maxid.setFont(QFont('Arial', 2.5 * self.font_size))
        self.tl_maxid.setAlignment(Qt.AlignCenter)
        self.tl_maxid.setStyleSheet(
            "QLabel {background-color: green; color : white;}")

        self.tl_drop = QLabel('Drop')
        self.tl_drop.setFont(QFont('Arial', self.font_size))
        self.tl_drop.setAlignment(Qt.AlignCenter)
        self.tl_drop.setStyleSheet("QLabel {background-color: white;}")

        self.le_drop = QLineEdit()
        self.only_int = QIntValidator()
        self.le_drop.setValidator(self.only_int)
        self.le_drop.setReadOnly(True)
        self.le_drop.setStyleSheet(
            "QLineEdit {background-color: white; color : black;}")
        self.le_drop.setAlignment(Qt.AlignCenter)

        self.tl_crop = QLabel(' CROP  :  ')
        self.tl_crop.setFont(QFont('Arial', 1.5 * self.font_size))
        self.tl_crop.setAlignment(Qt.AlignCenter)

        if self.ui_config.activate_crop == 1:
            self.tl_crop.setText(" CROP  :  ACTIVATED")
            self.tl_crop.setStyleSheet(
                "QLabel {background-color: green; color : white;}")
        elif self.ui_config.activate_crop == 0:
            self.tl_crop.setText(" CROP  :  DEACTIVATED")
            self.tl_crop.setStyleSheet(
                "QLabel {background-color: red; color : white;}")

        self.button_livelock = QPushButton('LIVELOCK', self)
        self.button_livelock.setStyleSheet(
            "background-color: #2234a8;color: white;font-weight: bold;")
        self.button_livelock.setFont(QFont('Arial', 2*self.font_size))
        ht = self.button_livelock.frameGeometry().height()
        self.button_livelock.setMinimumHeight(int(1.4*ht))
        self.button_livelock.clicked.connect(
            lambda: self.thread.livelock_clicked())

        self.button_insert_db = QPushButton('RECORD DATA', self)
        self.button_insert_db.setStyleSheet(
            "background-color: #2234a8;color: white;font-weight: bold;")
        self.button_insert_db.setFont(QFont('Arial', 2*self.font_size))
        ht = self.button_insert_db.frameGeometry().height()
        self.button_insert_db.setMinimumHeight(int(1.4*ht))
        self.button_insert_db.clicked.connect(
            lambda: self.thread.insertDb_clicked())

        self.button_air = QPushButton('DOWNSTREAM', self)
        self.button_air.setStyleSheet(
            "background-color: #2234a8;color: white;font-weight: bold;")
        self.button_air.setFont(QFont('Arial', 2*self.font_size))
        ht = self.button_air.frameGeometry().height()
        self.button_air.setMinimumHeight(int(1.4*ht))
        self.button_air.clicked.connect(
            lambda: self.thread.air_clicked())

        #########################
        # For Sapcing
        self.spacing1 = QLabel('')

        #########################

        self.button_reset_calib = QPushButton('Reset Calibration', self)
        self.button_reset_calib.setStyleSheet(
            "background-color: #2234a8;color: white;font-weight: bold;")
        self.button_reset_calib.setFont(QFont('Arial', 1.8*self.font_size))
        ht = self.button_reset_calib.frameGeometry().height()
        self.button_reset_calib.setMinimumHeight(int(ht))
        self.button_reset_calib.clicked.connect(
            lambda: self.thread.reset_calib_handler())

        self.button_reset = QPushButton('RESET', self)
        self.button_reset.setStyleSheet(
            "background-color: #2234a8;color: white;font-weight: bold;")
        self.button_reset.setFont(QFont('Arial', 1.8*self.font_size))
        ht = self.button_reset.frameGeometry().height()
        self.button_reset.setMinimumHeight(int(ht))
        self.button_reset.clicked.connect(lambda: self.button_reset_clicked())

        self.button_near_end = QPushButton("Near end", self)
        self.button_near_end.setStyleSheet(
            "background-color: #2234a8;color: white;font-weight: bold;")
        self.button_near_end.setFont(QFont('Arial', 1.8*self.font_size))
        ht = self.button_near_end.frameGeometry().height()
        self.button_near_end.setMinimumHeight(int(1.4*ht))
        self.button_near_end.clicked.connect(
            lambda: self.button_clicked_near())

        self.button_far_end = QPushButton('Far end', self)
        self.button_far_end.setStyleSheet(
            "background-color: #2234a8;color: white;font-weight: bold;")
        self.button_far_end.setFont(QFont('Arial', 1.8*self.font_size))
        ht = self.button_far_end.frameGeometry().height()
        self.button_far_end.setMinimumHeight(int(1.4*ht))
        self.button_far_end.clicked.connect(lambda: self.button_clicked_far())

        # self.hl_fieldplot = QHBoxLayout()
        # self.hl_fieldplot.addWidget(self.button_near_end)
        # self.hl_fieldplot.addWidget(self.button_far_end)

        self.button_left = QPushButton("Left Handed", self)
        self.button_left.setStyleSheet(
            "background-color: #2234a8;color: white;font-weight: bold;")
        self.button_left.setFont(QFont('Arial', 1.8*self.font_size))
        ht = self.button_left.frameGeometry().height()
        self.button_left.setMinimumHeight(int(1.4*ht))
        self.button_left.clicked.connect(lambda: self.button_clicked_left())

        self.button_right = QPushButton("Right handed", self)
        self.button_right.setStyleSheet(
            "background-color: #2234a8;color: white;font-weight: bold;")
        self.button_right.setFont(QFont('Arial', 1.8*self.font_size))
        ht = self.button_right.frameGeometry().height()
        self.button_right.setMinimumHeight(int(1.4*ht))
        self.button_right.clicked.connect(lambda: self.button_clicked_right())

        self.button_lr_automated = QPushButton("Automated LH/RH", self)
        self.button_lr_automated.setStyleSheet(
            "background-color: #2234a8;color: white;font-weight: bold;")
        self.button_lr_automated.setFont(QFont('Arial', 1.8*self.font_size))
        ht = self.button_lr_automated.frameGeometry().height()
        self.button_lr_automated.setMinimumHeight(int(1.4*ht))
        self.button_lr_automated.clicked.connect(
            lambda: self.button_clicked_lr_automated())

        # self.hl_batsmenpos = QHBoxLayout()
        # self.hl_batsmenpos.addWidget(self.button_left, 50)
        # self.hl_batsmenpos.addWidget(self.button_right, 50)

        # self.le_overs = QLineEdit()
        # self.le_overs.setFont(QFont('Arial', 1.4*self.font_size))
        # self.le_overs.setAlignment(Qt.AlignCenter)
        # self.le_overs.setReadOnly(True)
        # self.le_overs.setStyleSheet(
        #     "QLineEdit {background-color: white; color : black;font-weight: bold;}")

        # self.le_onstrike_batsman = QLineEdit()
        # self.le_onstrike_batsman.setFont(QFont('Arial', 1.4 * self.font_size))
        # self.le_onstrike_batsman.setAlignment(Qt.AlignCenter)
        # self.le_onstrike_batsman.setReadOnly(True)
        # self.le_onstrike_batsman.setStyleSheet(
        #     "QLineEdit {background-color: white; color : black;font-weight: bold;}")

        # self.le_offstrike_batsman = QLineEdit()
        # self.le_offstrike_batsman.setFont(QFont('Arial', 1.4*self.font_size))
        # self.le_offstrike_batsman.setAlignment(Qt.AlignCenter)
        # self.le_offstrike_batsman.setReadOnly(True)
        # self.le_offstrike_batsman.setStyleSheet(
        #     "QLineEdit {background-color: white; color : black;font-weight: bold;}")

        self.button_captureframes = QPushButton("Capture Frame", self)
        self.button_captureframes.setStyleSheet(
            "background-color: #2234a8;color: white;font-weight: bold;")
        self.button_captureframes.setFont(QFont('Arial', 1.8*self.font_size))
        ht = self.button_captureframes.frameGeometry().height()
        self.button_captureframes.setMinimumHeight(int(ht))
        self.button_captureframes.clicked.connect(
            lambda: self.button_captureframes_clicked())

        self.button_saveframe = QPushButton("Save Frame", self)
        self.button_saveframe.setStyleSheet(
            "background-color: #2234a8;color: white;font-weight: bold;")
        self.button_saveframe.setFont(QFont('Arial', 1.8*self.font_size))
        ht = self.button_saveframe.frameGeometry().height()
        self.button_saveframe.setMinimumHeight(int(ht))
        self.button_saveframe.clicked.connect(
            lambda: self.thread.save_frame())

        # self.button_showOCRfeed = QPushButton("OCR Feed", self)
        # self.button_showOCRfeed.setStyleSheet(
        #     "background-color: #2234a8;color: white;font-weight: bold;")
        # self.button_showOCRfeed.setFont(QFont('Arial', 2*self.font_size))
        # ht = self.button_showOCRfeed.frameGeometry().height()
        # self.button_showOCRfeed.setMinimumHeight(int(ht))
        # self.button_showOCRfeed.clicked.connect(
        #     lambda: self.button_showOCRfeed_clicked())

        # self.button_resetbatsmen = QPushButton("Reset Batsmen", self)
        # self.button_resetbatsmen.setStyleSheet(
        #     "background-color: #2234a8;color: white;font-weight: bold;")
        # self.button_resetbatsmen.setFont(QFont('Arial', 1.8*self.font_size))
        # ht = self.button_resetbatsmen.frameGeometry().height()
        # self.button_resetbatsmen.setMinimumHeight(int(ht))
        # self.button_resetbatsmen.clicked.connect(
        #     lambda: self.thread.reset_batsmen())

        self.button_resetbowler = QPushButton("Reset Bowler", self)
        self.button_resetbowler.setStyleSheet(
            "background-color: #2234a8;color: white;font-weight: bold;")
        self.button_resetbowler.setFont(QFont('Arial', 1.8*self.font_size))
        ht = self.button_resetbowler.frameGeometry().height()
        self.button_resetbowler.setMinimumHeight(int(ht))
        self.button_resetbowler.clicked.connect(
            lambda: self.thread.reset_bowler_flager())

        self.button_resetumpire = QPushButton("Reset Umpire", self)
        self.button_resetumpire.setStyleSheet(
            "background-color: #2234a8;color: white;font-weight: bold;")
        self.button_resetumpire.setFont(QFont('Arial', 1.8*self.font_size))
        ht = self.button_resetumpire.frameGeometry().height()
        self.button_resetumpire.setMinimumHeight(int(ht))
        self.button_resetumpire.clicked.connect(
            lambda: self.thread.reset_umpire_flager())

        self.button_resethighlight = QPushButton("Reset Highlights", self)
        self.button_resethighlight.setStyleSheet(
            "background-color: #2234a8;color: white;font-weight: bold;")
        self.button_resethighlight.setFont(QFont('Arial', 1.8*self.font_size))
        ht = self.button_resethighlight.frameGeometry().height()
        self.button_resethighlight.setMinimumHeight(int(ht))
        self.button_resethighlight.clicked.connect(
            lambda: self.button_resetHighlight_clicked())

        self.button_reset_fn = QPushButton("Reset Dummies", self)
        self.button_reset_fn.setStyleSheet(
            "background-color: #2234a8;color: white;font-weight: bold;")
        self.button_reset_fn.setFont(QFont('Arial', 1.8*self.font_size))
        ht = self.button_reset_fn.frameGeometry().height()
        self.button_reset_fn.setMinimumHeight(int(ht))
        self.button_reset_fn.clicked.connect(
            lambda: self.thread.reset_FN_flager())

        self.button_naming = QPushButton("Reset Naming", self)
        self.button_naming.setStyleSheet(
            "background-color: #2234a8;color: white;font-weight: bold;")
        self.button_naming.setFont(QFont('Arial', 1.8*self.font_size))
        ht = self.button_naming.frameGeometry().height()
        self.button_naming.setMinimumHeight(int(ht))
        self.button_naming.clicked.connect(
            lambda: self.thread.reset_naming_flager())

        # self.button_outside = QPushButton("Inner-Circle ", self)
        # self.button_outside.setStyleSheet(
        #     "background-color: #2234a8;color: white;font-weight: bold;")
        # self.button_outside.setFont(QFont('Arial', 1.8*self.font_size))
        # ht = self.button_outside.frameGeometry().height()
        # self.button_outside.setMinimumHeight(int(1*ht))
        # self.button_outside.clicked.connect(
        #     lambda: self.button_outside_clicked())

        # self.outside_circle_no = QLineEdit()
        # self.outside_circle_no.setFont(QFont('Arial', 1.4*self.font_size))
        # self.outside_circle_no.setAlignment(Qt.AlignCenter)
        # self.outside_circle_no.setReadOnly(True)
        # self.outside_circle_no.setStyleSheet(
        #     "QLineEdit {background-color: white; color : black;font-weight: bold;}")

        # db fields
        self.le_db_o = QLineEdit()
        self.only_int = QIntValidator()
        self.le_db_o.setValidator(self.only_int)
        self.le_db_o.setStyleSheet(
            "QLineEdit {background-color: white; color : black;}")
        self.le_db_o.setAlignment(Qt.AlignCenter)
        self.le_db_o.setMinimumHeight(int(1.2*ht))

        self.le_db_b = QLineEdit()
        self.only_int = QIntValidator()
        self.le_db_b.setValidator(self.only_int)
        self.le_db_b.setStyleSheet(
            "QLineEdit {background-color: white; color : black;}")
        self.le_db_b.setAlignment(Qt.AlignCenter)
        self.le_db_b.setMinimumHeight(int(1.2*ht))

        self.spacing5 = QLabel('')  # For Spacing
        self.spacing6 = QLabel('')  # For Spacing
        self.spacing7 = QLabel('')  # For Spacing
        self.spacing8 = QLabel('')  # For Spacing
        self.spacing9 = QLabel('')  # For Spacing
        self.spacing10 = QLabel('')  # For Spacing
        self.spacing11 = QLabel('')  # For Spacing
        self.spacing1 = QLabel('')  # For Spacing

        self.position_list = list(self.ui_config.fielder_position)
        self.search_model = QStandardItemModel(len(self.position_list), 1)
        self.search_model.setHorizontalHeaderLabels(['Company'])

        self.filter_proxy_model = QSortFilterProxyModel()
        self.filter_proxy_model.setSourceModel(self.search_model)
        self.filter_proxy_model.setFilterCaseSensitivity(Qt.CaseInsensitive)
        self.filter_proxy_model.setFilterKeyColumn(0)
        # print(filter_proxy_model)

        self.search_field = QLineEdit()
        self.search_field.setStyleSheet(
            'font-size:25px; height: 40px;color: white;')
        self.search_field.textChanged.connect(
            self.filter_proxy_model.setFilterRegExp)
        for row, position in enumerate(self.position_list):
            self.search_model.setItem(row, 0, QStandardItem(position))

        self.search_field = QLineEdit()
        self.search_field.setStyleSheet(
            'font-size:25px; height: 40px;color: white;')
        self.search_field.textChanged.connect(
            self.filter_proxy_model.setFilterRegExp)
        for row, position in enumerate(self.position_list):
            self.search_model.setItem(row, 0, QStandardItem(position))

        self.table = QTableView()
        self.table.setStyleSheet(
            "background-color: #2234a8;color: white;font-weight: bold;")
        self.table.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setVisible(False)
        self.table.setModel(self.filter_proxy_model)
        self.table.clicked.connect(self.position_clicked)
        self.table.setSelectionMode(QAbstractItemView.NoSelection)

        self.front_end_mode = QCheckBox('FRONT-END MODE')
        # self.front_end_mode.setStyleSheet("background-color: #2234a8;color:black;selection-color: black;selection-background-color: white;")
        self.front_end_mode.setStyleSheet(
            "background-color: #2234a8;color: white;font-weight: bold;")
        self.front_end_mode.setObjectName("front_end_mode")
        self.front_end_mode.setFont(QFont('Arial', self.font_size*1.2))
        self.front_end_mode.setMinimumHeight(int(1*ht))
        self.front_end_mode.setChecked(False)
        self.front_end_mode.clicked.connect(
            lambda: self.thread.command_front_end_mode(self.front_end_mode.isChecked()))

        # 3
        hbox = QHBoxLayout()  # add two vertical layouts 1. has stream, 2. panel

        # Stream layout
        vbox_image = QVBoxLayout()
        vbox_image.addWidget(self.image_label)
        hbox.addLayout(vbox_image)

        # PAnel layout

        self.vbox_panel = QVBoxLayout()
        self.panel_0()
        hbox.addLayout(self.vbox_panel)

        self.setLayout(hbox)
        # print("Total panel elements count",self.vbox_panel.count())

        # 333333
        self.showMaximized()

        # connect its signal to the update_image slot
        self.thread.change_pixmap_signal.connect(self.update_image)

        # start the thread
        self.thread.start()

    def panel_new(self, panel_val):

        back_to_panel_0 = self.back_button()
        self.vbox_panel.addWidget(back_to_panel_0)

        if panel_val == 1:
            f_names_lis = self.ui_config.fielder_list
        elif panel_val == 2:
            f_names_lis = self.ui_config.fielder_list_1
        elif panel_val == 3:
            f_names_lis = self.ui_config.fielder_position

        selected_f_name_lis = list(self.thread.fielder_dict.values(
        )) + list(self.thread.fielder_dict_PO.values())

        temp_vbox = QVBoxLayout()

        for f_names in f_names_lis:
            temp_field = self.create_playerButtons(
                f_names, selected_f_name_lis)
            temp_vbox.addWidget(temp_field)

        self.vbox_panel.addLayout(temp_vbox)

    def panel_search(self):
        self.vbox_panel.addWidget(self.back_button())
        # self.clearLayout(self.panel_search_vbox)
        self.vbox_panel.addWidget(self.search_field)
        self.vbox_panel.addWidget(self.table)
        # self.vbox_panel.addLayout(self.panel_search_vbox)
        # print("done with panel_search")

    def panel_0(self):
        self.hl_fieldplot = QHBoxLayout()
        self.hl_fieldplot.addWidget(self.button_near_end)
        self.hl_fieldplot.addWidget(self.button_far_end)

        self.hl_batsmenpos = QHBoxLayout()
        self.hl_batsmenpos.addWidget(self.button_left, 50)
        self.hl_batsmenpos.addWidget(self.button_right, 50)

        # self.hl_batsmen_name = QHBoxLayout()
        # self.hl_batsmen_name.addWidget(self.le_onstrike_batsman, 50)
        # self.hl_batsmen_name.addWidget(self.le_offstrike_batsman, 50)

        # self.outside_circle_layout = QHBoxLayout()
        # self.outside_circle_layout.addWidget(self.button_outside, 50)
        # self.outside_circle_layout.addWidget(self.outside_circle_no, 50)

        self.tick_box_layout = QHBoxLayout()
        self.tick_box_layout.addWidget(self.front_end_mode)

        self.vbox_panel.addWidget(self.button_livelock)
        self.vbox_panel.addLayout(self.tick_box_layout)
        self.vbox_panel.addWidget(self.tl_total)
        self.vbox_panel.addWidget(self.tl_fielder)
        self.vbox_panel.addWidget(self.tl_fn)
        self.vbox_panel.addWidget(self.tl_others)
        self.vbox_panel.addSpacing(2)

        # self.vbox_panel.addWidget(self.spacing7)
        self.vbox_panel.addWidget(self.button_insert_db)
        self.vbox_panel.addWidget(self.button_air)
        self.vbox_panel.addLayout(self.hl_fieldplot)
        self.vbox_panel.addSpacing(2)

        # self.vbox_panel.addWidget(self.spacing8)
        self.vbox_panel.addWidget(self.button_lr_automated)
        self.vbox_panel.addLayout(self.hl_batsmenpos)
        # self.vbox_panel.addWidget(self.le_overs)
        # self.vbox_panel.addLayout(self.hl_batsmen_name)
        self.vbox_panel.addSpacing(1)
        # self.vbox_panel.addLayout(self.outside_circle_layout)
        # self.vbox_panel.addSpacing(2)
        # db fields
        # self.vbox_panel.addLayout(self.gl_db_val)
        # self.vbox_panel.addWidget(self.button_fetchDb)
        self.vbox_panel.addSpacing(1)
        # db fields
        self.vbox_panel.addWidget(self.button_reset)
        # self.vbox_panel.addWidget(self.button_resetbatsmen)
        self.vbox_panel.addWidget(self.button_resetbowler)

        self.vbox_panel.addWidget(self.button_resetumpire)
        self.vbox_panel.addWidget(self.button_resethighlight)
        self.vbox_panel.addWidget(self.button_reset_fn)
        self.vbox_panel.addWidget(self.button_naming)
        self.vbox_panel.addWidget(self.tl_maxid)

        # self.vbox_panel.addWidget(self.spacing11)
        # self.vbox_panel.addWidget(self.spacing5)
        self.vbox_panel.addWidget(self.button_reset_calib)
        self.vbox_panel.addWidget(self.button_captureframes)
        self.vbox_panel.addWidget(self.button_saveframe)
        # self.vbox_panel.addWidget(self.button_showOCRfeed)
        # self.vbox_panel.addWidget(self.tl_drop)
        self.vbox_panel.addWidget(self.le_drop)
        self.vbox_panel.addWidget(self.tl_crop)
        # self.vbox_panel.addWidget(self.spacing1)
        self.vbox_panel.addWidget(self.spacing6)

    def clearLayout(self, layout):
        if layout is not None:
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.setParent(None)
                else:
                    self.clearLayout(item.layout())

    def back_button(self):
        back_to_panel_0 = QPushButton('Back', self)
        back_to_panel_0.setStyleSheet(
            "background-color: green; color : white;font-weight: bold;")
        back_to_panel_0.setFont(QFont('Arial', 2*self.font_size))
        ht = back_to_panel_0.frameGeometry().height()
        back_to_panel_0.setMinimumHeight(int(1.4*ht))
        back_to_panel_0.clicked.connect(
            lambda: self.clear_vbox(0))
        return back_to_panel_0

    def create_playerButtons(self, name, selected_f_name_lis):
        temp_field = QPushButton(name, self)
        if name in selected_f_name_lis:
            temp_field.setStyleSheet(
                "background-color: green;color: white;font-weight: bold;")
            temp_field.setEnabled(False)
        else:
            temp_field.setStyleSheet(
                "background-color: #2234a8;color: white;font-weight: bold;")

        temp_field.setFont(QFont('Arial', 1.5*self.font_size))
        ht = temp_field.frameGeometry().height()
        temp_field.setMinimumHeight(int(1.1*ht))
        temp_field.clicked.connect(
            lambda: self.player_clicked(self.sender().text()))
        # self.sender().text()
        return temp_field

    def position_clicked(self, item):
        if self.marked_id_PO != -1:
            f_name = item.data()
            self.thread.set_fielder_name_PO(f_name)
            self.marked_id_PO = -1
        self.clear_vbox(self.panel_val)

    def make_all_red(self):
        self.thread.make_all_red()
    
    def record(self):
        self.thread.record()

    def reswap(self):
            player_id = self.thread.find_clicked_player_id(
                self.current_mouse_pos[0],
                self.current_mouse_pos[1],
                self.img_size,
                self.frame_size)
            self.thread.reswap(player_id)

    def player_clicked(self, f_name):
        if self.panel_val == 1:
            if self.marked_id_TA != -1:
                self.thread.set_fielder_name_TA(f_name)
                self.marked_id_TA = -1
            self.clear_vbox(self.panel_val)
        elif self.panel_val == 2:
            if self.marked_id_TB != -1:
                self.thread.set_fielder_name_TB(f_name)
                self.marked_id_TB = -1
            self.clear_vbox(self.panel_val)

    def clear_vbox(self, panel_val):
        self.clearLayout(self.vbox_panel)
        if panel_val == 0:
            self.panel_0()
            self.thread.clear_naming_coloring()
            self.panel_val = 0
        elif panel_val == 1:
            self.panel_new(1)
            self.panel_val = 1
        elif panel_val == 2:
            self.panel_new(2)
            self.panel_val = 2
        elif panel_val == 3:
            self.panel_search()
            self.panel_val = 3

    def set_id_name(self, clear_vbox_val):
        player_id = self.thread.find_clicked_player_id(
            self.current_mouse_pos[0], self.current_mouse_pos[1], self.img_size, self.frame_size)
        if(player_id == -1):
            player_id = self.thread.find_clicked_FN_player_id(
                self.current_mouse_pos[0], self.current_mouse_pos[1], self.img_size, self.frame_size)
            if(player_id == -1):
                player_id = self.thread.find_clicked_FN_player_id_slipFielders(
                    self.current_mouse_pos[0], self.current_mouse_pos[1], self.img_size, self.frame_size)
                if player_id == -1:
                    player_id = self.thread.find_clicked_FN_player_id_outside_frame_a(
                        self.current_mouse_pos[0],
                        self.current_mouse_pos[1],
                        self.img_size,
                        self.frame_size)
                    if player_id == -1:
                        player_id = self.thread.find_clicked_FN_player_id_outside_frame_z(
                            self.current_mouse_pos[0],
                            self.current_mouse_pos[1],
                            self.img_size,
                            self.frame_size)
        np_ids = np.array(self.thread.identities)
        if player_id not in [-1, None] and player_id in np_ids:
            p_idx = np.where(np_ids == player_id)[0][0]
            # if str(player_id) in self.thread.fielder_dict_PO.keys():
            #     print("hope")
            #     del self.thread.fielder_dict_PO[str(player_id)]

            if self.thread.player_types[p_idx] == 0:
                if clear_vbox_val == 1:
                    if str(player_id) in self.thread.fielder_dict.keys():
                        print("hope")
                        del self.thread.fielder_dict[str(player_id)]
                    val = self.thread.set_clicked_fielder_TA(player_id)
                    self.marked_id_TA = player_id
                    if val != -1:
                        self.clear_vbox(1)
                    else:
                        self.clear_vbox(0)
                elif clear_vbox_val == 2:
                    if str(player_id) in self.thread.fielder_dict.keys():
                        print("hope")
                        del self.thread.fielder_dict[str(player_id)]
                    val = self.thread.set_clicked_fielder_TB(player_id)
                    self.marked_id_TB = player_id
                    if val != -1:
                        self.clear_vbox(2)
                    else:
                        self.clear_vbox(0)
                elif clear_vbox_val == 3:
                    if str(player_id) in self.thread.fielder_dict_PO.keys():
                        print("hope")
                        del self.thread.fielder_dict_PO[str(player_id)]
                    val = self.thread.set_clicked_fielder_PO(player_id)
                    self.marked_id_PO = player_id
                    if val != -1:
                        self.clear_vbox(3)
                    else:
                        self.clear_vbox(0)
        elif player_id in (list(self.thread.false_negatives.keys())+list(self.thread.false_negatives_outside_frame_z.keys())+list(self.thread.false_negatives_outside_frame_a.keys())+list(self.thread.false_negatives_slipFielders.keys())):
            if clear_vbox_val == 1:
                if str(player_id) in self.thread.fielder_dict.keys():
                    print("hope")
                    del self.thread.fielder_dict[str(player_id)]
                val = self.thread.set_clicked_fielder_TA(player_id)
                self.marked_id_TA = player_id
                if val != -1:
                    self.clear_vbox(1)
                else:
                    self.clear_vbox(0)
            elif clear_vbox_val == 2:
                if str(player_id) in self.thread.fielder_dict.keys():
                    print("hope")
                    del self.thread.fielder_dict[str(player_id)]
                val = self.thread.set_clicked_fielder_TB(player_id)
                self.marked_id_TB = player_id
                if val != -1:
                    self.clear_vbox(2)
                else:
                    self.clear_vbox(0)
            elif clear_vbox_val == 3:
                if str(player_id) in self.thread.fielder_dict_PO.keys():
                    print("hope")
                    del self.thread.fielder_dict[str(player_id)]
                val = self.thread.set_clicked_fielder_PO(player_id)
                self.marked_id_PO = player_id
                if val != -1:
                    self.clear_vbox(3)
                else:
                    self.clear_vbox(0)

    def InitialiseScreenSettings(self):
        self.screen_resolution_width = QDesktopWidget().availableGeometry().width()
        self.screen_resolution_height = QDesktopWidget().availableGeometry().height()

        self.stream_area_width = int(
            (78.6 * self.screen_resolution_width) / 100)
        self.stream_area_height = self.screen_resolution_height

        panel_area_width = self.screen_resolution_width - self.stream_area_width
        panel_area_height = self.screen_resolution_height

        self.img_size = [self.stream_area_width, self.stream_area_height]

        self.font_size = int((0.52 * self.screen_resolution_width) / 100)

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def closeEvent(self, event):
        self.thread.exithandler()
        event.accept()

    def highlight_players_streak(self):
        player_id = self.thread.find_clicked_player_id(
            self.current_mouse_pos[0], self.current_mouse_pos[1], self.img_size, self.frame_size)
        if player_id not in [-1, None]:
            self.thread.highlight_player_streak(player_id)

    # def set_batsmen_id(self):
    #     player_id = self.thread.find_clicked_batsman_player_id(
    #         self.current_mouse_pos[0], self.current_mouse_pos[1], self.img_size, self.frame_size)
    #     if player_id == -1:
    #         self.thread.create_batsmen_player(
    #             self.current_mouse_pos,
    #             self.img_size, self.frame_size)
    #     else:
    #         self.thread.remove_batsmen_player(player_id)
    def send_config_data(self):
        self.thread.send_config_flag = True

    def send_player_speed(self):
        player_id = self.thread.find_clicked_player_id(
            self.current_mouse_pos[0], self.current_mouse_pos[1], self.img_size, self.frame_size)
        np_ids = np.array(self.thread.identities)
        if player_id == -1 and self.thread.send_player_speed_id != -1:
            self.thread.send_player_speed_id = -1
            self.thread.send_player_speed_flag_downer = True
            self.thread.send_player_speed_val = []
            self.thread.speed_new_id = True
        if player_id == self.thread.send_player_speed_id:
            self.thread.send_player_speed_id = []
            self.thread.send_player_speed_flag_downer = True
            self.thread.send_player_speed_val = []
            self.thread.speed_new_id = True
        elif player_id not in [-1, None] and player_id in np_ids:
            self.thread.send_player_speed_val = []
            self.thread.speed_new_id = True
            self.thread.send_player_speed_id = player_id
            self.thread.send_player_speed_flag = True

    def set_bowler_id(self):
        player_id = self.thread.find_clicked_player_id(
            self.current_mouse_pos[0], self.current_mouse_pos[1], self.img_size, self.frame_size)
        np_ids = np.array(self.thread.identities)
        # print(self.thread.fielder_dict_PO)
        # del self.thread.fielder_dict_PO[str(player_id)]
        if player_id not in [-1, None] and player_id in np_ids:
            p_idx = np.where(np_ids == player_id)[0][0]
            # print("here0")

            if self.thread.player_types[p_idx] == 0:
                if player_id != self.thread.bowler_id and self.thread.bowler_id != -1 and self.thread.bowler_id in np_ids:
                    b_idx = np.where(np_ids == self.thread.bowler_id)[0][0]
                    if self.thread.player_types[b_idx] != 0:
                        self.thread.change_player_type(
                            self.thread.bowler_id, 0)

                self.thread.bowler_id = player_id
                if 'BOWLER' in list(self.thread.fielder_dict_PO.values()):
                    for key, value in dict(self.thread.fielder_dict_PO).items():
                        if value == 'BOWLER':
                            del self.thread.fielder_dict_PO[key]
                self.thread.fielder_dict_PO[str(player_id)] = 'BOWLER'

                self.thread.change_player_type(player_id, 9)
            elif player_id == self.thread.bowler_id:
                self.thread.change_player_type(self.thread.bowler_id, 0)
                self.thread.bowler_id = -1

        elif player_id == -1:
            # print("here1")
            if player_id != self.thread.bowler_id and self.thread.bowler_id != -1 and self.thread.bowler_id in np_ids:
                b_idx = np.where(np_ids == self.thread.bowler_id)[0][0]
                if self.thread.player_types[b_idx] != 0:
                    self.thread.change_player_type(self.thread.bowler_id, 0)

            player_id = self.thread.find_clicked_FN_player_id(
                self.current_mouse_pos[0], self.current_mouse_pos[1], self.img_size, self.frame_size)
            if(player_id == -1):
                player_id = self.thread.find_clicked_FN_player_id_slipFielders(
                    self.current_mouse_pos[0], self.current_mouse_pos[1], self.img_size, self.frame_size)
            if player_id == self.thread.bowler_id:
                self.thread.bowler_id = -1
            if (player_id != -1):
                self.thread.bowler_id = player_id
                if 'BOWLER' in list(self.thread.fielder_dict_PO.values()):
                    for key, value in dict(self.thread.fielder_dict_PO).items():
                        if value == 'BOWLER':
                            del self.thread.fielder_dict_PO[key]
                self.thread.fielder_dict_PO[str(player_id)] = 'BOWLER'
        
    def set_wk_id(self):
        player_id = self.thread.find_clicked_player_id(
            self.current_mouse_pos[0], self.current_mouse_pos[1], self.img_size, self.frame_size)
        np_ids = np.array(self.thread.identities)
        # print(self.thread.fielder_dict_PO)
        # del self.thread.fielder_dict_PO[str(player_id)]
        if player_id not in [-1, None] and player_id in np_ids:
            p_idx = np.where(np_ids == player_id)[0][0]
            # print("here0")

            if self.thread.player_types[p_idx] == 0:
                if player_id != self.thread.wk_id and self.thread.wk_id != -1 and self.thread.wk_id in np_ids:
                    b_idx = np.where(np_ids == self.thread.wk_id)[0][0]
                    if self.thread.player_types[b_idx] != 0:
                        self.thread.change_player_type(
                            self.thread.wk_id, 0)

                self.thread.wk_id = player_id
                if "WKT KEEPER" in list(self.thread.fielder_dict_PO.values()):
                    for key, value in dict(self.thread.fielder_dict_PO).items():
                        if value == "WKT KEEPER":
                            del self.thread.fielder_dict_PO[key]
                self.thread.fielder_dict_PO[str(player_id)] = "WKT KEEPER"
                
                self.thread.change_player_type(player_id, 0)
            elif player_id == self.thread.wk_id:
                self.thread.change_player_type(self.thread.wk_id, 0)
                self.thread.wk_id = -1

        elif player_id == -1:
            # print("here1")
            if player_id != self.thread.wk_id and self.thread.wk_id != -1 and self.thread.wk_id in np_ids:
                b_idx = np.where(np_ids == self.thread.wk_id)[0][0]
                if self.thread.player_types[b_idx] != 0:
                    self.thread.change_player_type(self.thread.wk_id, 0)

            player_id = self.thread.find_clicked_FN_player_id(
                self.current_mouse_pos[0], self.current_mouse_pos[1], self.img_size, self.frame_size)
            if(player_id == -1):
                player_id = self.thread.find_clicked_FN_player_id_slipFielders(
                    self.current_mouse_pos[0], self.current_mouse_pos[1], self.img_size, self.frame_size)
            if player_id == self.thread.wk_id:
                self.thread.wk_id = -1
            if (player_id != -1):
                self.thread.wk_id = player_id
                if "WKT KEEPER" in list(self.thread.fielder_dict_PO.values()):
                    for key, value in dict(self.thread.fielder_dict_PO).items():
                        if value == "WKT KEEPER":
                            del self.thread.fielder_dict_PO[key]
                self.thread.fielder_dict_PO[str(player_id)] = "WKT KEEPER"
                

        

    def set_dummy_connect(self):
        if len(self.thread.false_negatives_outside_frame_u) > 0 and self.thread.dummy_player_id == -1:
            player_id = self.thread.find_clicked_player_id(
                self.current_mouse_pos[0], self.current_mouse_pos[1], self.img_size, self.frame_size)
            if(player_id == -1):
                player_id = self.thread.find_clicked_FN_player_id(
                    self.current_mouse_pos[0], self.current_mouse_pos[1], self.img_size, self.frame_size)
                if(player_id == -1):
                    player_id = self.thread.find_clicked_FN_player_id_slipFielders(
                        self.current_mouse_pos[0], self.current_mouse_pos[1], self.img_size, self.frame_size)
                    if player_id == -1:
                        player_id = self.thread.find_clicked_FN_player_id_outside_frame_a(
                            self.current_mouse_pos[0],
                            self.current_mouse_pos[1],
                            self.img_size,
                            self.frame_size)
                        if player_id == -1:
                            player_id = self.thread.find_clicked_FN_player_id_outside_frame_z(
                                self.current_mouse_pos[0],
                                self.current_mouse_pos[1],
                                self.img_size,
                                self.frame_size)
            if(player_id != -1):
                if player_id == self.thread.dummy_connect_id:
                    self.thread.dummy_connect_id = -1
                else:
                    self.thread.dummy_connect_id = player_id

    def set_dummy_player(self):
        if len(self.thread.false_negatives_outside_frame_u) > 0 and self.thread.dummy_connect_id == -1:
            player_id = self.thread.find_clicked_player_id(
                self.current_mouse_pos[0], self.current_mouse_pos[1], self.img_size, self.frame_size)
            if(player_id == -1):
                player_id = self.thread.find_clicked_FN_player_id(
                    self.current_mouse_pos[0], self.current_mouse_pos[1], self.img_size, self.frame_size)
                if(player_id == -1):
                    player_id = self.thread.find_clicked_FN_player_id_slipFielders(
                        self.current_mouse_pos[0], self.current_mouse_pos[1], self.img_size, self.frame_size)
                    if player_id == -1:
                        player_id = self.thread.find_clicked_FN_player_id_outside_frame_a(
                            self.current_mouse_pos[0],
                            self.current_mouse_pos[1],
                            self.img_size,
                            self.frame_size)
                        if player_id == -1:
                            player_id = self.thread.find_clicked_FN_player_id_outside_frame_z(
                                self.current_mouse_pos[0],
                                self.current_mouse_pos[1],
                                self.img_size,
                                self.frame_size)

            if(player_id != -1):
                if player_id == self.thread.dummy_player_id:
                    self.thread.dummy_player_id = -1
                    self.thread.dummy_player_cache = [-1, -1]
                else:
                    self.thread.dummy_player_id = player_id
                    self.thread.dummy_player_cache = [-1, -1]
                    self.thread.dummy_player_falg = True
    
    def copy_scorefile_data(self):
        self.thread.copy_scorefile_data()

    def set_dist_pt_to_wkt(self):
        player_id = self.thread.find_clicked_FN_player_id_markpoint_o(
            self.current_mouse_pos[0], self.current_mouse_pos[1], self.img_size, self.frame_size)
        if(player_id == -1):
            self.thread.create_FN_player_mark_point_o(
                self.current_mouse_pos, self.img_size, self.frame_size)
            if([151, 124] not in self.dist_listt):
                self.thread.calculate_distance(151, 124)
                self.dist_listt.append([151, 124])

        else:
            self.thread.remove_FN_player_markpoint_o(player_id)
            i = -1
            for ele in self.dist_listt:
                i += 1
                if(151 in ele):
                    self.dist_listt.pop(i)
                    self.thread.remove_distance(i)

    def set_dist_fielder_to_wkt(self):
        print("function reached")
        player_id = self.thread.find_clicked_player_id(
            self.current_mouse_pos[0], self.current_mouse_pos[1], self.img_size, self.frame_size)
        if(player_id == -1):
            player_id = self.thread.find_clicked_FN_player_id(
                self.current_mouse_pos[0], self.current_mouse_pos[1], self.img_size, self.frame_size)
            if(player_id == -1):
                player_id = self.thread.find_clicked_FN_player_id_slipFielders(
                    self.current_mouse_pos[0], self.current_mouse_pos[1], self.img_size, self.frame_size)
                if player_id == -1:
                    player_id = self.thread.find_clicked_FN_player_id_outside_frame_a(
                        self.current_mouse_pos[0],
                        self.current_mouse_pos[1],
                        self.img_size,
                        self.frame_size)
                    if player_id == -1:
                        player_id = self.thread.find_clicked_FN_player_id_outside_frame_z(
                            self.current_mouse_pos[0],
                            self.current_mouse_pos[1],
                            self.img_size,
                            self.frame_size)
        if(player_id != -1):
            # print(self.b_clicked , player_id)

            i = -1
            for ele in self.dist_listt:
                i += 1

                if(124 in ele):
                    self.b_clicked = -1
                    # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!plyr",self.dist_listt[i])
                    self.dist_listt.pop(i)
                    self.thread.remove_distance(i)
                    break

            if(self.b_clicked == player_id):
                i = -1
                # print("disttt",self.dist_listt)
                for ele in self.dist_listt:
                    i += 1

                    if(player_id in ele):
                        self.b_clicked = -1
                        # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!plyr",self.dist_listt[i])
                        self.dist_listt.pop(i)
                        self.thread.remove_distance(i)
                        break
            else:
                self.thread.create_FN_player_mark_point_b(
                    player_id, self.img_size, self.frame_size)
                self.thread.calculate_distance(player_id, 124)
                self.dist_listt.append([player_id, 124])
                self.b_clicked = player_id


    def connect_players_line1(self):
        player_id = self.thread.find_clicked_player_id(
            self.current_mouse_pos[0],
            self.current_mouse_pos[1],
            self.img_size,
            self.frame_size)
        if player_id == -1:
            player_id = self.thread.find_clicked_FN_player_id(
                self.current_mouse_pos[0],
                self.current_mouse_pos[1],
                self.img_size,
                self.frame_size)
        if player_id == -1:
            player_id = self.thread.find_clicked_FN_player_id_slipFielders(
                self.current_mouse_pos[0],
                self.current_mouse_pos[1],
                self.img_size,
                self.frame_size)
        if player_id == -1:
            player_id = self.thread.find_clicked_FN_player_id_outside_frame_a(
                self.current_mouse_pos[0],
                self.current_mouse_pos[1],
                self.img_size,
                self.frame_size)
        if player_id == -1:
            player_id = self.thread.find_clicked_FN_player_id_outside_frame_z(
                self.current_mouse_pos[0],
                self.current_mouse_pos[1],
                self.img_size,
                self.frame_size)

        if player_id not in [-1, None]:
            if self.player_connect_count <= 5 and player_id not in self.player_connect_dict.values():
                self.player_connect_dict[self.player_connect_count] = player_id
                self.player_connect_count += 1
            else:
                self.player_connect_dict = {}
                self.player_connect_count = 1
            self.thread.update_player_connect(self.player_connect_dict)

    def add_gap_ids(self):
        player_id = self.thread.find_clicked_player_id(
            self.current_mouse_pos[0],
            self.current_mouse_pos[1],
            self.img_size,
            self.frame_size)
        if player_id == -1:
            player_id = self.thread.find_clicked_FN_player_id(
                self.current_mouse_pos[0],
                self.current_mouse_pos[1],
                self.img_size,
                self.frame_size)
        if player_id == -1:
            player_id = self.thread.find_clicked_FN_player_id_slipFielders(
                self.current_mouse_pos[0],
                self.current_mouse_pos[1],
                self.img_size,
                self.frame_size)

        if player_id == -1:
            player_id = self.thread.find_clicked_FN_player_id_outside_frame_a(
                self.current_mouse_pos[0],
                self.current_mouse_pos[1],
                self.img_size,
                self.frame_size)
        if player_id == -1:
            player_id = self.thread.find_clicked_FN_player_id_outside_frame_z(
                self.current_mouse_pos[0],
                self.current_mouse_pos[1],
                self.img_size,
                self.frame_size)

        if player_id not in [-1, None]:
            self.thread.set_gap_ids(player_id)

    def add_multi_gap_ids(self):
        player_id = self.thread.find_clicked_player_id(
            self.current_mouse_pos[0],
            self.current_mouse_pos[1],
            self.img_size,
            self.frame_size)
        if player_id == -1:
            player_id = self.thread.find_clicked_FN_player_id(
                self.current_mouse_pos[0],
                self.current_mouse_pos[1],
                self.img_size,
                self.frame_size)
        if player_id == -1:
            player_id = self.thread.find_clicked_FN_player_id_slipFielders(
                self.current_mouse_pos[0],
                self.current_mouse_pos[1],
                self.img_size,
                self.frame_size)

        if player_id == -1:
            player_id = self.thread.find_clicked_FN_player_id_outside_frame_a(
                self.current_mouse_pos[0],
                self.current_mouse_pos[1],
                self.img_size,
                self.frame_size)
        if player_id == -1:
            player_id = self.thread.find_clicked_FN_player_id_outside_frame_z(
                self.current_mouse_pos[0],
                self.current_mouse_pos[1],
                self.img_size,
                self.frame_size)

        if player_id not in [-1, None]:
            self.thread.set_multi_gap_ids(player_id)

    def add_ingap_ids(self):
        player_id = self.thread.find_clicked_player_id(
            self.current_mouse_pos[0],
            self.current_mouse_pos[1],
            self.img_size,
            self.frame_size)
        if player_id == -1:
            player_id = self.thread.find_clicked_FN_player_id(
                self.current_mouse_pos[0],
                self.current_mouse_pos[1],
                self.img_size,
                self.frame_size)
        if player_id == -1:
            player_id = self.thread.find_clicked_FN_player_id_slipFielders(
                self.current_mouse_pos[0],
                self.current_mouse_pos[1],
                self.img_size,
                self.frame_size)

        if player_id == -1:
            player_id = self.thread.find_clicked_FN_player_id_outside_frame_a(
                self.current_mouse_pos[0],
                self.current_mouse_pos[1],
                self.img_size,
                self.frame_size)
        if player_id == -1:
            player_id = self.thread.find_clicked_FN_player_id_outside_frame_z(
                self.current_mouse_pos[0],
                self.current_mouse_pos[1],
                self.img_size,
                self.frame_size)

        if player_id not in [-1, None]:
            self.thread.set_ingap_ids(player_id)

    def set_umpire_id(self):
        player_id = self.thread.find_clicked_player_id(
            self.current_mouse_pos[0],
            self.current_mouse_pos[1],
            self.img_size,
            self.frame_size)
        np_ids = np.array(self.thread.identities)
        if player_id not in [-1, None] and player_id in np_ids:
            p_idx = np.where(np_ids == player_id)[0][0]

            if self.thread.player_types[p_idx] != 0 and self.thread.player_types[p_idx] != 9:
                self.thread.set_umpire_id(player_id)

    def set_batsmen_id(self):
        player_id = self.thread.find_clicked_player_id(
            self.current_mouse_pos[0],
            self.current_mouse_pos[1],
            self.img_size,
            self.frame_size)
        np_ids = np.array(self.thread.identities)
        if player_id not in [-1, None] and player_id in np_ids:
            p_idx = np.where(np_ids == player_id)[0][0]
            
            if self.thread.player_types[p_idx] != 0 and self.thread.player_types[p_idx] != 9:
                
                self.thread.set_batsmen_id(player_id)

    def false_negative_clicked(self):
        player_id = self.thread.find_clicked_FN_player_id(
            self.current_mouse_pos[0],
            self.current_mouse_pos[1],
            self.img_size,
            self.frame_size)
        if player_id == -1:
            self.thread.create_FN_player(
                self.current_mouse_pos,
                self.img_size,
                self.frame_size)
        else:
            self.thread.remove_FN_player(player_id)

    def mark_point_clicked(self):
        player_id = self.thread.find_clicked_FN_player_id_markpoint_p(
            self.current_mouse_pos[0],
            self.current_mouse_pos[1],
            self.img_size,
            self.frame_size)
        if player_id == -1:
            self.thread.create_FN_player_mark_point_p(
                self.current_mouse_pos,
                self.img_size,
                self.frame_size)
        else:
            self.thread.remove_FN_player_markpoint_p(player_id)

    def false_negative_slip_fielders_clicked(self):
        player_id = self.thread.find_clicked_FN_player_id_slipFielders(
            self.current_mouse_pos[0],
            self.current_mouse_pos[1],
            self.img_size,
            self.frame_size)
        if player_id == -1:
            self.thread.create_FN_player_slipFielders(
                self.current_mouse_pos,
                self.img_size,
                self.frame_size)
        else:
            self.thread.remove_FN_player_slipFielders(player_id)

    def false_negative_outside_frame_z(self):
        player_id = self.thread.find_clicked_FN_player_id_outside_frame_z(
            self.current_mouse_pos[0],
            self.current_mouse_pos[1],
            self.img_size,
            self.frame_size)
        if(player_id == -1):
            self.thread.create_FN_player_outside_frame_z(
                self.current_mouse_pos,
                self.img_size,
                self.frame_size)
        else:
            self.thread.remove_FN_player_outside_frame_z(player_id)

    def false_negative_outside_frame_a(self):
        player_id = self.thread.find_clicked_FN_player_id_outside_frame_a(
            self.current_mouse_pos[0],
            self.current_mouse_pos[1],
            self.img_size,
            self.frame_size)
        print(player_id)
        if(player_id == -1):
            self.thread.create_FN_player_outside_frame_a(
                self.current_mouse_pos,
                self.img_size,
                self.frame_size)
        else:
            self.thread.remove_FN_player_outside_frame_a(player_id)

    def false_negative_outside_frame_u(self):
        player_id = self.thread.find_clicked_FN_player_id_outside_frame_u(
            self.current_mouse_pos[0],
            self.current_mouse_pos[1],
            self.img_size,
            self.frame_size)
        # print("clicked C -",player_id)
        if(player_id == -1):
            self.thread.create_FN_player_outside_frame_u(
                self.current_mouse_pos,
                self.img_size,
                self.frame_size)
        else:
            self.thread.remove_FN_player_outside_frame_u(player_id)

    def calculate_distance_clicked(self):
        player_id = self.thread.find_clicked_player_id(
            self.current_mouse_pos[0],
            self.current_mouse_pos[1],
            self.img_size,
            self.frame_size)
        player_id = self.thread.find_clicked_player_id(
            self.current_mouse_pos[0],
            self.current_mouse_pos[1],
            self.img_size,
            self.frame_size)
        if player_id == -1:
            player_id = self.thread.find_clicked_FN_player_id(
                self.current_mouse_pos[0],
                self.current_mouse_pos[1],
                self.img_size,
                self.frame_size)
        if player_id == -1:
            player_id = self.thread.find_clicked_FN_player_id_slipFielders(
                self.current_mouse_pos[0],
                self.current_mouse_pos[1],
                self.img_size,
                self.frame_size)
        if player_id == -1:
            player_id = self.thread.find_clicked_FN_player_id_markpoint_p(
                self.current_mouse_pos[0],
                self.current_mouse_pos[1],
                self.img_size,
                self.frame_size)
        if player_id == -1:
            player_id = self.thread.find_clicked_FN_player_id_outside_frame_a(
                self.current_mouse_pos[0],
                self.current_mouse_pos[1],
                self.img_size,
                self.frame_size)
        if player_id == -1:
            player_id = self.thread.find_clicked_FN_player_id_outside_frame_z(
                self.current_mouse_pos[0],
                self.current_mouse_pos[1],
                self.img_size,
                self.frame_size)
        # if player_id == -1:
        #     player_id = self.thread.find_clicked_batsman_player_id(
        #         self.current_mouse_pos[0],
        #         self.current_mouse_pos[1],
        #         self.img_size,
        #         self.frame_size)
        if player_id not in [-1, None]:
            if len(self.dist_ids) == 0:
                if self.dist_listt == []:
                    self.dist_ids.append(player_id)  # len(self.dist_ids) == 1
                    self.thread.activate_distance(player_id)
                else:
                    i = -1
                    for ele in self.dist_listt:
                        i += 1
                        if player_id in ele:
                            self.dist_listt.pop(i)
                            self.thread.remove_distance(i)
                            self.dist_ids = []
                            break

            elif len(self.dist_ids) == 1:
                self.dist_ids.append(player_id)  # len(self.dist_ids) == 1
                self.thread.calculate_distance(
                    self.dist_ids[0],
                    self.dist_ids[1])
                self.dist_listt.append(self.dist_ids)
                self.thread.activate_distance(-1)
                self.dist_ids = []

    def button_captureframes_clicked(self):
        if self.capture_frame is True:
            self.capture_frame = False
            self.thread.capture_frames = False
            self.button_captureframes.setStyleSheet(
                "background-color: #2234a8; color:white")
        else:
            self.capture_frame = True
            self.thread.capture_frames = True
            self.button_captureframes.setStyleSheet("background-color: red;")

    def button_resetHighlight_clicked(self):
        self.player_connect_count = 1
        self.player_connect_dict = {}
        self.thread.update_player_connect(self.player_connect_dict)
        self.thread.reset_Highlight_flager()
        self.thread.sort_tracker.reset_swap_flags()
        self.thread.reset_HighlightStreak()
        self.thread.reset_FN_highlights()
        self.dist_listt = []
        self.dist_ids = []
        self.thread.remove_distance(-1)
        self.b_clicked = -1

    def button_resetdistancehighlight_clicked(self):
        self.thread.reset_Highlight()
        self.thread.reset_HighlightStreak()
        self.thread.reset_FN_highlights()
        self.dist_listt = []
        self.dist_ids = []
        self.thread.remove_distance(-1)

    def button_outside_clicked(self):
        self.thread.activate_count_outside_player()
        self.outside_player_active = not self.outside_player_active

    def reset_ui(self):
        self.frame_count = 0
        self.swap_ids = []
        self.player_type_change_flag = False
        self.player_type_change_id = -1
        self.player_type_change_box_coords = []
        self.thread.flip_field_plot = 0

    def button_reset_clicked(self):
        self.thread.reset_handler_flager()
        self.reset_ui()

    def button_clicked_near(self):
        self.button_near_end.setStyleSheet("background-color: green;")
        self.button_far_end.setStyleSheet(
            "background-color: #2234a8;color:white;")
        self.thread.flip_field_plot = 0

    def button_clicked_far(self):
        self.button_far_end.setStyleSheet("background-color: green;")
        self.button_near_end.setStyleSheet(
            "background-color: #2234a8;color:white;")
        self.thread.flip_field_plot = 1

    def button_clicked_left(self):
        if self.lr_automated is False:
            self.button_left.setStyleSheet("background-color: green;")
            self.button_right.setStyleSheet(
                "background-color: #2234a8;color:white;")
            flip_batsmen_pos = 0
            self.thread.batsmen_pos_flip(flip_batsmen_pos)

    def button_clicked_right(self):
        if self.lr_automated is False:
            self.button_right.setStyleSheet("background-color: green;")
            self.button_left.setStyleSheet(
                "background-color: #2234a8;color:white;")
            flip_batsmen_pos = 1
            self.thread.batsmen_pos_flip(flip_batsmen_pos)

    def button_clicked_lr_automated(self):
        if self.lr_automated is True:
            self.lr_automated = False
            self.button_lr_automated.setStyleSheet(
                "background-color: #2234a8;")
        else:
            self.lr_automated = True
            self.button_lr_automated.setStyleSheet(
                "background-color: green;")
        self.thread.update_lr_automation(self.lr_automated)

    # def button_clicked_fetch_db(self):
    #     over_value = self.le_db_o.text()
    #     ball_value = self.le_db_b.text()
    #     print("Fetching data from...", over_value, "to...", ball_value)
    #     self.thread.get_mongoData(over_value, ball_value)
    #     self.le_db_o.setText("")
    #     self.le_db_b.setText("")

    def get_pos(self, event):
        if event.buttons() == QtCore.Qt.MiddleButton:
            pointx = event.pos().x()
            pointy = event.pos().y()
            player_id = self.thread.find_clicked_player_id(
                pointx, pointy, self.img_size, self.frame_size)
            if player_id != -1 and player_id is not None:
                self.thread.highlight_player(player_id)
            if player_id == -1:
                player_id = self.thread.find_clicked_FN_player_id(
                    self.current_mouse_pos[0],
                    self.current_mouse_pos[1],
                    self.img_size,
                    self.frame_size)
                if player_id != -1:
                    self.thread.highlight_fn(player_id)
            if player_id == -1:
                player_id = self.thread.find_clicked_FN_player_id_slipFielders(
                    self.current_mouse_pos[0],
                    self.current_mouse_pos[1],
                    self.img_size, self.frame_size)
                if player_id != -1:
                    self.thread.highlight_fn(player_id)

        elif event.buttons() == QtCore.Qt.LeftButton:
            pointx = event.pos().x()
            pointy = event.pos().y()
            player_id = self.thread.find_clicked_player_id(
                pointx, pointy, self.img_size, self.frame_size)
            if player_id != -1 and player_id is not None:
                # print("here",player_id)
                self.thread.change_player_type(player_id, 0)

        elif event.buttons() == QtCore.Qt.RightButton:
            pointx = event.pos().x()
            pointy = event.pos().y()
            player_id = self.thread.find_clicked_player_id(
                pointx, pointy, self.img_size, self.frame_size)
            if player_id != -1 and player_id is not None:
                if player_id in self.thread.umpire_id:
                    del self.thread.fielder_dict_PO[str(player_id)] 
                    self.thread.umpire_id.remove(player_id)
                if player_id in self.thread.batsmen_ids_automated:
                    del self.thread.fielder_dict_PO[str(player_id)] 
                    self.thread.batsmen_ids_automated.remove(player_id)
                # print("here",player_id)
                self.thread.change_player_type(player_id, 3)

                if self.panel_val != 0:
                    print("right clicked player")
                    self.clear_vbox(self.panel_val)

    def handleMouseHover(self, event):
        self.current_mouse_pos = (event.x(), event.y())

    def assign(self):
        self.thread.go_green()

    @pyqtSlot(np.ndarray, dict)
    def update_image(self, cv_img, current_coords):
        # Post processing
        self.center()
        ht, wd, _ = cv_img.shape
        self.frame_size = [wd, ht]
        if self.player_type_change_flag is True and self.player_type_change_id in current_coords:
            player_type_change_coords = current_coords[self.player_type_change_id]

        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)

        self.le_drop.setText("DROP  :   " + str(self.thread.dropped_frames))
        self.tl_fielder.setText("FIELDER : " + str(self.thread.fielder_count))
        self.tl_others.setText("OTHERS : " + str(self.thread.others_count))
        self.tl_fn.setText("FALSE NEG : " + str(self.thread.fn_count))

        self.tl_maxid.setText(str(int(self.thread.max_id)))
        if int(self.thread.max_id) > 80:
            self.tl_maxid.setStyleSheet(
                "QLabel {background-color: red; color : white;}")
        else:
            self.tl_maxid.setStyleSheet(
                "QLabel {background-color: green; color : white;}")
        if self.lr_automated:
            if self.thread.left_handed is True:
                self.button_left.setStyleSheet("background-color: green;")
                self.button_right.setStyleSheet(
                    "background-color: #2234a8;color:white;")
            else:
                self.button_left.setStyleSheet(
                    "background-color: #2234a8;color:white;")
                self.button_right.setStyleSheet("background-color: green;")

            if self.thread.flip_field_plot == 0:
                self.button_near_end.setStyleSheet("background-color: green;")
                self.button_far_end.setStyleSheet(
                    "background-color: #2234a8;color:white;")
            else:
                self.button_far_end.setStyleSheet("background-color: green;")
                self.button_near_end.setStyleSheet(
                    "background-color: #2234a8;color:white;")

        # if self.outside_player_active:
        #     pl_count = self.thread.outside_circle_players
        #     if int(pl_count) <= 2:
        #         self.outside_circle_no.setStyleSheet(
        #             "QLineEdit {background-color: green; color : white;font-weight: bold;}")
        #     else:
        #         self.outside_circle_no.setStyleSheet(
        #             "QLineEdit {background-color: red; color : white;font-weight: bold;}")
        #     self.outside_circle_no.setText(str(pl_count))
        #     self.button_outside.setStyleSheet("background-color: green;")
        # else:
        #     self.outside_circle_no.setText('0')
        #     self.button_outside.setStyleSheet(
        #         "background-color: #2234a8;color: white;font-weight: bold;")
        livelock_sts = self.thread.livelock_status
        ids_saved_sts = self.thread.livelock_ids_saved

        if livelock_sts and ids_saved_sts:
            self.button_livelock.setStyleSheet(
                "background-color: green;color: white;font-weight: bold;")
        elif livelock_sts and ids_saved_sts is False:
            self.button_livelock.setStyleSheet(
                "background-color: red;color: white;font-weight: bold;")
        elif livelock_sts is False:
            self.button_livelock.setStyleSheet(
                "background-color: #2234a8;color: white;font-weight: bold;")

        insertDb_sts = self.thread.insertDb_status

        if insertDb_sts:
            self.button_insert_db.setStyleSheet(
                "background-color: green;color: white;font-weight: bold;")
        else:
            self.button_insert_db.setStyleSheet(
                "background-color: #2234a8;color: white;font-weight: bold;")

        self.tl_total.setText(
            str(self.thread.fielder_count + self.thread.fn_count))
        if int(self.thread.fielder_count + self.thread.fn_count) != 11:
            self.tl_total.setStyleSheet(
                "QLabel {background-color: red; color : white;}")
        else:
            self.tl_total.setStyleSheet(
                "QLabel {background-color: green; color : white;}")

        air_init = self.thread.air_init
        air_status = self.thread.air_status
        if air_init == False and air_status:
            self.button_air.setStyleSheet(
                "background-color: green;color: white;font-weight: bold;")
        elif air_init and air_status:
            self.button_air.setStyleSheet(
                "background-color: red;color: white;font-weight: bold;")
        elif air_init and not air_status:
            self.button_air.setStyleSheet(
                "background-color: #2234a8;color: white;font-weight: bold;")

        if self.front_end_mode.isChecked():
            self.front_end_mode.setStyleSheet(
                "background-color: green ;color: white;font-weight: bold;")
        else:
            self.front_end_mode.setStyleSheet(
                "background-color: #2234a8;color: white;font-weight: bold;")

    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        cv_img = cv2.resize(
            cv_img, (self.stream_area_width, self.stream_area_height))
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
