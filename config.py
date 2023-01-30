from base64 import standard_b64decode
import sys
import json
import os
from pathlib import Path

from PyQt5 import QtCore, QtGui
from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QTabWidget,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QRadioButton,
    QButtonGroup,
    QLabel,
    QFormLayout,
    QTextEdit,
    QLineEdit,
    QPushButton,
    QGridLayout,
    QFileDialog,
    QComboBox,
)
from cv2 import QT_PUSH_BUTTON

class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Config")
        # self.resize(1080, 400)
        # Create a top-level layout
        # self.layout = QWidget()
        
        # Create the tab widget with two tabs
        self.tabs = QGridLayout()

        with open("Settings/config.json", "r") as f:
            param_dict = json.load(f)

        ##HARDWARE
        self.cardtype = QLabel("Decklink Type:")
        self.card1 = QRadioButton("Decklink 4K Extreme 12G")
        if(param_dict['decklink_12g_mode'] == 1):
            self.card1.setChecked(True)
        # self.card1.setChecked(True)
        self.card1.toggled.connect(lambda:self.cardstate(self.card1))
        self.tabs.addWidget(self.cardtype,1,0)
        self.tabs.addWidget(self.card1,1,1)
        # layout.setVerticalSpacing(0.1)    
        self.card2 = QRadioButton("Decklink 8K Pro")
        if(param_dict['decklink_12g_mode'] == 0):
            self.card2.setChecked(True)
        self.card2.toggled.connect(lambda:self.cardstate(self.card2))
        self.tabs.addWidget(self.card2,1,2)

        # layout.setVerticalSpacing(1)
        # layout2 = QVBoxLayout()
        self.porttype = QLabel("Port Number:")
        self.port1 = QRadioButton("Port 1")
        if(param_dict['decklink_port_no'] == 0):
            self.port1.setChecked(True)
        self.port1.toggled.connect(lambda:self.portstate(self.port1))
        self.tabs.addWidget(self.porttype,2,0)
        self.tabs.addWidget(self.port1,2,1)
        # layout.setVerticalSpacing(0.1)    
        self.port2 = QRadioButton("Port 2")
        if(param_dict['decklink_port_no'] == 1):
            self.port2.setChecked(True)
        # self.port2.setChecked(True)
        self.port2.toggled.connect(lambda:self.portstate(self.port2))
        self.tabs.addWidget(self.port2,2,2)
        # layout.setVerticalSpacing(0.1)
        self.port3 = QRadioButton("Port 3")
        if(param_dict['decklink_port_no'] == 2):
            self.port3.setChecked(True)
        self.port3.toggled.connect(lambda:self.portstate(self.port3))
        self.tabs.addWidget(self.port3,2,3)
        # layout.setVerticalSpacing(0.1)
        self.port4 = QRadioButton("Port 4")
        if(param_dict['decklink_port_no'] == 3):
            self.port4.setChecked(True)
        self.port4.toggled.connect(lambda:self.portstate(self.port4))
        self.tabs.addWidget(self.port4,2,4)

        self.btngroup1 = QButtonGroup()
        self.btngroup2 = QButtonGroup()
        self.btngroup1.addButton(self.card1)
        self.btngroup1.addButton(self.card2)
        self.btngroup2.addButton(self.port1)
        self.btngroup2.addButton(self.port2)
        self.btngroup2.addButton(self.port3)
        self.btngroup2.addButton(self.port4)
        # layout.addLayout(layout1)
        # layout.addLayout(layout2)
        # generalTab.setLayout(layout)
        # return generalTab

        ##stadium name
        self.stadiumlabel = QLabel("Stadium/City Name")
        self.stadiumedit = QLineEdit(self)
        stadiumdef = param_dict['stadium']
        self.stadiumedit.setText(str(stadiumdef))
        self.stadiumsave = QPushButton("save",self)
        self.stadiumsave.clicked.connect(lambda: self.stadium_save_clicked())
        
        self.tabs.addWidget(self.stadiumlabel,3,0)
        self.tabs.addWidget(self.stadiumedit,3,1)
        self.tabs.addWidget(self.stadiumsave,3,2)

        ##IP TAB
        self.vizipf = QLabel("VIZ Frontend IP:")
        self.vizipf_input = QLineEdit(self)
        frontip = param_dict['viz_udp_ip_address']
        self.vizipf_input.setText(str(frontip))
        self.vizipf_save = QPushButton("save",self)
        self.vizipf_save.clicked.connect(lambda: self.vizipf_save_clicked())

        self.tabs.addWidget(self.vizipf,4,0)
        self.tabs.addWidget(self.vizipf_input,4,1)
        self.tabs.addWidget(self.vizipf_save,4,2)

        self.vizudpp = QLabel("VIZ UDP Port:")
        self.vizudpp_input = QLineEdit(self)
        frontport = param_dict['viz_udp_port']
        self.vizudpp_input.setText(str(frontport))
        self.vizudpp_save = QPushButton("save",self)
        self.vizudpp_save.clicked.connect(lambda: self.vizudp_save_clicked())

        self.tabs.addWidget(self.vizudpp,5,0)
        self.tabs.addWidget(self.vizudpp_input,5,1)
        self.tabs.addWidget(self.vizudpp_save,5,2)

        self.viztcpp = QLabel("VIZ TCP Port:")
        self.viztcpp_input = QLineEdit(self)
        tcpport = param_dict['viz_tcp_port']
        self.viztcpp_input.setText(str(tcpport))
        self.viztcpp_save = QPushButton("save",self)
        self.viztcpp_save.clicked.connect(lambda: self.viztcp_save_clicked())

        self.tabs.addWidget(self.viztcpp,6,0)
        self.tabs.addWidget(self.viztcpp_input,6,1)
        self.tabs.addWidget(self.viztcpp_save,6,2)

        self.downudp = QLabel("Downstream UDP IP/Port :")
        self.downudp_input = QLineEdit(self)
        downip = param_dict['middleman_ip_address_port']
        self.downudp_input.setText(str((downip.split(':')[1]).split('/')[2]))
        self.downudp_save = QPushButton("save",self)
        self.downudp_save.clicked.connect(lambda: self.downudp_save_clicked())

        self.tabs.addWidget(self.downudp,7,0)
        self.tabs.addWidget(self.downudp_input,7,1)
        self.tabs.addWidget(self.downudp_save,7,2)

        self.unrealipf = QLabel("Unreal IP:")
        self.unrealipf_input = QLineEdit(self)
        ue4ip = param_dict['unreal_ip_address']
        self.unrealipf_input.setText(str(ue4ip))
        self.unrealipf_save = QPushButton("save",self)
        self.unrealipf_save.clicked.connect(lambda: self.ue4_ipf_save_clicked())

        self.tabs.addWidget(self.unrealipf,8,0)
        self.tabs.addWidget(self.unrealipf_input,8,1)
        self.tabs.addWidget(self.unrealipf_save,8,2)

        self.unrealudpp = QLabel("Unreal UDP Port:")
        self.unrealudpp_input = QLineEdit(self)
        ue4udp = param_dict['unreal_udp']
        self.unrealudpp_input.setText(str(ue4udp))
        self.unrealudpp_save = QPushButton("save",self)
        self.unrealudpp_save.clicked.connect(lambda: self.ue4_udp_save_clicked())

        self.tabs.addWidget(self.unrealudpp,9,0)
        self.tabs.addWidget(self.unrealudpp_input,9,1)
        self.tabs.addWidget(self.unrealudpp_save,9,2)

        self.unrealtcpp = QLabel("Unreal TCP Port:")
        self.unrealtcpp_input = QLineEdit(self)
        ue4tcp = param_dict['unreal_tcp']
        self.unrealtcpp_input.setText(str(ue4tcp))
        self.unrealtcpp_save = QPushButton("save",self)
        self.unrealtcpp_save.clicked.connect(lambda: self.ue4_tcp_save_clicked())

        self.tabs.addWidget(self.unrealtcpp,10,0)
        self.tabs.addWidget(self.unrealtcpp_input,10,1)
        self.tabs.addWidget(self.unrealtcpp_save,10,2)

        # self.historylabel = QLabel("History Plot IP/Port :")
        # self.history_input = QLineEdit(self)
        # self.history_input.setPlaceholderText('Example: 192.168.8.8:4000')
        # self.history_save = QPushButton("save",self)
        # self.history_save.clicked.connect(lambda: self.history_save_clicked())

        # self.tabs.addWidget(self.historylabel,7,0)
        # self.tabs.addWidget(self.history_input,7,1)
        # self.tabs.addWidget(self.history_save,7,2)

        ##SCORING TAB
        # scoringTab = QWidget()
        # layout = QGridLayout()

        self.remotepc = QLabel("Scoring File Network Machine Folder:")
        self.remotepcpath = QLineEdit(self)
        self.remotepcpath.setReadOnly(True)
        self.remotepcpath.setText(str(param_dict["remote_pc_score_path"]))
        self.remotepcbtn = QPushButton('Select Folder',self)
        self.remotepcbtn.clicked.connect(lambda: self.remotepcpathbtn())
        self.tabs.addWidget(self.remotepc,11,0)
        self.tabs.addWidget(self.remotepcpath,11,1)
        self.tabs.addWidget(self.remotepcbtn,11,2)

        self.scorefile = QLabel("Score File mode:")
        self.scorefilemodetype = QLineEdit(self)
        scoremode = param_dict['score_file_mode']
        self.scorefilemodetype.setText(str(scoremode))
        self.scorefilesave = QPushButton("save",self)
        self.scorefilesave.clicked.connect(lambda: self.scorefilemode())

        self.tabs.addWidget(self.scorefile,12,0)
        self.tabs.addWidget(self.scorefilemodetype,12,1)
        self.tabs.addWidget(self.scorefilesave,12,2)

        self.wtscorefile = QLabel("WT scoring file name:")
        self.wtscorefilename = QLineEdit(self)
        scorefilename = param_dict['wt_score_file']
        self.wtscorefilename.setText(str(scorefilename))
        self.wtscorefilesave = QPushButton("save",self)
        self.wtscorefilesave.clicked.connect(lambda: self.wtscorename())

        self.tabs.addWidget(self.wtscorefile,13,0)
        self.tabs.addWidget(self.wtscorefilename,13,1)
        self.tabs.addWidget(self.wtscorefilesave,13,2)
        
        ##MATCH REQ
        self.weightlabel = QLabel("Detection Weights:")
        self.weights = QComboBox()
        self.weights.addItem("Select Weight")
        self.weight_list = self.folder_list_with_extension()
        for item in self.weight_list:
            self.weights.addItem(item)    
        self.weights.currentIndexChanged.connect(self.weights_change)
    
        self.tabs.addWidget(self.weightlabel,14,0)
        self.tabs.addWidget(self.weights,14,1)

        self.innings = QLabel("Innings:")
        self.inningsedit = QLineEdit(self)
        inningno = param_dict['innings']
        self.inningsedit.setText(str(inningno))
        self.inningssave = QPushButton("save",self)
        self.inningssave.clicked.connect(lambda: self.inningsfunc())

        self.tabs.addWidget(self.innings,15,0)
        self.tabs.addWidget(self.inningsedit,15,1)
        self.tabs.addWidget(self.inningssave,15,2)

        self.dbname = QLabel("Database Name:")
        self.dbnameedit = QLineEdit(self)
        dbnamedef = param_dict['db_name']
        self.dbnameedit.setText(str(dbnamedef))
        self.dbnamesave = QPushButton("save",self)
        self.dbnamesave.clicked.connect(lambda: self.dbnamefunc())

        self.tabs.addWidget(self.dbname,16,0)
        self.tabs.addWidget(self.dbnameedit,16,1)
        self.tabs.addWidget(self.dbnamesave,16,2)

        self.mforpix = QLabel("Pixel to Distance Value:")
        self.mforpixedit = QLineEdit(self)
        mforpixdef = param_dict['m_for_pix']
        self.mforpixedit.setText(str(mforpixdef))
        self.mforpixsave = QPushButton("save",self)
        self.mforpixsave.clicked.connect(lambda: self.mforpixfunc())

        self.tabs.addWidget(self.mforpix,17,0)
        self.tabs.addWidget(self.mforpixedit,17,1)
        self.tabs.addWidget(self.mforpixsave,17,2)

        ##HISTORY
        self.historydbname = QLabel("History Database Name:")
        self.historydbnameedit = QLineEdit(self)
        # self.historydbnameedit = str(self.historydbnameedit) + ".json"
        historynamedef = param_dict['history_data_path']
        # historynamedef = str(historynamedef.split('.'))
        self.historydbnameedit.setText(str(historynamedef.split('.')[0]))
        self.historydbnamesave = QPushButton("save",self)
        self.historydbnamesave.clicked.connect(lambda: self.historydbnamefunc())

        self.tabs.addWidget(self.historydbname,18,0)
        self.tabs.addWidget(self.historydbnameedit,18,1)
        self.tabs.addWidget(self.historydbnamesave,18,2)

        ##MISCELLENOUS
        self.cropmode = QLabel("Crop Mode:")
        self.cropmodetype1 = QRadioButton("Activated")
        if(param_dict['activate_crop'] == 1):
            self.cropmodetype1.setChecked(True)
        self.cropmodetype1.toggled.connect(lambda:self.cropactivated(self.cropmodetype1))
        self.cropmodetype2 = QRadioButton("Deactivated")
        if(param_dict['activate_crop'] == 0):
            self.cropmodetype2.setChecked(True)
        self.cropmodetype2.toggled.connect(lambda:self.cropactivated(self.cropmodetype2))

        self.tabs.addWidget(self.cropmode,19,0)
        self.tabs.addWidget(self.cropmodetype1,19,1)
        self.tabs.addWidget(self.cropmodetype2,19,2)
        self.btngroup3 = QButtonGroup()
        self.btngroup3.addButton(self.cropmodetype1)
        self.btngroup3.addButton(self.cropmodetype2)

        self.collisonmode = QLabel("Player dots shrink mode:")
        self.collisonmodetype1 = QRadioButton("Activated")
        if(param_dict['collision_mode'] == 1):
            self.collisonmodetype1.setChecked(True)
        self.collisonmodetype1.toggled.connect(lambda:self.collisonmodefunc(self.collisonmodetype1))
        self.collisonmodetype2 = QRadioButton("Deactivated")
        if(param_dict['collision_mode'] == 0):
            self.collisonmodetype2.setChecked(True)
        self.collisonmodetype2.toggled.connect(lambda:self.collisonmodefunc(self.collisonmodetype2))

        self.tabs.addWidget(self.collisonmode,20,0)
        self.tabs.addWidget(self.collisonmodetype1,20,1)
        self.tabs.addWidget(self.collisonmodetype2,20,2)
        self.btngroup4 = QButtonGroup()
        self.btngroup4.addButton(self.collisonmodetype1)
        self.btngroup4.addButton(self.collisonmodetype2)

        self.lensdistortion = QLabel("Lens Distortion:")
        self.lensdistortiony = QRadioButton("Activated")
        if(param_dict['lens_distortion'] == 1):
            self.lensdistortiony.setChecked(True)
        self.lensdistortiony.toggled.connect(lambda:self.lensdistortionfunc(self.collisonmodetype1))
        self.lensdistortionn = QRadioButton("Deactivated")
        if(param_dict['lens_distortion'] == 0):
            self.lensdistortionn.setChecked(True)
        self.lensdistortionn.toggled.connect(lambda:self.lensdistortionfunc(self.collisonmodetype2))

        self.tabs.addWidget(self.lensdistortion,21,0)
        self.tabs.addWidget(self.lensdistortiony,21,1)
        self.tabs.addWidget(self.lensdistortionn,21,2)
        self.btngroup5 = QButtonGroup()
        self.btngroup5.addButton(self.lensdistortiony)
        self.btngroup5.addButton(self.lensdistortionn)

        ##DEV
        self.camerasource = QLabel("Camera Source")
        self.camerasourceedit1 = QRadioButton("Input from Decklink(Camera/Shogun Feed)")
        if(param_dict['camera_model'] == 2):
            self.camerasourceedit1.setChecked(True)
        self.camerasourceedit1.toggled.connect(lambda:self.camerasourcefunc(self.camerasourceedit1))
        self.camerasourceedit2 = QRadioButton("Local Video")
        if(param_dict['camera_model'] == 1):
            self.camerasourceedit2.setChecked(True)
        self.camerasourceedit2.toggled.connect(lambda:self.camerasourcefunc(self.camerasourceedit2))

        self.tabs.addWidget(self.camerasource,22,0)
        self.tabs.addWidget(self.camerasourceedit1,22,1)
        self.tabs.addWidget(self.camerasourceedit2,22,2)
        self.btngroup6 = QButtonGroup()
        self.btngroup6.addButton(self.camerasourceedit1)
        self.btngroup6.addButton(self.camerasourceedit2)

        self.videofile = QLabel("Local Video")
        self.videofileedit = QPushButton()
        # self.videofileedit.addItem("Select Video")
        self.videofileedit.clicked.connect(self.getFileName)
        # for item in self.videolist:
        #     self.videofileedit.addItem(item)    
        # self.videofileedit.currentIndexChanged.connect(self.video_change)
    
        self.tabs.addWidget(self.videofile,23,0)
        self.tabs.addWidget(self.videofileedit,23,1)

        self.udpcommand = QLabel("Print UDP Command")
        self.udpcommandy = QRadioButton("Yes")
        if(param_dict['print_udp_command'] == 1):
            self.udpcommandy.setChecked(True)
        self.udpcommandy.toggled.connect(lambda:self.udpcommandfunc(self.udpcommandy))
        self.udpcommandn = QRadioButton("No")
        if(param_dict['print_udp_command'] == 0):
            self.udpcommandn.setChecked(True)
        self.udpcommandn.toggled.connect(lambda:self.udpcommandfunc(self.udpcommandn))

        self.tabs.addWidget(self.udpcommand,24,0)
        self.tabs.addWidget(self.udpcommandy,24,1)
        self.tabs.addWidget(self.udpcommandn,24,2)
        self.btngroup7 = QButtonGroup()
        self.btngroup7.addButton(self.udpcommandy)
        self.btngroup7.addButton(self.udpcommandn)

        self.entryflag = QLabel("Clear Multiple Entry Flag in History data")
        self.entryflagy = QRadioButton("Yes")
        if(param_dict['clear_multiple_entry_flag'] == 1):
            self.entryflagy.setChecked(True)
        self.entryflagy.toggled.connect(lambda:self.entryflagfunc(self.entryflagy))
        self.entryflagn = QRadioButton("No")
        if(param_dict['clear_multiple_entry_flag'] == 0):
            self.entryflagn.setChecked(True)
        self.entryflagn.toggled.connect(lambda:self.entryflagfunc(self.entryflagn))

        self.tabs.addWidget(self.entryflag,25,0)
        self.tabs.addWidget(self.entryflagy,25,1)
        self.tabs.addWidget(self.entryflagn,25,2)
        self.btngroup8 = QButtonGroup()
        self.btngroup8.addButton(self.entryflagy)
        self.btngroup8.addButton(self.entryflagn)

        self.setLayout(self.tabs)


    def cardstate(self,b):
        with open("Settings/config.json", "r") as f:
            param_dict = json.load(f)
        if(b.text() == "Decklink 4K Extreme 12G"):
            if b.isChecked() == True:
                param_dict['decklink_12g_mode'] = 1
                    
        if(b.text() == "Decklink 8K Pro"):
            if b.isChecked() == True:
                param_dict['decklink_12g_mode'] = 0

        with open("Settings/config.json", "w") as f:
            json.dump(param_dict, f, indent=4)
                    

    def portstate(self,b):
        with open("Settings/config.json", "r") as f:
            param_dict = json.load(f)
        if(b.text() == "Port 1"):
            if b.isChecked() == True:
                param_dict['decklink_port_no'] = 0
                
        if(b.text() == "Port 2"):
            if b.isChecked() == True:
                param_dict['decklink_port_no'] = 1

        if(b.text() == "Port 3"):
            if b.isChecked() == True:
                param_dict['decklink_port_no'] = 2

        if(b.text() == "Port 4"):
            if b.isChecked() == True:
                param_dict['decklink_port_no'] = 3
            with open("Settings/config.json", "w") as f:
                json.dump(param_dict, f, indent=4)
        
    def stadium_save_clicked(self):
        with open("Settings/config.json", "r") as f:
            param_dict = json.load(f)
        param_dict["stadium"] = str(self.stadiumedit.text())
        with open("Settings/config.json", "w") as f:
            json.dump(param_dict, f, indent=4)
        self.set_style(True, False, False, False, False,False,False,False,False,False,False)


    
    def vizipf_save_clicked(self):
        with open("Settings/config.json", "r") as f:
            param_dict = json.load(f)
        param_dict["viz_udp_ip_address"] = str(self.vizipf_input.text())
        param_dict["viz_tcp_ip_address"] = str(self.vizipf_input.text())
        with open("Settings/config.json", "w") as f:
            json.dump(param_dict, f, indent=4)
        self.set_style(False, True, False, False, False,False,False,False,False,False,False)

    def vizudp_save_clicked(self):
        with open("Settings/config.json", "r") as f:
            param_dict = json.load(f)
        param_dict["viz_udp_port"] = str(self.vizudpp_input.text())
        with open("Settings/config.json", "w") as f:
            json.dump(param_dict, f, indent=4)
        self.set_style(False, False, True, False, False,False,False,False,False,False,False)

    def viztcp_save_clicked(self):
        with open("Settings/config.json", "r") as f:
            param_dict = json.load(f)
        param_dict["viz_tcp_port"] = str(self.viztcpp_input.text())
        with open("Settings/config.json", "w") as f:
            json.dump(param_dict, f, indent=4)
        self.set_style(False, False, False, True, False,False,False,False,False,False,False)

    def downudp_save_clicked(self):
        with open("Settings/config.json", "r") as f:
            param_dict = json.load(f)
        param_dict["middleman_ip_address_port"] = "tcp://" + str(self.downudp_input.text()) + ":2000"
        with open("Settings/config.json", "w") as f:
            json.dump(param_dict, f, indent=4)
        self.set_style(False, False, False, False, True,False,False,False,False,False,False)

    def history_save_clicked(self):
        with open("Settings/config.json", "r") as f:
            param_dict = json.load(f)
        param_dict["viz_udp_port"] = "tcp://" + str(self.vizudpp_input.text())
        with open("Settings/config.json", "w") as f:
            json.dump(param_dict, f, indent=4)
        

    def getDirectory(self):
        response = QFileDialog.getExistingDirectory(
            self,
            caption='Select a folder'
        )
        print(response)
        return response 

    def getFileName(self):
        file_filter = 'Video Files (*.mp4 *.mov *.MOV *.avi)'
        # path = self.getDirectory()
        # start = "/qt-deployment/"
        response = QFileDialog.getOpenFileName(
            parent=self,
            caption='Select a data file',
            directory= os.getcwd(),
            filter=file_filter,
            initialFilter='Video Files (*.mp4 *.mov *.MOV *.avi)'
        )
        self.savevideofile(response[0])
        return response[0]


    def remotepcpathbtn(self):
        with open("Settings/config.json", "r") as f:
            param_dict = json.load(f)
        path = self.getDirectory()
        
        remotepath = str(path) + "/"
        param_dict["remote_pc_score_path"] = str(remotepath)
        with open("Settings/config.json", "w") as f:
            json.dump(param_dict, f, indent=4)
        self.remotepcpath.setText(str(remotepath))
        
            
    def scorefilemode(self):
        with open("Settings/config.json", "r") as f:
            param_dict = json.load(f)
        param_dict["score_file_mode"] = str(self.scorefilemodetype.text())
        with open("Settings/config.json", "w") as f:
            json.dump(param_dict, f, indent=4)
        self.set_style(False, False, False, False,False, True,False,False,False,False,False)

    def wtscorename(self):
        with open("Settings/config.json", "r") as f:
            param_dict = json.load(f)
        param_dict["score_file_path"] = "../score_file/" + str(self.wtscorefilename.text()) 
        param_dict["wt_score_file"] = str(self.wtscorefilename.text())
        with open("Settings/config.json", "w") as f:
            json.dump(param_dict, f, indent=4)
        self.set_style(False, False, False, False,False,False,True, False,False,False,False)
    
    def folder_list_with_extension(self,img_folder_path="Settings/detection_weights/", img_extension =".pt"):
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

    def weights_change(self):
        i = self.weights.currentIndex()
        weight_name = self.weight_list[i-1]
        with open("../Settings/config.json", 'r') as f:
            config_dict = json.load(f)
        config_dict['detection_weight'] = "Settings/detection_weights/"+ weight_name
        with open("../Settings/config.json", 'w') as f:
            json.dump(config_dict, f, indent=4)
        self.set_style(False, False, False, False,False,False, False,False,True,False,False)

    def dbnamefunc(self):
        with open("Settings/config.json", "r") as f:
            param_dict = json.load(f)
        param_dict["db_name"] = str(self.dbnameedit.text())
        with open("Settings/config.json", "w") as f:
            json.dump(param_dict, f, indent=4)
        self.set_style(False, False, False, False,False,False, False,False,True,False,False)
    
    def mforpixfunc(self):
        with open("Settings/config.json", "r") as f:
            param_dict = json.load(f)
        param_dict["m_for_pix"] = str(self.mforpixedit.text())
        with open("Settings/config.json", "w") as f:
            json.dump(param_dict, f, indent=4)
        self.set_style(False, False, False, False,False,False, False,False,False,True,False)

    def inningsfunc(self):
        with open("Settings/config.json", "r") as f:
            param_dict = json.load(f)
        param_dict["innings"] = str(self.inningsedit.text())
        with open("Settings/config.json", "w") as f:
            json.dump(param_dict, f, indent=4)
        self.set_style(False, False, False, False,False,False, False,True,False,False,False)

    def historydbnamefunc(self):
        with open("Settings/config.json", "r") as f:
            param_dict = json.load(f)
        param_dict["history_data_path"] = str(self.historydbnameedit.text()) + ".json"
        with open("Settings/config.json", "w") as f:
            json.dump(param_dict, f, indent=4)
        self.set_style(False, False, False, False,False,False, False,False,False,False,True)

    def cropactivated(self,b):
        with open("Settings/config.json", "r") as f:
            param_dict = json.load(f)
        if(b.text() == "Activated"):
            if b.isChecked() == True:
                param_dict["activate_crop"] = 1           
        elif(b.text() == "Deactivated"):
            if b.isChecked() == True:
                param_dict["activate_crop"] = 0
        with open("Settings/config.json", "w") as f:
            json.dump(param_dict, f, indent=4)
    
    def lensdistortionfunc(self,b):
        with open("Settings/config.json", "r") as f:
            param_dict = json.load(f)
        if(b.text() == "Activated"):
            if b.isChecked() == True:
                param_dict["lens_distortion"] = 1           
        elif(b.text() == "Deactivated"):
            if b.isChecked() == True:
                param_dict["lens_distortion"] = 0
        with open("Settings/config.json", "w") as f:
            json.dump(param_dict, f, indent=4)
    
    def camerasourcefunc(self,b):
        with open("Settings/config.json", "r") as f:
            param_dict = json.load(f)
        if(b.text() == "Input from Decklink(Camera/Shogun Feed)"):
            if b.isChecked() == True:
                param_dict["camera_model"] = 2          
        elif(b.text() == "Local Video"):
            if b.isChecked() == True:
                param_dict["lens_distortion"] = 1
        with open("Settings/config.json", "w") as f:
            json.dump(param_dict, f, indent=4)

    def collisonmodefunc(self,b):
        with open("Settings/config.json", "r") as f:
            param_dict = json.load(f)
        if(b.text() == "Activated"):
            if b.isChecked() == True:
                param_dict["collision_mode"] = 1           
        elif(b.text() == "Deactivated"):
            if b.isChecked() == True:
                param_dict["collision_mode"] = 0
        with open("Settings/config.json", "w") as f:
            json.dump(param_dict, f, indent=4)

    def lensdistortionfunc(self,b):
        with open("Settings/config.json", "r") as f:
            param_dict = json.load(f)
        if(b.text() == "Activated"):
            if b.isChecked() == True:
                param_dict["lens_distortion"] = 1           
        elif(b.text() == "Deactivated"):
            if b.isChecked() == True:
                param_dict["lens_distortion"] = 0
        with open("Settings/config.json", "w") as f:
            json.dump(param_dict, f, indent=4)   
    
    def udpcommandgunc(self,b):
        with open("Settings/config.json", "r") as f:
            param_dict = json.load(f)
        if(b.text() == "Yes"):
            if b.isChecked() == True:
                param_dict["print_udp_command"] = 1           
        elif(b.text() == "No"):
            if b.isChecked() == True:
                param_dict["print_udp_command"] = 0
        with open("Settings/config.json", "w") as f:
            json.dump(param_dict, f, indent=4)
    
    def entryflagfunc(self,b):
        with open("Settings/config.json", "r") as f:
            param_dict = json.load(f)
        if(b.text() == "Yes"):
            if b.isChecked() == True:
                param_dict["clear_multiple_entry_flag"] = 1           
        elif(b.text() == "No"):
            if b.isChecked() == True:
                param_dict["clear_multiple_entry_flag"] = 0
        with open("Settings/config.json", "w") as f:
            json.dump(param_dict, f, indent=4)
    
    def video_list_with_extension(self):
        def open(self):
            path = QFileDialog.getOpenFileName(self, 'Open a file', '',
                                            'All Files (*.*)')
            if path != ('', ''):
                print(path[0])
    
    def savevideofile(self,path):
        with open("Settings/config.json", "r") as f:
            param_dict = json.load(f)
        param_dict['video_source'] = path
        with open("Settings/config.json", "w") as f:
            json.dump(param_dict, f, indent=4)
    
    def set_style(self, stadiumsave_clicked, vizipf_clicked, vizudpp_clicked, viztcpp_clicked, downudp_clicked,scorefilesave_clicked,wtscorefilesave_clicked,inningssave_clicked,dbnamesave_clicked,mforpixsave_clicked,historydbnamesave_clicked):
        self.stadiumsave.setStyleSheet("background-color: green;color:black") if stadiumsave_clicked else self.stadiumsave.setStyleSheet("background-color: white;color:black") 
        self.vizipf_save.setStyleSheet("background-color: green;color:black") if vizipf_clicked else self.vizipf_save.setStyleSheet("background-color: white;color:black") 
        self.vizudpp_save.setStyleSheet("background-color: green;color:black") if vizudpp_clicked else self.vizudpp_save.setStyleSheet("background-color: white;color:black") 
        self.viztcpp_save.setStyleSheet("background-color: green;color:black") if viztcpp_clicked else self.viztcpp_save.setStyleSheet("background-color: white;color:black") 
        self.downudp_save.setStyleSheet("background-color: green;color:black") if downudp_clicked else self.downudp_save.setStyleSheet("background-color: white;color:black") 
        self.scorefilesave.setStyleSheet("background-color: green;color:black") if scorefilesave_clicked else self.scorefilesave.setStyleSheet("background-color: white;color:black") 
        self.wtscorefilesave.setStyleSheet("background-color: green;color:black") if wtscorefilesave_clicked else self.wtscorefilesave.setStyleSheet("background-color: white;color:black") 
        self.inningssave.setStyleSheet("background-color: green;color:black") if inningssave_clicked else self.inningssave.setStyleSheet("background-color: white;color:black") 
        self.dbnamesave.setStyleSheet("background-color: green;color:black") if dbnamesave_clicked else self.dbnamesave.setStyleSheet("background-color: white;color:black") 
        self.mforpixsave.setStyleSheet("background-color: green;color:black") if mforpixsave_clicked else self.mforpixsave.setStyleSheet("background-color: white;color:black") 
        self.historydbnamesave.setStyleSheet("background-color: green;color:black") if historydbnamesave_clicked else self.historydbnamesave.setStyleSheet("background-color: white;color:black") 


        
    

    # def weights_change(self):
    #     i = self.weights.currentIndex()
    #     weight_name = self.weight_list[i-1]
    #     with open("../Settings/config.json", 'r') as f:
    #         config_dict = json.load(f)
    #     config_dict['detection_weight'] = "Settings/detection_weights/"+ weight_name
    #     with open("../Settings/config.json", 'w') as f:
    #         json.dump(config_dict, f, indent=4)
   
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec_())