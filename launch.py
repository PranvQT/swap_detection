import json
import socket
from traceback import print_tb
import os
import copy
import cv2

TCP_IP = '192.168.10.5'
TCP_PORT = 6100

path = "/run/user/1000/gvfs/smb-share:server=192.168.8.10,share=data/"
# path = "./"
past_time = 0
current_id = 0

# frame = cv2.imread("dst.jpg")
# print(frame.shape)
width = 3012
height = 3022
# serverSock.connect((TCP_IP, TCP_PORT))

while True:
    try:
        modified_time = os.path.getmtime(path+"launch_db.json")
        if modified_time != past_time:
            past_time = copy.deepcopy(modified_time)
            # print(os.path.getmtime(path+"launch_db.json"))
            serverSock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            side_string = "0 RENDERER*TREE*@SWITCH_SIDES*SCRIPT*INSTANCE*iWhichSide SET "
            over_string = "0 RENDERER*TREE*@TRANSFORM_COUNTERS*SCRIPT*INSTANCE*m_strHeaderData SET "
            player_string = "0 RENDERER*TREE*@TRANSFORM_COUNTERS*SCRIPT*INSTANCE*m_strPlayerData SET "
            with open(path+"launch_db.json", "r") as _file:
                _data = json.load(_file)
            _launch_data = _data.get("launch")
            if len(_launch_data) > 0:
                # print(current_id)
                if current_id != _launch_data[-1]["id"]:
                    current_id = _launch_data[-1]["id"]
                    marked_players = _launch_data[-1]["player"]
                    unmarked_players = _launch_data[-1]["unselected_players"]
                    # print(marked_players)
                    # print(unmarked_players)
                    counter = 1
                    for i in range(len(marked_players)):
                        # print(i)
                        # print(marked_players[i])
                        x = marked_players[i]["homo_track"][0]
                        y = marked_players[i]["homo_track"][1]

                        if counter % 2 == 0:
                            if _launch_data[-1]["selecteddata2"]["selecteddata2_flip"] == 1:
                                x = width - x - 1
                                y = height - y - 1

                            player_string += str(x) + "_" + str(
                                y*-1)+"_0#"

                            # cv2.circle(frame, (int(x), int(
                            #     y)), 5, (0, 255, 255), -1)
                            # cv2.putText(frame, str(marked_players[i]["identities"]), (int(y), int(
                            #     y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
                        else:
                            if _launch_data[-1]["selecteddata1"]["selecteddata1_flip"] == 1:
                                x = width - x - 1
                                y = height - y - 1
                            player_string += str(marked_players[i]["identities"]) + "@" + str(
                                x) + "_" + str(y*-1)+"_0_"

                            # cv2.circle(frame, (int(x), int(
                            #     y)), 5, (0, 0, 255), -1)
                            # cv2.putText(frame, str(marked_players[i]["identities"]), (int(y), int(
                            #     y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                        counter += 1

                    for _player in unmarked_players:
                        x = _player["homo_track"][0]
                        y = _player["homo_track"][1]

                        if _launch_data[-1]["selecteddata1"]["selecteddata1_flip"] == 1:
                            x = width - x - 1
                            y = height - y - 1

                        # cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)
                        # cv2.putText(frame, str(_player["identities"]),(int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

                        player_string += str(_player["identities"]) + "@" + str(
                            x) + "_" + str(y*-1)+"_0#"

                    player_string = player_string[:-1]+"\0"

                    over_string += str(_launch_data[-1]["selecteddata1"]["selecteddata1_over"])+"."+str(_launch_data[-1]["selecteddata1"]["selecteddata1_ball"]) + " Overs_" + str(
                        _launch_data[-1]["selecteddata2"]["selecteddata2_over"])+"."+str(_launch_data[-1]["selecteddata2"]["selecteddata2_ball"]) + " Overs"+"\0"
                    if str(_launch_data[-1]["selecteddata1"]["selecteddata1_Bat_type"]) == "R":
                        side_string += str(1) + "\0"
                    elif str(_launch_data[-1]["selecteddata1"]["selecteddata1_Bat_type"]) == "L":
                        side_string += str(0) + "\0"
                    print(over_string)
                    print(side_string)
                    print(player_string)
                    # cv2.imwrite("test1.jpg", frame)

                    serverSock.connect((TCP_IP, TCP_PORT))

                    serverSock.send(over_string.encode('utf-8'))
                    serverSock.send(side_string.encode('utf-8'))
                    serverSock.send(player_string.encode('utf-8'))
                    serverSock.close()

                    # serverSock.sendto(over_string.encode('utf-8'), (TCP_IP, TCP_PORT))
                    # serverSock.sendto(player_string.encode('utf-8'), (TCP_IP, TCP_PORT))
    except Exception as e:
        print("here",e)
import json
import socket
from traceback import print_tb
import os
import copy
import cv2

TCP_IP = '192.168.10.5'
TCP_PORT = 6100

path = "/run/user/1000/gvfs/smb-share:server=192.168.8.10,share=data/"
# path = "./"
past_time = 0
current_id = 0

# frame = cv2.imread("dst.jpg")
# print(frame.shape)
width = 3012
height = 3022
# serverSock.connect((TCP_IP, TCP_PORT))

while True:
    try:
        modified_time = os.path.getmtime(path+"launch_db.json")
        if modified_time != past_time:
            past_time = copy.deepcopy(modified_time)
            # print(os.path.getmtime(path+"launch_db.json"))
            serverSock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            side_string = "0 RENDERER*TREE*@SWITCH_SIDES*SCRIPT*INSTANCE*iWhichSide SET "
            over_string = "0 RENDERER*TREE*@TRANSFORM_COUNTERS*SCRIPT*INSTANCE*m_strHeaderData SET "
            player_string = "0 RENDERER*TREE*@TRANSFORM_COUNTERS*SCRIPT*INSTANCE*m_strPlayerData SET "
            with open(path+"launch_db.json", "r") as _file:
                _data = json.load(_file)
            _launch_data = _data.get("launch")
            if len(_launch_data) > 0:
                # print(current_id)
                if current_id != _launch_data[-1]["id"]:
                    current_id = _launch_data[-1]["id"]
                    marked_players = _launch_data[-1]["player"]
                    unmarked_players = _launch_data[-1]["unselected_players"]
                    # print(marked_players)
                    # print(unmarked_players)
                    counter = 1
                    for i in range(len(marked_players)):
                        # print(i)
                        # print(marked_players[i])
                        x = marked_players[i]["homo_track"][0]
                        y = marked_players[i]["homo_track"][1]

                        if counter % 2 == 0:
                            if _launch_data[-1]["selecteddata2"]["selecteddata2_flip"] == 1:
                                x = width - x - 1
                                y = height - y - 1

                            player_string += str(x) + "_" + str(
                                y*-1)+"_0#"

                            # cv2.circle(frame, (int(x), int(
                            #     y)), 5, (0, 255, 255), -1)
                            # cv2.putText(frame, str(marked_players[i]["identities"]), (int(y), int(
                            #     y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
                        else:
                            if _launch_data[-1]["selecteddata1"]["selecteddata1_flip"] == 1:
                                x = width - x - 1
                                y = height - y - 1
                            player_string += str(marked_players[i]["identities"]) + "@" + str(
                                x) + "_" + str(y*-1)+"_0_"

                            # cv2.circle(frame, (int(x), int(
                            #     y)), 5, (0, 0, 255), -1)
                            # cv2.putText(frame, str(marked_players[i]["identities"]), (int(y), int(
                            #     y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                        counter += 1

                    for _player in unmarked_players:
                        x = _player["homo_track"][0]
                        y = _player["homo_track"][1]

                        if _launch_data[-1]["selecteddata1"]["selecteddata1_flip"] == 1:
                            x = width - x - 1
                            y = height - y - 1

                        # cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)
                        # cv2.putText(frame, str(_player["identities"]),(int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

                        player_string += str(_player["identities"]) + "@" + str(
                            x) + "_" + str(y*-1)+"_0#"

                    player_string = player_string[:-1]+"\0"

                    over_string += str(_launch_data[-1]["selecteddata1"]["selecteddata1_over"])+"."+str(_launch_data[-1]["selecteddata1"]["selecteddata1_ball"]) + " Overs_" + str(
                        _launch_data[-1]["selecteddata2"]["selecteddata2_over"])+"."+str(_launch_data[-1]["selecteddata2"]["selecteddata2_ball"]) + " Overs"+"\0"
                    if str(_launch_data[-1]["selecteddata1"]["selecteddata1_Bat_type"]) == "R":
                        side_string += str(1) + "\0"
                    elif str(_launch_data[-1]["selecteddata1"]["selecteddata1_Bat_type"]) == "L":
                        side_string += str(0) + "\0"
                    print(over_string)
                    print(side_string)
                    print(player_string)
                    # cv2.imwrite("test1.jpg", frame)

                    serverSock.connect((TCP_IP, TCP_PORT))

                    serverSock.send(over_string.encode('utf-8'))
                    serverSock.send(side_string.encode('utf-8'))
                    serverSock.send(player_string.encode('utf-8'))
                    serverSock.close()

                    # serverSock.sendto(over_string.encode('utf-8'), (TCP_IP, TCP_PORT))
                    # serverSock.sendto(player_string.encode('utf-8'), (TCP_IP, TCP_PORT))
    except Exception as e:
        print("here",e)

