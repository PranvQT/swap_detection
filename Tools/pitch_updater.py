from io import StringIO
import numpy as np
import cv2 as cv
import json
import copy
import socket   
from numba import jit
from socket import socket, AF_INET, SOCK_STREAM

with open("../Settings/config.json","r") as f:
    param_dict = json.load(f)

viz_ip = param_dict["viz_udp_ip_address"]
unreal_ip = param_dict["unreal_ip_address"]
mpix = param_dict["m_for_pix"]
axis_offset = []
stump = param_dict["stump"]
axis_offset = [stump["near_end"],stump["far_end"]]
with open("../Settings/cam_params.json", 'r') as json_file:
    camera_params = json.load(json_file)
    cam_matrix = np.array(camera_params['CAM'])

drawing = False # true if mouse is pressed
ellipse_draw = False

src_x, src_y = None,None
dst_x, dst_y = None, None 
i_s = 0
i_d = 0

el_list_src = []
line_point = []
select_points_srcn= False
inner_circle_points =[]
crease_near = []
crease_far = []
gap_points = []

@jit(forceobj=True)
def calculate_homography(obj, cam_matrix):
    # homo_start_time = time.time()
    x_current,y_current =obj
    points_map = np.array([[x_current,y_current]],dtype='float32')
    points_map = np.array([points_map])
    tranformed_points = cv.perspectiveTransform(points_map,cam_matrix)
    # homo_end_time = time.time() - homo_start_time
    # ##print("homography time: ", homo_end_time)
    return tranformed_points


def crease_point(list_src):
    x1 = list_src[0][0]
    y1 = list_src[0][1]
    x2 = list_src[1][0]
    y2 = list_src[1][1]

    x= int((x1+x2)/2)
    y= int((y1+y2)/2)
    point = [x,y]
    print(point)
    homo_pt = calculate_homography(point,cam_matrix)
    x, y = (int(homo_pt[0][0][0]), int(homo_pt[0][0][1]))
    return [x,y]

def select_points_src(event,x,y,flags,param):
    global src_x, src_y, drawing , src_copy , new_src,select_points_srcn
    new_src = cv.line(src_copy.copy(), (x-15,y),(x+15,y) , (0,0,255), 1)
    new_src = cv.line(new_src, (x,y-15),(x,y+15) , (0,0,255), 1)
    if event == cv.EVENT_MBUTTONDOWN:
        drawing = True
        select_points_srcn = True
        print("selecting_src")
        src_x, src_y = x,y
        cv.circle(src_copy,(x,y),1,(0,0,255),-1)
    elif event == cv.EVENT_MBUTTONUP:
        drawing = False

def send_data_viz(value):
    global viz_ip
    try:
        tcp_socket = socket(AF_INET, SOCK_STREAM)
        tcp_socket.connect((viz_ip, 6100))
        tcp_socket.send(value.encode('utf-8'))
        tcp_socket.close()
    except socket.error as msg:
        print("Caught exception socket.error :", msg)

def send_data_ue4(value):
    global unreal_ip
    try:
        tcp_socket = socket(AF_INET, SOCK_STREAM)
        tcp_socket.connect((unreal_ip, 1503))
        tcp_socket.send(value.encode('utf-8'))
        tcp_socket.close()
    except socket.error as msg:
        print("Caught exception socket.error :", msg)

src = cv.imread('../Settings/src.jpg', -1)
src_copy = copy.deepcopy(src)
new_src = src_copy
cv.namedWindow("src",cv.WINDOW_NORMAL)
cv.moveWindow("src", 80,80)
cv.setMouseCallback('src', select_points_src)
src_w,src_h ,src_c= src.shape

u_src = [copy.deepcopy(src_copy)]

flag = None
while True:
    cv.imshow('src',src_copy)
    cv.imshow("src", new_src)
    new_src = src_copy
   

    if len(u_src)==10:
        u_src.pop(0)

    if len(u_src)==0:
        u_src = [copy.deepcopy(src_copy)]


    k = cv.waitKey(1) & 0xFF
    #select ellipse points 
    if k == ord("q"): 
        i_s += 1
        print("Select the {i}th point for ellipse in source image".format(i=i_s))
        if src_x and src_y != None:
            flag = "el_list_src"
            cv.circle(src_copy,(src_x,src_y),1,(255,0,0),-1)
            el_list_src.append([src_x,src_y])
            u_src.append(copy.deepcopy(src_copy))
            print(el_list_src)
    
    if k == ord("w"): 
        i_s += 1
        print("Select the {i}th point for ellipse in source image".format(i=i_s))
        if src_x and src_y != None:
            flag = "inner_circle"
            cv.circle(src_copy,(src_x,src_y),1,(255,0,0),-1)
            inner_circle_points.append([src_x,src_y])
            u_src.append(copy.deepcopy(src_copy))
            print(inner_circle_points)
        
    elif k ==ord("e"):
        print("SORCE")
        flag = "line_src"
        if src_x and src_y != None and len(line_point) <=4:
            line_point.append([src_x,src_y])
            cv.circle(src_copy,(src_x,src_y),1,(255,0,0),-1)
            src_x, src_y= None, None
            print(line_point)
        else:
            print("Points are NONE or already selected 4 points")
        u_src.append(copy.deepcopy(src_copy))
    
    elif k ==ord("z"):
        """string data boundary"""
        print(line_point, el_list_src)
        string ="-1 RENDERER*TREE*@BOUNDARY*GEOM*pointsData SET "
        ue4_string ="OB@@"
        p0_count = 0
        for point in el_list_src:
            if point is not None and point !=[]:                
                homo_pt = calculate_homography(point,cam_matrix)
                x ,y  = (int(homo_pt[0][0][0]),int(homo_pt[0][0][1]))
                if p0_count == 0:
                    x1 = copy.deepcopy(x)
                    y1 = copy.deepcopy(y)
                string+=str(x)+":"+str(y)+":0_"
                ue4_string+= "X=" + str(round((x - axis_offset[1][0])*mpix*100)) + ",Y=" + str(round((y - axis_offset[1][1])*mpix*100)) +",Z=0" +"$$"
                p0_count+=1
        string+=str(x1)+":"+str(y1)+":0_"
        ue4_string+= "X=" + str(round((x1 - axis_offset[1][0])*mpix*100)) + ",Y=" + str(round((y1 - axis_offset[1][1])*mpix*100)) +",Z=0" 
        
        boundary=copy.deepcopy(el_list_src)
        boundary = boundary[0:-1]
        with open('boundary_coordinates.txt', 'w') as f:
            f.write(str(boundary))
        
        ue4_boundary=copy.deepcopy(ue4_string)
        ue4_boundary = ue4_string[0:-1]
        with open('ue4_OB.txt', 'w') as f:
            f.write(ue4_string)

        send_data_viz(boundary)
        send_data_ue4(ue4_boundary)
        string= ""
        ue4_string= ""

    elif k ==ord("c"):
        """string data pitch"""
        # print(line_point, el_list_src)
        string ="-1 RENDERER*TREE*@PITCH*SCRIPT*INSTANCE*PITCH_DATA SET "
        ue4_string = "PITCH@@"
        for point in line_point:
            if point is not None and point !=[]:
                homo_pt = calculate_homography(point,cam_matrix)
                x ,y  = (int(homo_pt[0][0][0]),int(homo_pt[0][0][1]))
                string+=str(x)+":"+str(y)+":0_"
                ue4_string+= "X=" + str(round((x - axis_offset[1][0])*mpix*100)) + ",Y=" + str(round((y - axis_offset[1][1])*mpix*100)) +",Z=0" +"$$"
                
        pitch=copy.deepcopy(string)
        pitch = pitch[0:-1]

        ue4_pitch=copy.deepcopy(ue4_string)
        ue4_pitch = ue4_pitch[0:-1]
        # print(pitch)
        with open('pitch_updater.txt', 'w') as f:
            f.write(pitch)
        with open('ue4_pitch_updater.txt', 'w') as f:
            f.write(ue4_pitch)

        send_data_viz(pitch)
        send_data_ue4(ue4_pitch)
        string= ""
        ue4_string= ""


    elif k ==ord("x"):
        """string data inner circle"""
        i_data = []
        string ="-1 RENDERER*TREE*@Boundary_INNER*SCRIPT*INSTANCE*pointsDATA SET "
        ue4_string ="IB@@"
        p1_count = 0
        for point in inner_circle_points:
            if point is not None and point !=[]:
                homo_pt = calculate_homography(point,cam_matrix)
                x ,y  = (int(homo_pt[0][0][0]),int(homo_pt[0][0][1]))
                if p1_count == 0:
                    x1 = copy.deepcopy(x)
                    y1 = copy.deepcopy(y)
                i_data.append([x,y])
                string+=str(x)+":"+str(y)+":0_"
                ue4_string+= "X=" + str(round((x - axis_offset[1][0])*mpix*100)) + ",Y=" + str(round((y - axis_offset[1][1])*mpix*100)) +",Z=0" +"$$"
                p1_count+=1
        ue4_string+= "X=" + str(round((x1 - axis_offset[1][0])*mpix*100)) + ",Y=" + str(round((y1 - axis_offset[1][1])*mpix*100)) +",Z=0" 

        inner_circle=copy.deepcopy(string)
        inner_circle = inner_circle[0:-1]
        ue4_inner_circle=copy.deepcopy(ue4_string)
        ue4_inner_circle = ue4_inner_circle[0:-1]

        json_data = {
                        "inner_circle": i_data
                        }
         
        with open('../Settings/inner_circle.json','w') as data :
            json.dump(json_data,data)

        with open('inner_circle_updater.txt', 'w') as f:
            f.write(inner_circle)
        with open('ue4_inner_circle_updater.txt', 'w') as f:
            f.write(ue4_inner_circle)
        
        send_data_viz(inner_circle)
        send_data_ue4(ue4_inner_circle)
        string= ""
        ue4_string=""
    elif k ==ord("n"):
        # src_line_points = line_generator(src_x,src_y,src_copy ,src_d,u_src).copy()
        if src_x and src_y != None:
            crease_near.append([src_x,src_y])
            cv.circle(src_copy,(src_x,src_y),1,(255,0,0),-1)
            flag = "Near_end_crease"
        if len(crease_near)==2:
            print(crease_near)
            point = crease_point(crease_near)
            with open("../Settings/config.json", 'r') as json_file:
                config_dict = json.load(json_file)
            config_dict["crease"]["near_end"] = point
            with open("../Settings/config.json", 'w') as f:
                json.dump(config_dict,f, indent = 4)
            crease_near.clear()
        u_src.append(copy.deepcopy(src_copy))

    elif k ==ord("m"):
        # src_line_points = line_generator(src_x,src_y,src_copy ,src_d,u_src).copy()
        if src_x and src_y != None:
            crease_far.append([src_x,src_y])
            cv.circle(src_copy,(src_x,src_y),1,(255,0,0),-1)
            flag = "Far_end_crease"
        if len(crease_far)==2:
            point = crease_point(crease_far)
            with open("../Settings/config.json", 'r') as json_file:
                config_dict = json.load(json_file)
            config_dict["crease"]["far_end"] = point
            with open("../Settings/config.json", 'w') as f:
                json.dump(config_dict,f, indent = 4)
            crease_far.clear()
        u_src.append(copy.deepcopy(src_copy))
    
    elif k ==ord("s"):
        if src_x and src_y != None:
            gap_points.append([src_x,src_y])
            cv.circle(src_copy,(src_x,src_y),1,(255,0,0),-1)
            flag = "gap"
        if len(gap_points) ==2:
            stump_data = {}
            stump_src = {}
            if gap_points[0] is not None and gap_points[0] != []:
                stump_src["near_end"] = (gap_points[0][0],gap_points[0][1])
                homo_pt = calculate_homography(gap_points[0], cam_matrix)
                x1, y1 = (int(homo_pt[0][0][0]), int(homo_pt[0][0][1]))
                stump_data["near_end"] = [x1, y1]

            if gap_points[1] is not None and gap_points[1] != []:
                stump_src["far_end"] = (gap_points[1][0],gap_points[1][1])
                homo_pt = calculate_homography(gap_points[1], cam_matrix)
                x2, y2 = (int(homo_pt[0][0][0]), int(homo_pt[0][0][1]))
                stump_data["far_end"] = [x2, y2]
            with open("../Settings/config.json", 'r') as f:
                config_dict = json.load(f)
            config_dict['stump'] = stump_data
            config_dict['stump_src'] = stump_src
            with open("../Settings/config.json", 'w') as f:
                json.dump(config_dict, f, indent=4)
            gap_points.clear()
        


    elif k == ord("u"):
        def undo(flag):
            global src_copy,dst_copy,select_points_srcn
            if flag == "el_list_src":
                if len(el_list_src)>=1:
                    if len(el_list_src)!=0:
                        el_list_src.pop()
                    if len(u_src)>1:
                        u_src.pop()
                        src_copy = copy.deepcopy(u_src[-1])
                flag = None
                select_points_srcn=False

            elif flag == "line_src":
                if len(u_src)>1:
                    if len(line_point)!=0:
                        line_point.pop()
                    u_src.pop()
                    src_copy = copy.deepcopy(u_src[-1])
                select_points_srcn=False
            
            elif flag == "Near_end_crease":
                if len(crease_near)>=1:
                    cv.circle(src_copy,(crease_near[-1][0],crease_near[-1][1]),1,(255,255,255),-1)
                    if len(u_src)!=0 and len(crease_near)>=1:
                        crease_near.pop()
                        u_src.pop()
                        src_copy = copy.deepcopy(u_src[-1])
                    select_points_srcn=False
            
            elif flag == "Far_end_crease":
                if len(crease_far)>=1:
                    cv.circle(src_copy,(crease_far[-1][0],crease_far[-1][1]),1,(255,255,255),-1)
                    if len(u_src)!=0 and len(crease_far)>=1:
                        crease_far.pop()
                        u_src.pop()
                        src_copy = copy.deepcopy(u_src[-1])
                    select_points_srcn=False

            elif flag == "gap":
                if len(gap_points)>=1:
                    cv.circle(src_copy,(gap_points[-1][0],gap_points[-1][1]),1,(255,255,255),-1)
                    if len(u_src)!=0 and len(crease_far)>=1:
                        gap_points.pop()
                        u_src.pop()
                        src_copy = copy.deepcopy(u_src[-1])
                    select_points_srcn=False
            flag = None


        if select_points_srcn == True:
            src_copy = copy.deepcopy(u_src[-1])
            select_points_srcn = False

                
        undo(flag)
        
        cv.imshow('src',src_copy)
   

    elif k == 27:
        break

        
