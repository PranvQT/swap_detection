import numpy as np
import cv2 as cv
import json
import math
import copy

crease_to_crease_distance = 17.68#1768cmss


drawing = False # true if mouse is pressed
ellipse_draw = False

src_x, src_y = None,None
dst_x, dst_y = None, None 
i_s = 0
i_d = 0

el_list_src = []
el_list_dst = []
line_point = []
line_points = []
src_line_points = []
dst_line_points =[]
src_list = []
dst_list = []
select_points_dstn= False
select_points_srcn= False

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

# mouse callback function
def select_points_dst(event,x,y,flags,param):
    global dst_x, dst_y, drawing ,dst_copy ,new_dst,select_points_dstn
    new_dst = cv.line(dst_copy.copy(), (x-5,y),(x+5,y) , (0,0,255), 1)
    new_dst = cv.line(new_dst, (x,y-5),(x,y+5) , (0,0,255), 1)
    cv.imshow("dst", new_dst)
    if event == cv.EVENT_MBUTTONDOWN:
        drawing = True
        select_points_dstn = True
        print("selecting_dst")
        dst_x, dst_y = x,y
        cv.circle(dst_copy,(x,y),1,(0,0,255),-1)
    elif event == cv.EVENT_MBUTTONUP:
        drawing = False

def midpoint(p):
    p1, p2 = p
    return [(p1[0]+p2[0])/2,(p1[1]+p2[1])/2]

def line_generator(src_x,src_y,src_copy ,src_d, u_src):
    global line_points , line_point
    if src_x and src_y != None and len(line_point) <=4:
        line_point.append([src_x,src_y])
        cv.circle(src_copy,(src_x,src_y),1,(255,0,0),-1)
        src_x, src_y= None, None
        # print(line_point)
    else:
        print("Points are NONE or already selected 4 points")
    if len(line_point)==2:
        line_point_array = np.asarray(line_point)
        line_points.append(line_point.copy())
        # print(line_points)
        line = cv.fitLine(line_point_array,cv.DIST_L2, 0, 0.1, 0.1)
        vx, vy, cx, cy = line
        cv.line(src_copy, (int(cx-vx*src_d), int(cy-vy*src_d)), (int(cx+vx*src_d), int(cy+vy*src_d)), (255, 255, 255),1)
        # u_src.append(copy.deepcopy(src_copy))
        line_point.clear()
        if len(line_points)==2:
            line3_point = [line_points[0][0],line_points[1][0]]
            line3_point_array = np.asarray(line3_point)
            # print(line3_point_array)
            line3 = cv.fitLine(line3_point_array,cv.DIST_L2, 0, 0.1, 0.1)
            vx1, vy1, cx1, cy1 = line3
            cv.line(src_copy, (int(cx1-vx1*src_d), int(cy1-vy1*src_d)), (int(cx1+vx1*src_d), int(cy1+vy1*src_d)), (255, 255, 255),1)

            line4_point = [line_points[0][1],line_points[1][1]]
            line4_point_array = np.asarray(line4_point)
            line4 = cv.fitLine(line4_point_array,cv.DIST_L2, 0, 0.1, 0.1)
            vx, vy, cx, cy = line4
            cv.line(src_copy, (int(cx-vx*src_d), int(cy-vy*src_d)), (int(cx+vx*src_d), int(cy+vy*src_d)), (255, 255, 255),1)


            line5_point = [line_points[0][0],line_points[1][1]]
            line5_point_array = np.asarray(line5_point)
            line5 = cv.fitLine(line5_point_array,cv.DIST_L2, 0, 0.1, 0.1)
            vx2, vy2, cx2, cy2 = line5
            cv.line(src_copy, (int(cx2-vx2*src_d), int(cy2-vy2*src_d)), (int(cx2+vx2*src_d), int(cy2+vy2*src_d)), (255, 255, 255),1)
            
            line6_point = [line_points[0][1],line_points[1][0]]
            line6_point_array = np.asarray(line6_point)
            line6 = cv.fitLine(line6_point_array,cv.DIST_L2, 0, 0.1, 0.1)
            vx3, vy3, cx3, cy3 = line6
            cv.line(src_copy, (int(cx3-vx3*src_d), int(cy3-vy3*src_d)), (int(cx3+vx3*src_d), int(cy3+vy3*src_d)), (255, 255, 255),1)


            
    return line_points

def get_plan_view(src, dst):
    src_pts = np.array(src_list).reshape(-1,1,2)
    dst_pts = np.array(dst_list).reshape(-1,1,2)
    H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,35.0) #(50,100,)
    # H, mask = cv.findHomography(src_pts, dst_pts)

    plan_view = cv.warpPerspective(src, H, (dst.shape[1], dst.shape[0]))
    return plan_view ,H


def dsit_to_pix(list_src):
    # pixel_distance = math.dist((list_src[0][0],list_src[0][1]),(list_src[1][0],list_src[1][1]))
    x1 = list_src[0][0]
    y1 = list_src[0][1]
    x2 = list_src[1][0]
    y2 = list_src[1][1]
    pixel_distance = math.hypot(x2 - x1, y2 - y1)
    print(pixel_distance)
    val = crease_to_crease_distance/pixel_distance
    print("1 pixel = {0} Meters".format(val))
    return val

def calculate_homography(obj):
    # homo_start_time = time.time()
    x_current,y_current =obj
    points_map = np.array([[x_current,y_current]],dtype='float32')
    points_map = np.array([points_map])
    with open("../Settings/cam_params.json", 'r') as json_file:
        camera_params = json.load(json_file)
        cam_matrix = np.array(camera_params['CAM']) 
    tranformed_points = cv.perspectiveTransform(points_map,cam_matrix)
    # homo_end_time = time.time() - homo_start_time
    # ##print("homography time: ", homo_end_time)
    return tranformed_points
    


src = cv.imread('../Settings/src.jpg', -1)
src_copy = src.copy()
new_src = src_copy
cv.namedWindow("src",cv.WINDOW_NORMAL)
cv.moveWindow("src", 80,80);
cv.setMouseCallback('src', select_points_src)
src_w,src_h ,src_c= src.shape
src_d = np.sqrt(np.square(src_w)+np.square(src_h))
dst = cv.imread('../Settings/dst.jpg', -1)
dst_copy = dst.copy()
new_dst = dst_copy
cv.namedWindow("dst",cv.WINDOW_NORMAL)
cv.moveWindow("dst", 780,80);
cv.setMouseCallback('dst', select_points_dst)

dst_w,dst_h ,dst_c= src.shape
dst_d = np.sqrt(np.square(dst_w)+np.square(dst_h))
list_dst =[]
u_src = [copy.deepcopy(src_copy)]
u_dst = [copy.deepcopy(dst_copy)]
flag = None
while True:
    cv.imshow('src',src_copy)
    cv.imshow("src", new_src)
    new_src = src_copy
    cv.imshow('dst',dst_copy)
    cv.imshow("dst", new_dst)
    new_dst = dst_copy

    if len(u_src)==10:
        u_src.pop(0)
    if len(u_dst)==10:
        u_dst.pop(0)
    if len(u_dst)==0:
        u_dst = [copy.deepcopy(dst_copy)]

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


    elif k==ord("p"):
        print("selecting point for pixel ratio calculation")
        if dst_x and dst_y != None:
            list_dst.append([dst_x,dst_y])
            cv.circle(dst_copy,(dst_x,dst_y),1,(255,0,0),-1)
            flag = "list_dst"
        if len(list_dst)==2:
            m_for_pix = dsit_to_pix(list_dst)
            m_for_pix = round(m_for_pix,4)
            list_dst.clear()
            with open("../Settings/config.json", 'r') as f:
                config_dict = json.load(f)
            config_dict['m_for_pix'] = m_for_pix
            with open("../Settings/config.json", 'w') as f:
                json.dump(config_dict,f, indent = 4)
            


    elif k == ord("e"):
        i_d +=1
        print("Select the {i}th point for ellipse in source image".format(i=i_d))
        if dst_x and dst_y != None:
            flag = "el_list_dst"
            el_list_dst.append([dst_x,dst_y])
            cv.circle(dst_copy,(dst_x,dst_y),1,(255,0,0),-1)
            u_dst.append(copy.deepcopy(dst_copy))


    elif  k ==ord('t'):
        print("Drawing Ellipses")
        print(el_list_src)
        flag = "draw_el"
        el_array_src = np.asarray(el_list_src)
        ellipse_src = cv.fitEllipse(el_array_src)
        print("ellipse_values src:",ellipse_src)
        #mask = np.zeros_like(src_copy)
        src_copy= cv.ellipse(src_copy,ellipse_src,(0,0,0),2)
        print(el_list_dst)
        el_array_dst = np.asarray(el_list_dst)
        ellipse_dst = cv.fitEllipse(el_array_dst)
        print("ellipse_values dst:",ellipse_dst)
        dst_copy = cv.ellipse(dst_copy,ellipse_dst,(0,0,0),2)
        src_x, src_y= None, None 
        u_src.append(copy.deepcopy(src_copy))
        u_dst.append(copy.deepcopy(dst_copy))
        print("Select the edges of pith from batman's right to left both up and down")
        
    elif k ==ord("a"):
        print("SORCE")
        flag = "line_src"
        # src_line_points = line_generator(src_x,src_y,src_copy ,src_d,u_src).copy()
        if src_x and src_y != None and len(line_point) <=4:
            line_point.append([src_x,src_y])
            cv.circle(src_copy,(src_x,src_y),1,(255,0,0),-1)
            src_x, src_y= None, None
            print(line_point)
        else:
            print("Points are NONE or already selected 4 points")
        if len(line_point)==2:
            line_point_array = np.asarray(line_point)
            line_points.append(line_point.copy())
            line = cv.fitLine(line_point_array,cv.DIST_L2, 0, 0.1, 0.1)
            vx, vy, cx, cy = line
            cv.line(src_copy, (int(cx-vx*src_d), int(cy-vy*src_d)), (int(cx+vx*src_d), int(cy+vy*src_d)), (255, 255, 255),1)
            line_point.clear()
        u_src.append(copy.deepcopy(src_copy))



        
    elif k ==ord("d"):
        print("DEST")
        flag = "line_dst"
        if dst_x and dst_y != None and len(line_point) <=4:
            line_point.append([dst_x,dst_y])
            cv.circle(dst_copy,(dst_x,dst_y),1,(255,0,0),-1)
            dst_x, dst_y= None, None
            print(line_point)
        else:
            print("Points are NONE or already selected 4 points")
        if len(line_point)==2:
            line_point_array = np.asarray(line_point)
            line_points.append(line_point.copy())
            line = cv.fitLine(line_point_array,cv.DIST_L2, 0, 0.1, 0.1)
            vx, vy, cx, cy = line
            cv.line(dst_copy, (int(cx-vx*dst_d), int(cy-vy*dst_d)), (int(cx+vx*dst_d), int(cy+vy*dst_d)), (255, 255, 255),1)
            line_point.clear()
        u_dst.append(copy.deepcopy(dst_copy))


    elif k == ord("z"):
        print("Select 8 or more intersection points in src in clockwise starting from right side of batsman")
        if src_x and src_y != None:
            flag = "src_inter"
            src_list.append([src_x,src_y])
            print(src_list)
            cv.circle(src_copy,(src_x,src_y),1,(255,0,0),-1)
            u_src.append(copy.deepcopy(src_copy))

    elif k == ord("c"):
        print("Select 8 or more intersection points in src in clockwise starting from right side of batsman")
        if dst_x and dst_y != None:
            flag = "dst_inter"
            dst_list.append([dst_x,dst_y])
            cv.circle(dst_copy,(dst_x,dst_y),1,(255,0,0),-1)
            print(dst_list)
            u_dst.append(copy.deepcopy(dst_copy))


    
        
    elif k ==ord("b"):
        print("homography")
        print("dst",dst_list)
        print("src", src_list)
        if len(dst_list)>=8 and len(src_list) >= 8:
            print('Plan view')
            plan_view ,h_matrix = get_plan_view(src, dst)
            cv.imwrite("required1.jpg",plan_view)
            added_image = cv.addWeighted(dst,1,plan_view,0.5,0)
            cv.namedWindow("plan view",cv.WINDOW_NORMAL)
            cv.imwrite("required2.jpg",added_image)
            cv.imshow("plan view", added_image) 

            json_data = {
                            "CAM": [
                                [
                                    h_matrix[0][0],
                                    h_matrix[0][1],
                                    h_matrix[0][2]
                                ],
                                [
                                    h_matrix[1][0],
                                    h_matrix[1][1],
                                    h_matrix[1][2]
                                ],
                                [
                                    h_matrix[2][0],
                                    h_matrix[2][1],
                                    h_matrix[2][2]
                                ]
                            ]
                        }
          
            with open('../Settings/cam_params.json','w') as data :
                json.dump(json_data,data)

        else:
            print("Make sure all points are there")
    


    elif k == ord("u"):
        def undo(flag):
            global src_copy,dst_copy,select_points_srcn,select_points_dstn
            if flag == "el_list_src":
                if len(el_list_src)>=1:
                    # cv.circle(src_copy,(el_list_src[-1][0],el_list_src[-1][1]),1,(255,255,255),-1)
                    if len(el_list_src)!=0:
                        el_list_src.pop()
                    if len(u_src)>1:
                        u_src.pop()
                        src_copy = copy.deepcopy(u_src[-1])
                    # print(el_list_src)    
                flag = None
                select_points_srcn=False
            elif flag == "el_list_dst":
                if len(el_list_dst)>=1:
                    if len(el_list_dst)!=0:
                        el_list_dst.pop()
                    if len(u_dst)>1:
                        u_dst.pop()
                        dst_copy = copy.deepcopy(u_dst[-1])
                flag = None
                select_points_dstn=False

            elif flag == "list_dst":
                if len(list_dst)>=1:
                    cv.circle(src_copy,(list_dst[-1][0],list_dst[-1][1]),1,(255,255,255),-1)
                    if len(u_dst)!=0 and len(list_dst)>=1:
                        list_dst.pop()
                        u_dst.pop()
                        dst_copy = copy.deepcopy(u_dst[-1])
                    select_points_dstn=False

            elif flag == "draw_el":
                if len(u_dst)>1:
                    u_dst.pop()
                    dst_copy = copy.deepcopy(u_dst[-1])
                if len(u_src)>=1:
                    u_src.pop()
                    src_copy = copy.deepcopy(u_src[-1])
            elif flag == "line_src":
                if len(u_src)>1:
                    if len(line_point)!=0:
                        line_point.pop()
                    if len(line_points)!=0:
                        line_points.pop()
                    u_src.pop()
                    src_copy = copy.deepcopy(u_src[-1])
                select_points_srcn=False

            elif flag == "line_dst":
                if len(u_dst)>1:
                    if len(line_point)!=0:
                        line_point.pop()
                    if len(line_points)!=0:
                        line_points.pop()
                    u_dst.pop()
                    dst_copy = copy.deepcopy(u_dst[-1])
                select_points_dstn=False

            elif flag =="src_inter":
                if len(src_list)!=0:
                    src_list.pop()
                if len(u_src)>1:
                    u_src.pop()
                    src_copy = copy.deepcopy(u_src[-1])   
                select_points_srcn=False

            elif flag =="dst_inter":
                if len(dst_list)!=0:
                    dst_list.pop()

                if len(u_dst)>1:
                    print("ho")
                    u_dst.pop()
                    dst_copy = copy.deepcopy(u_dst[-1])
                select_points_dstn=False

            flag = None


        if select_points_dstn == True:
            dst_copy = copy.deepcopy(u_dst[-1])
            select_points_dstn = False

        if select_points_srcn == True:
            src_copy = copy.deepcopy(u_src[-1])
            select_points_srcn = False

                
        undo(flag)
        

        
        # for i in range(len(u_src)):
        #     cv.imwrite(str(i)+".jpg",u_src[i])
        # #undo function
        cv.imshow('src',src_copy)
   
        cv.imshow('dst',dst_copy)


    elif k == 27:
        break

        
