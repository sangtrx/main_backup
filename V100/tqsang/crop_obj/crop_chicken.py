import os
import cv2
# import json
import time
# import torch
import numpy as np

from tqdm import tqdm


import math
import numpy 
if __name__ == '__main__':
    vid_dir_out = '/data/tqsang/smart_plant/videos_out/'
    vid_dir = '/data/tqsang/smart_plant/videos/'
    frame_dir = '/data/tqsang/smart_plant/frames/'


    vid_names = sorted(os.listdir(vid_dir))

    for vid_name in tqdm(vid_names):
        start = time.time()
        vid_name_path = os.path.join(vid_dir, vid_name)
        frame_folder = os.path.join(frame_dir, os.path.splitext(vid_name)[0])
        try:
            os.makedirs(frame_folder)
        except:
            pass


        print("Video name: ", vid_name)
        vidcap = cv2.VideoCapture(vid_name_path)

        frame_width = int(vidcap.get(3))
        frame_height = int(vidcap.get(4))
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
        out = cv2.VideoWriter(os.path.join(vid_dir_out,'out_video_count.avi'), fourcc, fps, (frame_width,frame_height))
        success,image = vidcap.read()

        middle_frames = []
        n_frames = 0

        current_chick = False
        pre_chick = False
        chick_count = 0

        while success:
            
            success,image = vidcap.read()
            n_frames += 1
            if (n_frames < 2900 + 50) or (n_frames > 7679): #2044:# 1920:
                continue
            # if n_frames > 3000: #2044:# 1920:
            #     break           

            print(n_frames)
            middle_frame = image # (y = 2160, x = 3840, 3)

            ## crop màn đen 
            y = 0
            h = 2160
            x = 940 ## 890 + 50  
            w = 1400
            middle_frame = middle_frame[y:y+h, x:x+w]

            # ##################### crop 
            imgray = cv2.cvtColor(middle_frame,cv2.COLOR_BGR2GRAY)
            # # imgray = cv2.blur(imgray,(15,15))
            
            #### tach lam 2
            imgray_up = imgray[ :2160//2 , : ]
            imgray_down = imgray[ 2160//2 : , : ]
            ret,thresh_down = cv2.threshold(imgray_down,44,255,cv2.THRESH_BINARY)
            ret,thresh_up = cv2.threshold(imgray_up, 44 + 30,255,cv2.THRESH_BINARY)
            thresh = np.concatenate((thresh_up, thresh_down), axis=0)


            # ret,thresh = cv2.threshold(imgray,math.floor(numpy.average(imgray)),255,cv2.THRESH_BINARY)
            # ret,thresh = cv2.threshold(imgray,math.floor(numpy.average(imgray)) + 20,255,cv2.THRESH_BINARY)
            # if n_frames == 124:
            #     print(math.floor(numpy.average(imgray)))
            # cv2.imwrite(os.path.join(frame_folder, '%d_imgray.jpg' % n_frames), thresh)

            dilated=cv2.morphologyEx(thresh, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10)))
            _,contours,_ = cv2.findContours(dilated,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

            new_contours=[]
            # max_con = 0
            for c in contours:
                # print(cv2.contourArea(c))
                # max_con = max(max_con,cv2.contourArea(c))
                # if n_frames == 706:
                #     print(max_con)
                if 1000000>cv2.contourArea(c)>300000:
                    new_contours.append(c)

            best_box=[-1,-1,-1,-1]
            for c in new_contours:
                x,y,w,h = cv2.boundingRect(c)
                # start_point = (x,y+h)
                # end_point = (x+w,y)
                # color = (0, 0, 255)
                # thickness = 3
                # middle_frame = cv2.rectangle(middle_frame, start_point, end_point, color, thickness)
                if best_box[0] < 0:
                    best_box=[x,y,x+w,y+h]
                else:
                    if x<best_box[0]:
                        best_box[0]=x
                    if y<best_box[1]:
                        best_box[1]=y
                    if x+w>best_box[2]:
                        best_box[2]=x+w
                    if y+h>best_box[3]:
                        best_box[3]=y+h   


            

            #### mask big contour
            mask = np.zeros(image.shape, np.uint8)
            cv2.drawContours(mask, new_contours, -1, color=(255, 255, 255), thickness=cv2.FILLED)
            M = np.float32([[1, 0, 940],[0, 1, 0]])
            mask = cv2.warpAffine(mask, M, (mask.shape[1], mask.shape[0]))
            # cv2.imwrite(os.path.join(frame_folder, '%d_mask.jpg' % n_frames), mask)

            # color_mask = image
            # alpha = 0.1
            # color_mask  = (1-alpha)*image + (~mask)*[0,255,0]*alpha

            # a =  ~mask*[0,255,0]

            ### gà only
            # a =  ~mask*[255,255,255]
            # color_mask = cv2.addWeighted(image, 1, a, 1, 0, dtype = cv2.CV_8U)

            ### gà blue
            a =  mask*[255,255,0]
            color_mask = cv2.addWeighted(image, 1-0.0019, a, 0.0019, 0, dtype = cv2.CV_8U)

            # cv2.imwrite(os.path.join(frame_folder, '%d_color_mask.jpg' % n_frames), color_mask)
            image = color_mask
            # start_point = (best_box[0],best_box[3])
            # end_point = (best_box[2],best_box[1])
            # color = (255, 255, 0)
            # thickness = 3
            # middle_frame = cv2.rectangle(middle_frame, start_point, end_point, color, thickness)
            # cv2.imwrite(os.path.join(frame_folder, '%d.jpg' % n_frames), middle_frame)
            #### middle_frame = middle_frame[best_box[1]:best_box[3], best_box[0]:best_box[2]]

            ### vẽ khung
            start_point = (940 , 2160)
            end_point = (940 + 1400 , 0)
            color = (255, 255, 0)
            thickness = 8
            image = cv2.rectangle(image, start_point, end_point, color, thickness)

            #### check if khung con gà overlaps khung lớn or not 
            if (940 + best_box[0] > 940) and (940 + best_box[2] < 940 + 1400):
                if (current_chick == False) and (pre_chick == False):
                    current_chick = True
                    chick_count = chick_count + 1
                # if current_chick == False:
                #     current_chick = True
                #     chick_count = chick_count + 1

                ### vẽ khung con gà
                start_point = (940 + best_box[0],best_box[3])
                end_point = (940 + best_box[2],best_box[1])
                color = (0, 0, 255)
                thickness = 3
                image = cv2.rectangle(image, start_point, end_point, color, thickness)
                # cv2.imwrite(os.path.join(frame_folder, '%d.jpg' % n_frames), image)

                pre_chick = current_chick
            else:
                if current_chick == True:
                    current_chick = False

            ##### put chick_count 
            # font
            font = cv2.FONT_HERSHEY_SIMPLEX
            # org
            org = (round(940 + 1400/2)-10, 200)
            # fontScale
            fontScale = 6
            # Blue color in BGR
            color = (255, 0, 0)
            # Line thickness of 2 px
            thickness = 8
            # Using cv2.putText() method
            image = cv2.putText(image, str(chick_count), org, font, 
                            fontScale, color, thickness, cv2.LINE_AA)
            

            cv2.imwrite(os.path.join(frame_folder, '%d.jpg' % n_frames), image)
            #### ghi video
            out.write(np.uint8(image))


        vidcap.release()
        out.release()       
 

