from bdb import set_trace
import os
from turtle import pd
import cv2
# import json
import time
# import torch
import numpy as np

from tqdm import tqdm


import math
import numpy 
if __name__ == '__main__':
    vid_dir_out = '/mnt/tqsang/smart_plant/vid_out/front'
    vid_dir = '/home/tqsang/smart_plant'
    frame_dir = '/mnt/tqsang/smart_plant/frames/front'


    vid_names = sorted(os.listdir(vid_dir))
    count_vid = 0
    for vid_name in tqdm(vid_names):
        count_vid = count_vid + 1
        start = time.time()
        vid_name_path = os.path.join(vid_dir, vid_name)
        # frame_folder = os.path.join(frame_dir, os.path.splitext(vid_name)[0])

        # if vid_name not in ['4_Trim_1.mp4' , '4_Trim_2.mp4', '5_Trim.mp4', '10_Trim.mp4']:
        # # if vid_name not in ['4_Trim_1.mp4']:
        #     continue
        vid_name = os.path.splitext(vid_name)[0]
        frame_folder = '/home/tqsang/V100/tqsang/crop_obj/front_2_class_test'


        try:
            os.makedirs(frame_folder)
            os.makedirs(frame_folder + '/mask')
            os.makedirs(frame_folder + '/img')
            os.makedirs(frame_folder + '/vis')
        except:
            pass


        print("Video name: ", vid_name)
        vidcap = cv2.VideoCapture(vid_name_path)

        frame_width = int(vidcap.get(3))
        frame_height = int(vidcap.get(4))
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
        # out = cv2.VideoWriter(os.path.join(vid_dir_out,'out_' + vid_name + '.avi'), fourcc, fps, (frame_width,frame_height))
        out = cv2.VideoWriter(os.path.join(vid_dir_out,'out_' + vid_name + '.avi'), fourcc, fps, (frame_width,frame_height))
        success,image = vidcap.read()

        middle_frames = []
        n_frames = 0

        current_chick = False
        pre_chick = False
        chick_count = 0

        while success:
            
            success,image = vidcap.read()
            n_frames += 1
            # if (n_frames < 2900 + 50) or (n_frames > 7679): #2044:# 1920:
            #     continue
            # if n_frames > 3000: #2044:# 1920:
            #     break           

            # print(n_frames)
            if image is None: 
                continue
            middle_frame = image # (h = 1080, w = 1920, 3)
            
            height = 2160
            width = 3840
            ####### crop màn đen (calib x_crop với w_crop)
            y_crop = 300 ## 0
            h_crop = 2160
            x_crop = 940 
            w_crop = 1400

            middle_frame = middle_frame[y_crop:y_crop + h_crop, x_crop:x_crop + w_crop]


            ### red channel 
            red_channel = middle_frame.copy()
            red_channel[:, :, 0] = 0 # set blue and green channels to 0
            red_channel[:, :, 1] = 0
            
            red_imgray = cv2.cvtColor(red_channel,cv2.COLOR_BGR2GRAY)
            
            #### tách làm 2 để lọc hình cho dễ
            red_height = red_imgray.shape[0]
            red_width = red_imgray.shape[1]
            red_imgray_up = red_imgray[ : red_height//2 , : ]
            red_imgray_down = red_imgray[ red_height//2 : , : ]

            ret,red_thresh_down = cv2.threshold(red_imgray_down, 20,255,cv2.THRESH_BINARY) #### 44 calib
            ret,red_thresh_up = cv2.threshold(red_imgray_up, 23,255,cv2.THRESH_BINARY) ## 80 càng lớn càng mất 
            red_thresh = np.concatenate((red_thresh_up, red_thresh_down), axis=0)
            


            # ##################### 3 channels
            imgray = cv2.cvtColor(middle_frame,cv2.COLOR_BGR2GRAY)

            #### tách làm 2 để lọc hình cho dễ
            height = imgray.shape[0]
            width = imgray.shape[1]
            imgray_up = imgray[ : height//2 , : ]
            imgray_down = imgray[ height//2 : , : ]

            ret,thresh_down = cv2.threshold(imgray_down, 80 ,255,cv2.THRESH_BINARY) #### 44 calib
            ret,thresh_up = cv2.threshold(imgray_up, 70,255,cv2.THRESH_BINARY) ## 80 càng lớn càng mất 
            # thresh_up = cv2.adaptiveThreshold(imgray_up,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY , 11 ,2)
            thresh = np.concatenate((thresh_up, thresh_down), axis=0)


            thresh =  np.add(thresh,red_thresh)
            # cv2.imwrite(os.path.join(frame_folder, '%d_imgray.jpg' % n_frames), thresh)

            ###### lọc hình trực tiếp 
            # ret,thresh = cv2.threshold(imgray,math.floor(numpy.average(imgray)),255,cv2.THRESH_BINARY)
            # ret,thresh = cv2.threshold(imgray,math.floor(numpy.average(imgray)) + 20,255,cv2.THRESH_BINARY)
            # if n_frames == 124:
            #     print(math.floor(numpy.average(imgray)))
            # cv2.imwrite(os.path.join(frame_folder, '%d_imgray.jpg' % n_frames), thresh)

            # thresh  = cv2.adaptiveThreshold(imgray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY , 11 ,2)



            dilated=cv2.morphologyEx(thresh, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(1,1))) # (10,10) càng nhỏ thì detail càng nhiều
            # _,contours,_ = cv2.findContours(dilated,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)  ### cv2 bản cũ 
            contours,_ = cv2.findContours(dilated,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
            
            new_contours=[]
            max_con = 0
            for c in contours:
                #### tìm khoảng hợp lý cho contour
                # print(cv2.contourArea(c))
                # max_con = max(max_con,cv2.contourArea(c))
                # if n_frames == 706:
                #     print(max_con)
                if 2000000>cv2.contourArea(c)>15000:
                    new_contours.append(c)
            
            # new_contours = max(new_contours, key = cv2.contourArea) ### tìm contour lớn nhất 


            best_box=[-1,-1,-1,-1]
            for c in new_contours:
                x,y,w,h = cv2.boundingRect(c)
                ##### vẽ tất cả các contours
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


            

            #### mask big contour trên toàn bộ image lớn (RGB để view) muốn chuyển thành mask thật phải giảm chiều
            # mask = np.zeros(image.shape, np.uint8)   ### tạo shape giống image
            # cv2.drawContours(mask, new_contours, -1, color=(255, 255, 255), thickness=cv2.FILLED) 
            # M = np.float32([[1, 0, x_crop],[0, 1, 0]]) ### transformation matrix, dịch  đoạn x = x_crop
            # mask = cv2.warpAffine(mask, M, (mask.shape[1], mask.shape[0]))
            # cv2.imwrite(os.path.join(frame_folder, '%d_mask.jpg' % n_frames), mask)


            #### mask big contour trên middle frame (RGB để view) muốn chuyển thành mask thật phải giảm chiều
            # mask = np.zeros(middle_frame.shape, np.uint8)   ### tạo shape giống image
            # cv2.drawContours(mask, new_contours, -1, color=(255, 255, 255), thickness=cv2.FILLED) 
            # cv2.imwrite(os.path.join(frame_folder, '%d_mask.jpg' % n_frames), mask)

            '''
            ###### mask màu
            # color_mask = image
            # alpha = 0.1
            # color_mask  = (1-alpha)*image + (~mask)*[0,255,0]*alpha

            # a =  ~mask*[0,255,0]

            ### gà only
            # a =  ~mask*[255,255,255]
            # color_mask = cv2.addWeighted(image, 1, a, 1, 0, dtype = cv2.CV_8U)

            ### gà blue
            # a =  mask*[255,255,0]
            # color_mask = cv2.addWeighted(image, 1-0.0019, a, 0.0019, 0, dtype = cv2.CV_8U)

            # cv2.imwrite(os.path.join(frame_folder, '%d_color_mask.jpg' % n_frames), color_mask)

            # image = color_mask
            '''

            ### vẽ khung con gà trong hình crop
            # start_point = (best_box[0],best_box[3])
            # end_point = (best_box[2],best_box[1])
            # color = (255, 255, 0)
            # thickness = 3
            # middle_frame = cv2.rectangle(middle_frame, start_point, end_point, color, thickness)
            # cv2.imwrite(os.path.join(frame_folder, '%d.jpg' % n_frames), middle_frame)
            #### middle_frame = middle_frame[best_box[1]:best_box[3], best_box[0]:best_box[2]]

            ### vẽ khung của màn đen / khung crop trên toàn image 
            # start_point = (x_crop , h_crop)
            # end_point = (x_crop + w_crop , 0)
            # color = (255, 255, 0)
            # thickness = 8
            # image = cv2.rectangle(image, start_point, end_point, color, thickness)
            # cv2.imwrite(os.path.join(frame_folder, '%d_all.jpg' %(n_frames)), image) ### all 

            #### check if khung con gà overlaps khung crop or not 
            if (x_crop + best_box[0] > x_crop) and (x_crop + best_box[2] < x_crop + w_crop):
                # if (current_chick == False) and (pre_chick == False):
                #     current_chick = True
                #     chick_count = chick_count + 1
                
                if current_chick == False:
                    current_chick = True
                    chick_count = chick_count + 1

                ### vẽ khung con gà
                # start_point = (x_crop + best_box[0],best_box[3])
                # end_point = (x_crop + best_box[2],best_box[1])
                # color = (0, 0, 255)
                # thickness = 3
                # image = cv2.rectangle(image, start_point, end_point, color, thickness)


                ##### save mỗi phần crop (middle_frame)  và con gà (save_frame)
                # save_frame = middle_frame[best_box[1]:best_box[3], best_box[0]:best_box[2]]
                # cv2.imwrite(os.path.join(frame_folder, '%d_%d_obj.jpg' %(n_frames,chick_count)), save_frame) ### crop con gà 

                # cv2.imwrite(os.path.join(frame_folder, 'img/%d_%d.jpg' %(n_frames,chick_count)), middle_frame) #### crop khung
                cv2.imwrite(os.path.join(frame_folder, 'img/%s_%d_%d.jpg' %(vid_name,n_frames,chick_count)), middle_frame) #### crop khung

                mask = np.zeros(middle_frame.shape, np.uint8)   ### tạo shape giống image
                cv2.drawContours(mask, new_contours, -1, color=(255, 255, 255), thickness=cv2.FILLED) 
                # cv2.imwrite(os.path.join(frame_folder, 'mask/%d_%d.jpg' %(n_frames,chick_count)), mask) #### mask
                cv2.imwrite(os.path.join(frame_folder, 'mask/%s_%d_%d.jpg' %(vid_name,n_frames,chick_count)), mask) #### mask

                ###### mỗi class 1 màu 


                #### vis
                a =  mask*[255,255,0]
                color_mask = cv2.addWeighted(middle_frame, 1-0.0019, a, 0.0019, 0, dtype = cv2.CV_8U)

                cv2.imwrite(os.path.join(frame_folder, 'vis/%s_%d_%d.jpg' %(vid_name,n_frames,chick_count)), color_mask) #### mask



                # cv2.imwrite(os.path.join(frame_folder, '%d_%d_all.jpg' %(n_frames,chick_count)), image) ### all 

                # path_txt = os.path.join(frame_folder, '%d_%d.txt' %(n_frames,chick_count))
                # with open(path_txt, 'a+') as f:
                #     f.write('?' + ' ' + str(best_box[0]) + ' ' + str(best_box[1]) + ' ' + str(best_box[2]-best_box[0]) + ' ' + str(best_box[3]-best_box[1]) + '\n')

                # pre_chick = current_chick
            else:
                if current_chick == True:
                    current_chick = False

            ##### put chick_count 
            # # font
            # font = cv2.FONT_HERSHEY_SIMPLEX
            # # org
            # org = (round(940 + 1400/2), 200)
            # # fontScale
            # fontScale = 6
            # # Blue color in BGR
            # color = (255, 0, 0)
            # # Line thickness of 2 px
            # thickness = 5
            # # Using cv2.putText() method
            # image = cv2.putText(image, str(chick_count), org, font, 
            #                 fontScale, color, thickness, cv2.LINE_AA)
            

            # cv2.imwrite(os.path.join(frame_folder, '%d.jpg' % n_frames), image)
            #### ghi video
            import pdb;pdb.set_trace()
            middle_frame
            out.write(np.uint8(image))


        vidcap.release()
        out.release()       
 

