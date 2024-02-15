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
    vid_dir_out = '/mnt/tqsang/smart_plant/vid_out/front'
    vid_dir = '/mnt/tqsang/cut/front'
    frame_dir = '/mnt/tqsang/smart_plant/frames/front'


    vid_names = sorted(os.listdir(vid_dir))
    count_vid = 1

    count_vid = count_vid + 1
    start = time.time()
    # frame_folder = os.path.join(frame_dir, os.path.splitext(vid_name)[0])
    vid_name_path = '/mnt/tqsang/cut/back/6_dinh_Trim.mp4'

    frame_folder = '/home/tqsang/V100/tqsang/crop_obj/test_data_DrLe'


    try:
        os.makedirs(frame_folder)
        os.makedirs(frame_folder + '/mask')
        os.makedirs(frame_folder + '/img')
    except:
        pass


    vidcap = cv2.VideoCapture(vid_name_path)

    frame_width = int(vidcap.get(3))
    frame_height = int(vidcap.get(4))
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
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
        if n_frames % 90 != 0:
            continue
        middle_frame = image # (h = 1080, w = 1920, 3)
        
        
        cv2.imwrite(os.path.join(frame_folder, 'img/b_%d_%d_%d.jpg' %(count_vid,n_frames,chick_count)), middle_frame) #### crop khung


    vidcap.release()
    # out.release()       
 

