from bdb import set_trace
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
    # vid_dir_out = '/mnt/tqsang/smart_plant/vid_out/front'
    vid_dir = '/mnt/tqsang/chicken_part2'
    # frame_dir = '/mnt/tqsang/chicken_part/frames'


    vid_names = sorted(os.listdir(vid_dir))
    count_vid = 0
    for vid_name in tqdm(vid_names):
        count_vid = count_vid + 1
        start = time.time()
        vid_name_path = os.path.join(vid_dir, vid_name)
        # frame_folder = os.path.join(frame_dir, os.path.splitext(vid_name)[0])

        # if vid_name not in ['2_dinh_Trim.mp4']:
        # # if vid_name not in ['4_Trim_1.mp4']:
        #     continue
        vid_name = os.path.splitext(vid_name)[0]
        frame_folder = '/mnt/tqsang/chicken_part2/frames'


        # try:
        #     os.makedirs(frame_folder)
        #     os.makedirs(frame_folder + '/mask')
        #     os.makedirs(frame_folder + '/img')
        #     os.makedirs(frame_folder + '/vis')
        # except:
        #     pass


        print("Video name: ", vid_name)
        vidcap = cv2.VideoCapture(vid_name_path)

        # frame_width = int(vidcap.get(3))
        # frame_height = int(vidcap.get(4))
        # fps = vidcap.get(cv2.CAP_PROP_FPS)
        # fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
        # out = cv2.VideoWriter(os.path.join(vid_dir_out,'out_' + vid_name + '.avi'), fourcc, fps, (frame_width,frame_height))
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
            # if image is None: 
            #     continue
            middle_frame = image 



            cv2.imwrite(os.path.join(frame_folder, '%08d.png' %(n_frames)), middle_frame) 



        vidcap.release()
        # out.release()       
 

