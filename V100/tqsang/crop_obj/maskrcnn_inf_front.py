import os
import cv2
# import json
import time
# import torch
import numpy as np

from tqdm import tqdm


import math
import numpy 


### cho maskrcnn
from PIL import Image
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from engine import train_one_epoch, evaluate
import utils
import transforms as T

if __name__ == '__main__':
    vid_dir_out = '/mnt/tqsang/smart_plant/vid_out/front'
    vid_dir = '/mnt/tqsang/cut/front'
    frame_dir = '/mnt/tqsang/smart_plant/frames_inf/front'


    vid_names = sorted(os.listdir(vid_dir))


    ### load model 

    model = torch.jit.load('/home/tqsang/V100/tqsang/crop_obj/maskrcnn.pt')
    model.eval()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    for vid_name in tqdm(vid_names):
        start = time.time()
        vid_name_path = os.path.join(vid_dir, vid_name)
        frame_folder = os.path.join(frame_dir, os.path.splitext(vid_name)[0])

        # if vid_name not in ['4_Trim_1.mp4' , '4_Trim_2.mp4', '5_Trim.mp4', '10_Trim.mp4']:
        #     continue
        vid_name = os.path.splitext(vid_name)[0]
        # frame_folder = '/home/tqsang/V100/tqsang/crop_obj/front'


        try:
            os.makedirs(frame_folder)
            os.makedirs(frame_folder + '/mask')
            os.makedirs(frame_folder + '/img')
        except:
            pass


        print("Video name: ", vid_name)
        vidcap = cv2.VideoCapture(vid_name_path)

        frame_width = int(vidcap.get(3))
        frame_height = int(vidcap.get(4))
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
        out = cv2.VideoWriter(os.path.join(vid_dir_out,'out_' + vid_name + '.avi'), \
        fourcc, fps, (1400 - 450,(1080 - 120)*2)) ### w_crop h_crop , 2 img stack nên là 2*h_crop
        success,image = vidcap.read()

        middle_frames = []
        n_frames = 0

        current_chick = False
        pre_chick = False
        chick_count = 0

        while success:
            
            success,image = vidcap.read()
            n_frames += 1     

            # print(n_frames)
            if image is None: 
                continue
            middle_frame = image # (h = 1080, w = 1920, 3)
            
            height = 1080
            width = 1920
            ####### crop màn đen (calib x_crop với w_crop)
            y_crop = 0
            h_crop = height - 120 
            x_crop = 450 
            w_crop = 1400 - 450

            middle_frame = middle_frame[y_crop:y_crop + h_crop, x_crop:x_crop + w_crop]
            sav_middle_frame = np.uint8(middle_frame)

            #### convert cv to PIL image
            # You may need to convert the color.

            middle_frame = cv2.cvtColor(middle_frame, cv2.COLOR_BGR2RGB)
            middle_frame = Image.fromarray(middle_frame)

            middle_frame = torchvision.transforms.ToTensor()(middle_frame)#.unsqueeze_(0)

            # convert_tensor = torchvision.transforms.ToTensor()
            # convert_tensor(middle_frame)

            # middle_frame = torch.from_numpy(middle_frame).to(device='cuda')
            try:
                with torch.no_grad():
                    _,prediction = model([middle_frame.to(device)])
            except:
                continue
            # print(prediction)
            # import pdb; pdb.set_trace()
            # print(prediction)
            if len(prediction)>0:
                try:
                    image  = Image.fromarray(prediction[0]['masks'][0, 0].mul(255).byte().cpu().numpy())
                    #### ghi video
                    image = numpy.array(image) 
                    # image = image[:, :, ::-1].copy() 

                    cv2.imwrite(os.path.join(frame_folder, '%d.jpg' %(n_frames)), image) ### all 
                    # import pdb; pdb.set_trace()
                    image = cv2.merge((image,image,image)) ### chuyển thành 3 channel để ghi vid 

                    vid_img = cv2.vconcat([sav_middle_frame, image]) ### ghép 2 img theo phương thẳng đứng
                    out.write(np.uint8(vid_img))
                except:
                    image = np.zeros((h_crop, w_crop, 3), np.uint8) #### tạo hình blank 
                    try:
                        vid_img = cv2.vconcat([sav_middle_frame, image]) ### ghép 2 img theo phương thẳng đứng
                    except:
                        import pdb; pdb.set_trace()
                    out.write(np.uint8(vid_img))
                    continue


        vidcap.release()
        out.release()       
 

