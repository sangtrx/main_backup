from mmdet.apis import init_detector, inference_detector
import mmcv

import cv2
from tqdm import tqdm
import os

# Specify the path to model config and checkpoint file
config_file = '/home/tqsang/mmdetection/work_dirs/mask2former_r50_lsj_8x2_50e_coco_front2class_syn/mask2former_r50_lsj_8x2_50e_coco_front2class_syn.py'
checkpoint_file = '/home/tqsang/mmdetection/work_dirs/mask2former_r50_lsj_8x2_50e_coco_front2class_syn/latest.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# # test a single image and show the results
img = '/home/tqsang/V100/tqsang/crop_obj/front_2_class_COCO_synthetic_bg/val2017/4_Trim_1_43_1__4_Trim_2_1222_6__5_Trim_1240_6.jpg'  # or img = mmcv.imread(img), which will only load it once
result = inference_detector(model, img)
# visualize the results in a new window
# model.show_result(img, result)
# or save the visualization results to image files
model.show_result(img, result, out_file='result.jpg')

# test a video and show the results

# vid_dir_out = '/mnt/tqsang/smart_plant/vid_out/back'
# vid_dir = '/mnt/tqsang/cut/back'
# frame_dir = '/mnt/tqsang/smart_plant/frames_inf/back'
# vid_names = sorted(os.listdir(vid_dir))

# for vid_name in tqdm(vid_names):
#     vid_name_path = os.path.join(vid_dir, vid_name)
#     vid_name = os.path.splitext(vid_name)[0]

#     print("Video name: ", vid_name)

#     # fps = vidcap.get(cv2.CAP_PROP_FPS) #30
#     # fps = 30
#     # fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
#     # out = cv2.VideoWriter(os.path.join(vid_dir_out,'out_' + vid_name + '.avi'), \
#     # fourcc, fps, (1400 - 450,(1080 - 120)*2)) ### w_crop h_crop , 2 img stack nên là 2*h_crop

#     video_reader = mmcv.VideoReader(vid_name_path)
#     video_writer = None

#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     video_writer = cv2.VideoWriter(
#         os.path.join(vid_dir_out,'out_' + vid_name + '.avi'), fourcc, video_reader.fps,
#         (video_reader.width, video_reader.height))

#     for frame in mmcv.track_iter_progress(video_reader):
#         result = inference_detector(model, frame)
#         frame = model.show_result(frame, result, score_thr=0.3)

#         video_writer.write(frame)


#     video_writer.release()
#     # cv2.destroyAllWindows()
