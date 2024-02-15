import cv2
from tqdm import tqdm
import os 

if __name__ == '__main__':
    scr_folder = '/home/tqsang/V100/tqsang/crop_obj/front_2_class/mask'
    out_folder = '/home/tqsang/V100/tqsang/crop_obj/front_2_class_pad/mask'


    img_names = sorted(os.listdir(scr_folder))

    borderType = cv2.BORDER_CONSTANT

    for img_name in tqdm(img_names):
        # read image
        src = cv2.imread(os.path.join(scr_folder,img_name))
        top = int(0 * src.shape[0])  # shape[0] = rows
        bottom = top
        left = int(0.75 * src.shape[1])  # shape[1] = cols
        right = left
        pad_img = cv2.copyMakeBorder(src, top, bottom, left, right, borderType, value = [0,0,0]) ### img thi [46,45,48] mask thi [0,0,0]
        # print(img_name)
        cv2.imwrite(os.path.join(out_folder,img_name), pad_img)
 

