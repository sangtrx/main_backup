import json, os
from os.path import join
import shutil

annFile = "/mnt/tqsang/runs/feather/test.json"
srcImgFold = "/mnt/tqsang/andre_files"
dstImgFold = "/mnt/tqsang/adre_test_img"

try:
    os.mkdir(dstImgFold)
except:
    pass


with open(annFile, "r") as f:
    ann = json.load(f)

for img in ann['images']:
    print(img['file_name'])
    imgPath = join(srcImgFold, img['file_name'])
    dstPath = join(dstImgFold, img['file_name'])
    shutil.copy(imgPath, dstPath)