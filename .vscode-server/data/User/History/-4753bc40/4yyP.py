import splitfolders

# splitfolders.ratio("/home/tqsang/V100/tqsang/crop_obj/back/data/", output="back_pad", seed=1337, ratio=(.8, 0.2,0.0)) 
# splitfolders.ratio("/mnt/tqsang/andre", output="/mnt/tqsang/andre_labelme_train_val", seed=1337, ratio=(.8, 0.2,0.0)) 
# splitfolders.ratio("/mnt/tqsang/chicken_part/train_val", output="/mnt/tqsang/chicken_part/split_train_val_xml", seed=1337, ratio=(.8, 0.2,0.0)) 
splitfolders.ratio("/mnt/tqsang/chicken_part/train_val", output="/mnt/tqsang/chicken_part/split_train_val_xml", seed=1337, ratio=(.8, 0.2,0.0)) 