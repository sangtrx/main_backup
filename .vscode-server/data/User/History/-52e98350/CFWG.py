import fiftyone as fo

# The directory containing the source images
# data_path = "/home/tqsang/V100/tqsang/crop_obj/front_2_class_feather/train2017"
data_path = "/mnt/tqsang/data_chick_part/output/"

# The path to the COCO labels JSON file
# labels_path = "/home/tqsang/V100/tqsang/crop_obj/front_2_class_1chick1img/train.json"
# labels_path = "/home/tqsang/V100/tqsang/crop_obj/front_2_class_feather/annotations/instances_train2017.json"
labels_path = "/mnt/tqsang/data_chick_part/val.json"


# Import the dataset
dataset = fo.Dataset.from_dir(
    dataset_type=fo.types.COCODetectionDataset,
    label_types=["detections", "segmentations"],
    data_path=data_path,
    labels_path=labels_path,
)

session = fo.launch_app(dataset)
session.wait()