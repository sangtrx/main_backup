import fiftyone as fo

# The directory containing the source images
data_path = "/home/tqsang/V100/tqsang/crop_obj/front_2_class_COCO/val2017"

# The path to the COCO labels JSON file
labels_path = "/home/tqsang/V100/tqsang/crop_obj/front_2_class_COCO/annotations/instances_val2017.json"


# Import the dataset
dataset = fo.Dataset.from_dir(
    dataset_type=fo.types.COCODetectionDataset,
    label_types=["detections", "segmentations"],
    data_path=data_path,
    labels_path=labels_path,
)

session = fo.launch_app(dataset)
session.wait()