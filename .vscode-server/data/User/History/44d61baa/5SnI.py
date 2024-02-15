import fiftyone as fo

name = "hand_data3"
dataset_dir = "/mnt/tqsang/hand_dataset_YOLO/datasets"

# The splits to load
splits = ["train", "val"]

# Load the dataset, using tags to mark the samples in each split
dataset = fo.Dataset(name)
for split in splits:
    dataset.add_dir(
        dataset_dir=dataset_dir,
        dataset_type=fo.types.YOLOv5Dataset,
        split=split,
        tags=split,
)

session = fo.launch_app(dataset)
session.wait()