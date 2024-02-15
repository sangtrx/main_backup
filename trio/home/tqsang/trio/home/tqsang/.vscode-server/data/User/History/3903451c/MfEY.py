import argparse
import glob
import multiprocessing as mp
import numpy as np
import os
import tempfile
import time
import warnings
import cv2
import tqdm
import xml.etree.ElementTree as ET

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from predictor import VisualizationDemo

WINDOW_NAME = "COCO detections"


def create_annotation(predictions, frame_number, video_path, frame_width, frame_height):
    # Create the structure of the XML file
    annotation = ET.Element("annotation")
    folder = ET.SubElement(annotation, "folder")
    folder.text = "Camera Roll"
    filename = ET.SubElement(annotation, "filename")
    filename.text = os.path.basename(video_path)
    path = ET.SubElement(annotation, "path")
    path.text = video_path
    source = ET.SubElement(annotation, "source")
    database = ET.SubElement(source, "database")
    database.text = "Unknown"
    size = ET.SubElement(annotation, "size")
    width = ET.SubElement(size, "width")
    width.text = str(frame_width)
    height = ET.SubElement(size, "height")
    height.text = str(frame_height)
    depth = ET.SubElement(size, "depth")
    depth.text = "3"
    segmented = ET.SubElement(annotation, "segmented")
    segmented.text = "0"

    # Iterate over detected instances and create object elements
    instances = predictions["instances"].to("cpu")
    for i in range(len(instances)):
        obj = ET.SubElement(annotation, "object")
        name = ET.SubElement(obj, "name")
        name.text = str(instances.pred_classes[i].item())
        pose = ET.SubElement(obj, "pose")
        pose.text = "Unspecified"
        truncated = ET.SubElement(obj, "truncated")
        truncated.text = "0"
        difficult = ET.SubElement(obj, "difficult")
        difficult.text = "0"
        bndbox = ET.SubElement(obj, "bndbox")
        xmin = ET.SubElement(bndbox, "xmin")
        ymin = ET.SubElement(bndbox, "ymin")
        xmax = ET.SubElement(bndbox, "xmax")
        ymax = ET.SubElement(bndbox, "ymax")
        xmin.text, ymin.text, xmax.text, ymax.text = [str(x) for x in instances.pred_boxes[i].tolist()]

    # Save the XML file
    tree = ET.ElementTree(annotation)
    tree.write(f"frame_{frame_number:09d}.xml")


# ... (keep the rest of the code unchanged until the `if __name__ == "__main__":` block)

if __name__ == "__main__":
    # ... (keep the rest of the code unchanged until the `elif args.video_input:` block)

    elif args.video_input:
        vidcap = cv2.VideoCapture(args.video_input)
        width = int(vidcap.get(3))
        height = int(vidcap.get(4))

        fps = 30
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')

        success, image = vidcap.read()
        n_frames = 0
        while success:
            success, image = vidcap.read()
            n_frames += 1
            if image is None:
                continue

            predictions, visualized_output = demo.run_on_image(image)

            # Create and save XML annotation file for the current frame
            create_annotation(predictions, n_frames, args.video_input, width, height)

            # The following lines are for visualization and saving the video with detected objects
            # You can comment these lines out if you don't want to save the video
            out.write(np.uint8(visualized_output.get_image()[:, :, ::-1]))

        # Release the video writer object
        out.release()
