# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detectron2/blob/master/demo/demo.py
import argparse
import glob
import multiprocessing as mp
import os

# fmt: off
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# fmt: on

import tempfile
import time
import warnings

import cv2
import numpy as np
import tqdm

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger

from mask2former import add_maskformer2_config
from predictor import VisualizationDemo


# constants
WINDOW_NAME = "mask2former demo"


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="maskformer2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/coco/panoptic-segmentation/maskformer2_R50_bs16_50ep.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


def test_opencv_video_format(codec, file_ext):
    with tempfile.TemporaryDirectory(prefix="video_format_test") as dir:
        filename = os.path.join(dir, "test_file" + file_ext)
        writer = cv2.VideoWriter(
            filename=filename,
            fourcc=cv2.VideoWriter_fourcc(*codec),
            fps=float(30),
            frameSize=(10, 10),
            isColor=True,
        )
        [writer.write(np.zeros((10, 10, 3), np.uint8)) for _ in range(30)]
        writer.release()
        if os.path.isfile(filename):
            return True
        return False

from detectron2.data import transforms as T
def build_transform_gen():
    """
    Create a list of default :class:`Augmentation` from config.
    Now it includes resizing and flipping.
    Returns:
        list[Augmentation]
    """
    image_size = 256
    min_scale = 1
    max_scale = 1

    augmentation = []


    augmentation.extend([
        T.ResizeScale(
            min_scale=min_scale, max_scale=max_scale, target_height=image_size, target_width=image_size
        ),
        # T.FixedSizeCrop(crop_size=(image_size, image_size)),
    ])

    return augmentation

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)

    if args.input:
        if len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"

        fps = 30
        fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
        out = cv2.VideoWriter(os.path.join('out_test'+ '.avi'), fourcc, fps, (1400,1860))
        
        for path in tqdm.tqdm(args.input, disable=not args.output):
            # use PIL, to be consistent with evaluation
            img = read_image(path, format="BGR")
            start_time = time.time()
            predictions, visualized_output = demo.run_on_image(img)
            # import pdb;pdb.set_trace()
            # Get only the person mask (binary)
            # panoptic_img = predictions['instances']
            # person_id = 1
            # person_img = np.where(panoptic_img == person_id, 1, 0) * 255
            # person_img = person_img.astype(np.uint8)

            # import pdb;pdb.set_trace()
            logger.info(
                "{}: {} in {:.2f}s".format(
                    path,
                    "detected {} instances".format(len(predictions["instances"]))
                    if "instances" in predictions
                    else "finished",
                    time.time() - start_time,
                )
            )

            if args.output:
                if os.path.isdir(args.output):
                    assert os.path.isdir(args.output), args.output
                    out_filename = os.path.join(args.output, os.path.basename(path))
                else:
                    assert len(args.input) == 1, "Please specify a directory with args.output"
                    out_filename = args.output
                visualized_output.save(out_filename)
                out.write(np.uint8(visualized_output.get_image()[:, :, ::-1]))  ## lưu video
            else:
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
                if cv2.waitKey(0) == 27:
                    break  # esc to quit

        out.release()
    elif args.video_input:

        vidcap = cv2.VideoCapture(args.video_input)
        width = int(vidcap.get(3))
        height = int(vidcap.get(4))

        fps = 30
        fourcc = cv2.VideoWriter_fourcc('M','J','P','G')


        y_crop = 300
        h_crop =  700 #600 
        x_crop = 650
        w_crop = 700 #600

        out = cv2.VideoWriter(os.path.join('out_'+ '.avi'), fourcc, fps, (w_crop,h_crop))

        success,image = vidcap.read()
        n_frames = 0
        while success:
            success,image = vidcap.read()
            n_frames += 1
            if image is None: 
                continue
            # import pdb;pdb.set_trace()


            cropped_frame = image[y_crop:y_crop + h_crop, x_crop:x_crop + w_crop]


            ### resize 

            # tfm_gens = build_transform_gen()
            # cropped_frame, transforms = T.apply_transform_gens(tfm_gens, cropped_frame)



            predictions, visualized_output = demo.run_on_image(cropped_frame)

            # import pdb;pdb.set_trace()


            visualized_output.save('/home/tqsang/Mask2Former/demo/test/'+str(n_frames)+'.jpg')

            out.write(np.uint8(visualized_output.get_image()[:, :, ::-1]))

        out.release()
