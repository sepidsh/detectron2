# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import tqdm

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from predictor import VisualizationDemo

# constants
WINDOW_NAME = "COCO detections"


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 Demo")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/e2e_mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument("--input", nargs="+", help="A list of space separated input images")
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
        help="Modify model config options using the command-line",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)


    demo = VisualizationDemo(cfg)
    f = open("/mnt/datasets/vlog/list.txt", "r")
    for x in f:
        
        video = cv2.VideoCapture("/mnt/datasets/vlog/"+x[:-1])
        if args.output:
           
            
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            output_file="/mnt/datasets/vlog/"+x[:-7]+"with_object_detection"+".avi"
            out = cv2.VideoWriter(output_file,fourcc, 20.0, (256,256))
            while(video.isOpened()):
                ret,frame = video.read()
               
                if ret==True:
                    predictions, visualized_output = demo.run_on_image(frame)
            
                
                
                
              
                    out.write(visualized_output.get_image()[:, :, ::-1])
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    break 
            video.release()
            out.release()
            cv2.destroyAllWindows()   


