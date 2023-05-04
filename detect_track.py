import argparse
from datetime import datetime
from typing import Union
import time
import os
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
from models.cmc import CMC
from numpy import random

from random import randint
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, \
                check_imshow, non_max_suppression, \
                scale_coords, xyxy2xywh, strip_optimizer, set_logging \
#                 increment_path
# from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
# from utils.download_weights import download
from bounding_box_plotter import BoundingBoxPlotter
from mot_converter import MOTConverter
from tracking.sort import Sort
from libs.tracker import Sort_OH
from tracking.deepsort.deepsort import DeepSort

from loguru import logger
from tqdm import tqdm
from typing import Tuple

from torchreid.utils.avgmeter import AverageMeter

def preprocess(img, device: torch.device, half: bool) -> torch.Tensor:
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    return img

def get_ori_imsize(dir: str) -> Tuple[int, int]:
    im_path = list(Path(dir).iterdir())[0]
    img = cv2.imread(str(im_path))
    return img.shape[:2]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-w', '--weights', nargs='+', type=str, default='./weights/yolov7-tiny.pt', help='model.pt path(s)')
    parser.add_argument('--webcam', action="store_true", help='Whether to use webcam or not')
    parser.add_argument('-s', '--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img_size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--max-age', type=int, default=1, help='Max age of a track before it is deleted')
    parser.add_argument('--min-hits', type=int, default=3, help='Min number of detections before a track is confirmed')
    parser.add_argument('--track-iou-thres', type=float, default=0.3, help='IOU threshold for Data Association in Tracking')
    parser.add_argument('--save-vid', type=str, default='output1.mp4', help='Filename of the output video')
    parser.add_argument('-o', '--output-result', action="store_true", help='Store Tracking Results and Videos')
    parser.add_argument('--tracker', type=str, default='sort', choices=['sort', 'sort_oh', 'deepsort'], help='Tracker to use')
    parser.add_argument('--pipe-result', action="store_true", help='Whether to Pipe Results to TrackEval')
    parser.add_argument('--trackeval-dir', type=str, default='../TrackEval', help='TrackEval Directory')
    parser.add_argument('--benchmark', type=str, default='MyDataset', help='Benchmark in TrackEval to use')
    parser.add_argument('--ema-weight', type=float, default=0.9, help='Weight to update appreance in EMA')
    parser.add_argument('--ema', action="store_true", help='Whether to use EMA Scheme to Update')
    parser.add_argument('--cmc-scale', type=float, default=0.5, help='Scaling factor for CMC')
    parser.add_argument('--max-dist', type=float, default=0.3, help='Threshold Distance for Cosine Distance')

    args = parser.parse_args()
    weights, img_size = args.weights, args.img_size

    device = select_device(args.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model

    model = attempt_load(weights, map_location=device, )  # load FP32 model
    mot_converter = MOTConverter()
    model.eval()
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(img_size, s=stride)  # check img_size

    model = TracedModel(model, device, img_size).eval()

    detect_runtime = AverageMeter()
    nms_runtime = AverageMeter()
    track_runtime = AverageMeter()

    if half:
        model.half()
        cudnn.benchmark = True

    cmc = CMC("ECC", scale=args.cmc_scale)

    vid_path, vid_writer = None, None
    webcam = args.webcam
    #! TODO: Either receive inference from webcam or from image
    mot_converter = MOTConverter()
    bb_plotter = BoundingBoxPlotter()
    if args.output_result:
        out_file = Path(f'output/{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt').resolve()
        out_file.touch(exist_ok=False)
        
    if args.tracker == 'sort':
        sort = Sort(max_age=args.max_age, min_hits=args.min_hits, iou_threshold=args.track_iou_thres)
    elif args.tracker == 'sort_oh':
        sort_oh = Sort_OH(max_age=args.max_age, min_hits=args.min_hits, conf_trgt=args.conf_thres, conf_obj=0.75)
    elif args.tracker == 'deepsort':
        deepsort = DeepSort(n_init=args.min_hits, 
                            max_age=args.max_age, 
                            max_iou_distance=args.track_iou_thres, 
                            max_dist=args.max_dist, 
                            use_EMA=args.ema, 
                            )
    else:
        raise NotImplementedError(f"Tracker {args.tracker} not implemented")

    # Logging
    logger.remove()
    logger.add("logs/detect_track.log", level="INFO")

    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(args.source, img_size=imgsz, stride=stride)
    else:
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadImages(args.source, img_size=imgsz, stride=stride)

    if len(dataset) > 1:
        cap = cv2.VideoWriter_fourcc(*'mp4v')
        #! NOTE: This is a hack to get the video size
        vid_writer = cv2.VideoWriter(args.save_vid, cap, 30, (640, 480))# get_ori_imsize(args.source))

    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    prev_frame = None

    for frame, (path, img, im0s, vid_cap) in enumerate(tqdm(dataset), start=1):
        img = preprocess(img, device, half)

        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=None)[0]

        # ------------------------ DETECTION ------------------------
        t1 = time_synchronized()
        with torch.no_grad():
            pred = model(img, augment=None)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, args.conf_thres, args.iou_thres, classes=0, agnostic=None)
        t3 = time_synchronized()

        logger.info(f"Detection Time: {t2 - t1:.3f}ms. NMS Time: {t3 - t2:.3f}ms")

        detect_runtime.update(t2 - t1)
        nms_runtime.update(t3 - t2)

        t4 = time_synchronized()

        logger.info(f"CMC Time: {t4 - t3:.3f}ms.")
        track_runtime.update(t4 - t3)
        # -----------------------------------------------------------

        trackers = []
        # Process detections and track
        for i, det in enumerate(pred, start=1):  # detections per image

            if len(det):
                if webcam:  # batch_size >= 1
                    p, s, im0s, frame = path[i - 1], '%g: ' % i, im0s[i - 1].copy(), dataset.count
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()
                mot_data = mot_converter(det, frame = i)
                dets = mot_data[:, 2:7] # [[x1, y1, w, h, conf]]
                if args.tracker == 'sort':
                    dets[: , 2:4] += dets[:, :2]
                    trackers = sort.update(dets)
                elif args.tracker == 'deepsort':
                    trackers = deepsort.update(dets, im0s)
                # trackers, unm_tr, unm_gt = sort_oh.update(dets, [])

                bb_plotter.draw_sort(im0s, trackers, id = True)
        
            if len(dataset) > 1 and args.output_result:
                vid_writer.write(im0s)

            if args.output_result:
                with open(out_file, 'a+') as f:
                    for d in trackers:
                        print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1'%(frame,d[4],d[0],d[1],d[2]-d[0],d[3]-d[1]),file=f)

    if len(dataset) > 1 and args.output_result:
        vid_writer.release()

    if args.output_result and args.pipe_result and args.benchmark:
        logger.info("Piping Result to MOTChallenge Benchmark")
        track_name = Path(Path(args.source).name).with_suffix('.txt')
        # track_name = static_01.txt
        tracker_dir = Path(f"{args.benchmark}-train") / "data" / "data" / track_name
        # tracker_dir = MyDataset-train/data/data/static_01.txt
        trackeval_trackdir = Path(args.trackeval_dir) / "data" / "trackers" / "mot_challenge" \
            / tracker_dir
        # trackeval_trackdir = ../TrackEval/data/trackers/mot_challenge/MyDataset-train/data/data/static_01.txt
        
        import shutil
        shutil.copy(out_file, trackeval_trackdir)

    logger.info(f"Average Detection Time: {detect_runtime.avg:.3f}ms")
    logger.info(f"Average NMS Time: {nms_runtime.avg:.3f}ms")
    logger.info(f"Average Tracking Time: {track_runtime.avg:.3f}ms")

    logger.success("Code is Complete, Check the logs for more info")