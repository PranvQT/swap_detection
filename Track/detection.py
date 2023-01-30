import json

import numpy as np
import torch
from models.experimental import attempt_load
from Utils.datasets import letterbox
from Utils.general import check_img_size, non_max_suppression, scale_coords
from Utils.torch_utils import select_device

imgsz = 1120
conf_thres = 0.66
device = ""
augment = False
iou_thres = 0.30
agnostic_nms = False
classes = [0,1,2]
frame_count = 0
with open("Settings/config.json", 'r') as _file:
    _data = json.load(_file)
    weights = _data['detection_weight']


with open("Settings/hyperparms.json", 'r') as _file:
    _data = json.load(_file)
    iou_thres = _data['iou']
    conf_thres = _data['confidence']


DET_DTYPE = np.dtype(
    [('tlbr', float, 4),
     ('label', int),
     ('conf', float)],
    align=True
)

device = select_device(device)
half = device.type != 'cpu'  # half precision only supported on CUDA
model = attempt_load(weights, map_location=device)  # load FP32 model
imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size

if half:
    model.half()
img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
_ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once


def detect(frame, conf_thres=conf_thres, iou_thres=iou_thres, classes=classes, agnostic_nms=agnostic_nms):
    """
    This function is used for detections on a given frame. 
    The detections are generated using yolov5 algorithm.

    Args:
        frame (np array): The current image on which detect has to be applied.

    Returns:
        detections (list): The output consisting of 4 coordinates for each detection in a 2d array.
    """

    img = letterbox(frame, new_shape=imgsz)[0]
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    pred = model(img, augment=augment)[0]

    # Apply NMS
    pred = non_max_suppression(
        pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)

    # Process detections
    for det in pred:  # detections per image
        if len(det):
            det[:, :4] = scale_coords(
                img.shape[2:], det[:, :4], frame.shape).round()
            detections = []
            # print(det)
            for *xyxy, conf, cls in reversed(det):
                x1 = int(xyxy[0])
                y1 = int(xyxy[1])
                x2 = int(xyxy[2])
                y2 = int(xyxy[3])

                detections.append((x1, y1, x2, y2,conf))
            return detections