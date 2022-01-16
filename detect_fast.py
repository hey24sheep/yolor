import torch
import torch.backends.cudnn as cudnn

from utils.datasets import LoadImages
from utils.general import (non_max_suppression, scale_coords, xyxy2xywh)
from utils.torch_utils import select_device

from models import *
from utils.datasets import *
from utils.general import *

class Detector:
    def __init__(self,weights, cfg, imgsz, device):
        self.model = self.load_model(weights, cfg, imgsz, device)
        return self.model

    def load_model(self, weights, cfg, imgsz, device):
        device = select_device(device)
        half = device.type != 'cpu'
        # Load model
        model = Darknet(cfg, imgsz).cuda()
        model.load_state_dict(torch.load(weights[0], map_location=device)['model'])
        #model = attempt_load(weights, map_location=device)  # load FP32 model
        #imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
        model.to(device).eval()
        if half:
            model.half()  # to FP16
        return model

    def detect_image_fast(self, model, img, imgsz, device, augment, conf_thres=0.40, iou_thres=0.50, classes=None, agnostic_nms=False):
        device = select_device(device)
        half = device.type != 'cpu'  # half precision only supported on CUDA
        pred_results = dict(
                imgids=[],
                boxes=[],
                scores=[],
                classes=[]
        )
        dataset = LoadImages(img, img_size=imgsz, auto_size=64)
        img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
        _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
        for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            pred = model(img, augment=augment)[0]
            # Apply NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                p, s, im0 = path, '', im0s
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                    # Write results
                    for *xyxy, conf, cls in det:
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        pred_results['imgids'].append(p)
                        pred_results['boxes'].append(xywh)
                        pred_results['scores'].append(conf)
                        pred_results['classes'].append(cls)
            return pred_results
