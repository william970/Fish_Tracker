import argparse
import numpy
import cv2
import torch
import os

from numpy import random
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path
from utils.datasets import letterbox
from utils.torch_utils import select_device, load_classifier, time_synchronized
import sys


class Detector(object):
    def __init__(self, weights, augment, img_sz=640, conf_thres=0.15, iou_thres=0.45, classes=None,agnostic_nms=None, use_cuda=''):
        self.device = select_device(use_cuda)
        # 浮点16位 只有CUDA支持
        self.half = self.device.type != 'cpu'
        # 定义网咯
        self.weights = weights
        # 加载模型
        self.model = attempt_load(weights, map_location=self.device)  # 加载浮点32位模型
        self.img_sz = check_img_size(img_sz, s=self.model.stride.max())  # 输入图片size
        if self.half:
            self.model.half()  # to FP16
        # 得到类名和颜色
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]

        self.augment = augment
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.classes = classes
        self.agnostic_nms = agnostic_nms

    def __call__(self, ori_img):
        img = letterbox(ori_img, self.img_sz)[0]
        # BGR 转 RGB, 同时转置维度轴
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = numpy.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        # 前向传播
        pred = self.model(img, augment=self.augment)[0]
        # NMS去重
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=self.classes, agnostic=self.agnostic_nms)
        box = []
        cls_ids = []
        cls_conf = []
        for i, det in enumerate(pred):  # detections per image
            #
            # save_path = str(save_dir / p.name)
            # txt_path = str(save_dir / 'labels' / p.stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
            gn = torch.tensor(ori_img.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], ori_img.shape).round()

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()  # normalized xywh
                    box.append(xywh)
                    cls_ids.append(float(cls))
                    cls_conf.append(float(conf))
        return numpy.array(box), numpy.array(cls_ids), numpy.array(cls_conf)

def plot_one_box2(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]-x[2]/2), int(x[1]-x[3]/2)), (int(x[0]+x[2]/2), int(x[1]+x[3]/2))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--weights', nargs='+', type=str, default='yolov3.pt', help='model.pt path(s)')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--conf-thres', type=float, default=0.2, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    imgsz = 640
    opt = parser.parse_args()
    detector =Detector(opt.weights, augment=opt.augment, img_sz = imgsz, conf_thres = opt.conf_thres, iou_thres=opt.iou_thres,
                       classes=opt.classes, agnostic_nms=opt.agnostic_nms, use_cuda = opt.device)

    cap = cv2.VideoCapture("I:/track/jupyter/2020-12-14_10_47_15.mp4")
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    while True:
        success, frame = cap.read()

        if success:

            box, cls_ids, cls_conf = detector(frame)

            for i in range(len(cls_ids)):
                label = '%s %.2f' % (detector.names[int(cls_ids[i])], cls_conf[i])
                plot_one_box2(box[i], frame, label=label, color=detector.colors[int(cls_ids[i])], line_thickness=3)
            cv2.imshow("fish", frame)
            if cv2.waitKey(1) == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()


