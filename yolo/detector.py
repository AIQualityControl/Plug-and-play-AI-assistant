import logging

import numpy as np
import torch
import cv2

# from models.experimental import attempt_load
from .utils.augmentations import letterbox
from .models.common import DetectMultiBackend
from .utils.general import (check_img_size,
                            non_max_suppression, scale_coords)
from .utils.torch_utils import select_device
from ultralytics import YOLO


class YOLOv5(object):
    '''
    def __init__(self, weights, img_size, device, augment, classes, agnost ic_nms, namesfile,
    is_xywh=False, iou_thresh=0.5, conf_thresh=0.4):
    '''

    def __init__(self, weights='', img_size=640, device='0', iou_thresh=0.25, conf_thresh=0.25, stride=32,
                 agnostic_nms=False, auto=True):
        # Initialize
        self.device = select_device(device)
        self.img_size = check_img_size(img_size, s=stride)
        self.auto = auto
        self.augment = False
        self.iou_thresh = iou_thresh
        self.conf_thresh = conf_thresh
        self.classes = None
        self.agnostic_nms = agnostic_nms
        model = DetectMultiBackend(weights, device=self.device)
        stride, _, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
        self.stride = stride
        self.half = (pt or jit or onnx or engine) and self.device.type != 'cpu'
        if pt or jit:
            model.model.half() if self.half else model.model.float()
        self.model = model
        logger = logging.getLogger("root.detector")
        logger.info('Loading weights from %s... Done!' % (weights))
        # no need to do warmup for each model
        # model.warmup(imgsz=(1, 3, *img_size), half=self.half)  # warmup

    def _xyxy_to_xywh(self, bbox_xyxy):
        x1, y1, x2, y2 = bbox_xyxy
        x = (x1 + x2) / 2
        y = (y1 + y2) / 2
        w = (x2 - x1)
        h = (y2 - y1)
        return x, y, w, h

    def preprocess(self, img0):
        # Padded resize
        img = letterbox(img0, self.img_size, stride=self.stride, auto=self.auto)[0]

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        return img

    @torch.no_grad()
    def predict_on_batch(self, img0):
        img_batch = []
        for img in img0:
            img = self.preprocess(img)
            img_batch.append(img)

        # # img = torch.zeros((1, 3, self.imgsz, self.imgsz), device=self.device)  # init img
        img = np.array(img_batch)
        img = torch.from_numpy(img).to(self.device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = self.model(img, augment=self.augment, val=True)
        # print("pred=",pred)
        # Apply NMS
        pred = non_max_suppression(pred, self.conf_thresh, self.iou_thresh, classes=self.classes,
                                   agnostic=self.agnostic_nms, max_det=15)

        boxes_batch = []
        confs_batch = []
        clsids_batch = []
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            im0 = img0[i]
            # gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  #  normalization gain whwh
            if det is not None and len(det):
                # print('det:',det)
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                boxes = det[:, :4]

                # print('boxex:',boxes)
                boxes = boxes.cpu().numpy()
                for i in range(len(boxes)):
                    boxes[i] = self._xyxy_to_xywh(boxes[i])
                boxes_batch.append(boxes)
                confs = det[:, 4:5]
                confs = confs.cpu().numpy()
                confs_batch.append(confs)
                clsids = det[:, -1:]
                clsids = clsids.cpu().numpy()
                clsids_batch.append(clsids)
            else:
                boxes = [[0, 0, 0, 0]]
                confs = [[0.01]]
                clsids = [[0]]
                boxes_batch.append(np.array(boxes))
                confs_batch.append(np.array(confs))
                clsids_batch.append(np.array(clsids))
        boxes_batch = np.array(boxes_batch)
        confs_batch = np.array(confs_batch)
        clsids_batch = np.array(clsids_batch)
        return boxes_batch, confs_batch, clsids_batch

    @torch.no_grad()
    def predict(self, img0):

        boxes_batch = []
        confs_batch = []
        clsids_batch = []
        for noo, img in enumerate(img0):
            img = self.preprocess(img)

            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inferences
            pred = self.model(img, augment=self.augment)

            # Apply NMS
            pred = non_max_suppression(pred, self.conf_thresh, self.iou_thresh,
                                       classes=self.classes, agnostic=self.agnostic_nms, max_det=15)

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                im0 = img0[noo]
                # gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  #  normalization gain whwh
                if det is not None and len(det):

                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    boxes = det[:, :4]
                    boxes = boxes.double().cpu().numpy()
                    boxes_batch.append(boxes)
                    confs = det[:, 4]
                    confs = confs.double().cpu().numpy()
                    confs_batch.append(confs)
                    clsids = det[:, -1]
                    clsids = clsids.int().cpu().numpy()
                    clsids_batch.append(clsids)
                else:
                    boxes = [[0, 0, 0, 0]]
                    confs = [0.01]
                    clsids = [0]
                    boxes_batch.append(np.array(boxes, dtype='float64'))
                    confs_batch.append(np.array(confs, dtype='float64'))
                    clsids_batch.append(np.array(clsids, dtype='int32'))
        boxes_batch = np.array(boxes_batch, dtype=object)
        confs_batch = np.array(confs_batch, dtype=object)
        clsids_batch = np.array(clsids_batch, dtype=object)

        np.set_printoptions(threshold=np.inf, precision=4, suppress=True)


        return boxes_batch, confs_batch, clsids_batch


# Restore the segmented mask to the original image size
def scale_image(masks, im0_shape, ratio_pad=None):
    """
    Takes a mask, and resizes it to the original image size

    Args:
      masks (torch.Tensor): resized and padded masks/images, [h, w, num]/[h, w, 3].
      im0_shape (tuple): the original image shape
      ratio_pad (tuple): the ratio of the padding to the original image.

    Returns:
      masks (torch.Tensor): The masks that are being returned.
    """
    # Rescale coordinates (xyxy) from im1_shape to im0_shape
    im1_shape = masks.shape
    if im1_shape[:2] == im0_shape[:2]:
        return masks
    if ratio_pad is None:  # calculate from im0_shape
        gain = min(im1_shape[0] / im0_shape[0], im1_shape[1] / im0_shape[1])  # gain  = old / new
        pad = (im1_shape[1] - im0_shape[1] * gain) / 2, (im1_shape[0] - im0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]
    top, left = int(pad[1]), int(pad[0])  # y, x
    bottom, right = int(im1_shape[0] - pad[1]), int(im1_shape[1] - pad[0])

    if len(masks.shape) < 2:
        raise ValueError(f'"len of masks shape" should be 2 or 3, but got {len(masks.shape)}')
    masks = masks[top:bottom, left:right]

    masks = cv2.resize(masks, (im0_shape[1], im0_shape[0]))
    if len(masks.shape) == 2:
        masks = np.expand_dims(masks, 2)

    return masks


class YOLOv8(object):
    def __init__(self, weights='', img_size=640, device='0', iou_thresh=0.25, conf_thresh=0.25):
        """
        Initializes the YOLO object detector with the given parameters.

        :param weights: Path to the file containing the weights of the YOLO model.
        :type weights: str

        :param img_size: The size of the input image that the YOLO model expects.
        :type img_size: int

        :param device: The id of the GPU to use for running the YOLO model.
        :type device: str

        :param iou_thresh: The IoU threshold for non-maximum suppression.
        :type iou_thresh: float

        :param conf_thresh: The confidence threshold for object detection.
        :type conf_thresh: float

        :param stride: The stride of the YOLO model.
        :type stride: int

        :param agnostic_nms: Whether to use agnostic non-maximum suppression.
        :type agnostic_nms: bool

        :param auto: Whether to automatically download the YOLO weights if not found.
        :type auto: bool
        """
        # Initialize
        self.model = YOLO(weights)
        self.img_size = img_size
        self.iou_thresh = iou_thresh
        self.conf_thresh = conf_thresh
        self.device = device

    @torch.no_grad()
    def predict(self, img0):
        """
        Predicts bounding boxes, confidence scores, and class IDs for a batch of images.

        Args:
            img0 (list): List of ndarray images.

        Returns:
            tuple: A tuple of three numpy arrays containing the predicted bounding boxes,
            confidence scores, and class IDs respectively for all images in the batch.
        """
        img_batch = [img for img in img0]

        preds = self.model(img_batch, imgsz=self.img_size,
                           conf=self.conf_thresh,
                           iou=self.iou_thresh,
                           device=self.device)
        boxes_batch = []
        confs_batch = []
        clsids_batch = []
        quaids_batch = []
        for pred in preds:
            boxes_batch.append(pred.boxes.xyxy.cpu().numpy())
            confs_batch.append(pred.boxes.conf.cpu().numpy())
            clsids_batch.append(pred.boxes.cls.long().cpu().numpy())
        return boxes_batch, confs_batch, clsids_batch

    @torch.no_grad()
    def predict_with_mask(self, img_list):
        """
        Predicts bounding boxes, confidence scores, and class IDs for a batch of images.

        Args:
            img0 (list): List of ndarray images.

        Returns:
            tuple: A tuple of three numpy arrays containing the predicted bounding boxes,
            confidence scores, and class IDs respectively for all images in the batch.
        """
        if not isinstance(img_list, list):
            img_batch = [img for img in img_list]
        else:
            img_batch = img_list

        preds = self.model(img_batch, imgsz=self.img_size,
                           conf=self.conf_thresh,
                           iou=self.iou_thresh,
                           device=self.device)
        boxes_batch = []
        confs_batch = []
        clsids_batch = []
        mask_batch = []
        for pred, image in zip(preds, img_batch):
            boxes_batch.append(pred.boxes.xyxy.cpu().numpy())
            confs_batch.append(pred.boxes.conf.cpu().numpy())
            clsids_batch.append(pred.boxes.cls.long().cpu().numpy())
            if pred.masks is not None:
                mask = pred.masks.data.cpu().numpy()

                # a = np.where(0 < mask < 1)
                # whether has to multiply with 255
                mask *= 255
                mask = mask.astype(np.uint8)

                mask_batch.append(mask)

        return boxes_batch, confs_batch, clsids_batch, mask_batch

    @torch.no_grad()
    def predict_track_with_mask(self, img_list):
        """
        Predicts bounding boxes, confidence scores, and class IDs for a batch of images.

        Args:
            img0 (list): List of ndarray images.

        Returns:
            tuple: A tuple of three numpy arrays containing the predicted bounding boxes,
            confidence scores, and class IDs respectively for all images in the batch.
        """
        if not isinstance(img_list, list):
            img_batch = [img for img in img_list]
        else:
            img_batch = img_list

        preds = self.model.track(img_batch, imgsz=self.img_size,
                                 conf=self.conf_thresh,
                                 iou=self.iou_thresh,
                                 device=self.device,
                                 persist=True)
        boxes_batch = []
        confs_batch = []
        clsids_batch = []
        mask_batch = []
        trackid_batch = []
        for pred, image in zip(preds, img_batch):
            boxes_batch.append(pred.boxes.xyxy.cpu().numpy())
            confs_batch.append(pred.boxes.conf.cpu().numpy())
            clsids_batch.append(pred.boxes.cls.long().cpu().numpy())
            if pred.boxes.id is not None:
                trackid_batch.append(pred.boxes.id.long().cpu().numpy())
            else:
                trackid_batch.append(np.array([]))
            if pred.masks is not None:
                mask = pred.masks.data.cpu().numpy()

                # a = np.where(0 < mask < 1)
                # whether has to multiply with 255
                mask *= 255
                mask = mask.astype(np.uint8)

                mask_batch.append(mask)

        return boxes_batch, confs_batch, clsids_batch, trackid_batch, mask_batch

