
import yaml
import re
import cv2
from pathlib import Path
import math
import numpy as np
import os
import sys

# Allow relative imports when being executed as script.
if __name__ == "__main__" and (__package__ is None or __package__ == ''):
    pkg_path = os.path.join(os.path.dirname(__file__), '..', '..')
    sys.path.insert(0, pkg_path)

    # print(sys.path)
    # import utility.sub_region_detector  # noqa: F401
    __package__ = "QcDetection.yoloair"

from yoloair.utils.datasets import LoadImagesAndLabels


# Other Constants
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLO
ASSETS = ROOT / 'assets'  # default images
DEFAULT_CFG_PATH = ROOT / 'data/hyps/hyp.scratch-med.yaml'


def yaml_load(file='data.yaml', append_filename=False):
    """
    Load YAML data from a file.

    Args:
        file (str, optional): File name. Default is 'data.yaml'.
        append_filename (bool): Add the YAML filename to the YAML dictionary. Default is False.

    Returns:
        (dict): YAML data and file name.
    """
    with open(file, errors='ignore', encoding='utf-8') as f:
        s = f.read()  # string

        # Remove special characters
        if not s.isprintable():
            s = re.sub(r'[^\x09\x0A\x0D\x20-\x7E\x85\xA0-\uD7FF\uE000-\uFFFD\U00010000-\U0010ffff]+', '', s)

        # Add YAML filename to dict and return
        data = yaml.safe_load(s) or {}  # always return a dict (yaml.safe_load() may return None for empty files)
        if append_filename:
            data['yaml_file'] = str(file)
        return data


DEFAULT_CFG_DICT = yaml_load(DEFAULT_CFG_PATH)


def test_with_dataset(data_path):

    data = yaml_load(data_path)

    # cfg = get_cfg(DEFAULT_CFG, {'mosaic': 0.0, 'mixup': 0.0, 'overlap_mask': False, 'task': 'detect'})
    cfg = DEFAULT_CFG_DICT

    batch_size = 1
    img_path = data['path']
    # dataset = build_yolo_dataset(cfg=cfg, img_path=img_path, batch=batch_size, data=data)
    # dataset = build_my_dataset(cfg=cfg, img_path=img_path, batch=batch_size, data=data)
    dataset = LoadImagesAndLabels(img_path, batch_size=batch_size, augment=True, hyp=cfg)

    for i in range(len(dataset)):
        orig_image, _, _ = dataset.load_image(i)
        cv2.imshow('original image', orig_image)

        image_info = dataset[i]

        image = image_info[0]
        bboxes = image_info[1][..., 2:]

        # names = [data['names'][int(cls)] for cls in image_info['cls']]

        # segments = image_info['segments'] if 'segments' in image_info else None
        # masks = image_info['masks'] if 'masks' in image_info else None

        show_image(image, bboxes)
        cv2.waitKey()

    # data_loader = build_dataloader(dataset, batch=batch_size, workers=1, shuffle=True)
    # for data in data_loader:
    #     # bboxes are stacked together
    #     images, bboxes = data['img'], data['bboxes']
    #     for image in images:
    #         show_image(image, bboxes)
    #         cv2.waitKey()


def show_image(image, bboxes, segments=None, masks=None, names=None):
    # convert to numpy
    if not isinstance(image, np.ndarray):
        image = image.permute(1, 2, 0).numpy()
        if not image.data.contiguous:
            image = np.ascontiguousarray(image)

    if bboxes is not None and len(bboxes) > 0:
        h, w = image.shape[:2]
        if not isinstance(bboxes, np.ndarray):
            bboxes = bboxes.numpy()
            if bboxes.ndim == 1:
                bboxes = [bboxes]
        if not names:
            names = [None] * len(bboxes)
        for bbox, name in zip(bboxes, names):
            pt0 = [int((bbox[0] - bbox[2] / 2) * w), int((bbox[1] - bbox[3] / 2) * h)]
            pt1 = [int((bbox[0] + bbox[2] / 2) * w), int((bbox[1] + bbox[3] / 2) * h)]

            cv2.rectangle(image, pt0, pt1, (0, 0, 255), lineType=cv2.LINE_AA)

            if name:
                cv2.putText(image, name, pt0, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255))

    if segments is not None and len(segments) > 0:
        h, w = image.shape[:2]
        segments[..., 0] *= w
        segments[..., 1] *= h

        polygons = [np.array(seg, dtype=np.int32) for seg in segments]

        cv2.polylines(image, polygons, True, (0, 0, 255))

    cv2.imshow('augmented image', image)

    if masks is not None and len(masks) > 0:
        masks = masks.numpy()
        if len(masks) > 1:
            masks *= 255
        else:
            ninstance = max(len(bboxes), 1)
            masks *= int(255 / ninstance)

        for i in range(len(masks)):
            cv2.imshow(f'mask{i}', masks[i])


def load_image(image_path, imgsz):
    image = cv2.imdecode(np.fromfile(image_path), flags=cv2.IMREAD_COLOR)
    if image is None:
        return

    h0, w0 = image.shape[:2]  # orig hw
    r = imgsz / max(h0, w0)  # ratio
    if r != 1:  # if sizes are not equal
        interp = cv2.INTER_LINEAR if r > 1 else cv2.INTER_AREA
        image = cv2.resize(image, (min(math.ceil(w0 * r), imgsz), min(math.ceil(h0 * r), imgsz)),
                           interpolation=interp)

    return image


if __name__ == '__main__':

    data_path = ROOT / r'data\nt.yaml'

    # test_with_transform(data_path)
    test_with_dataset(data_path)
