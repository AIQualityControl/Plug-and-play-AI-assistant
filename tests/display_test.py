#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@description:
@date       : 2023/08/04 15:21:37
@author     : Guanghua Tan
@email      : guanghuatan@hnu.edu.cn
@version    : 1.0
'''

import cv2
from loguru import logger
from PySide6.QtGui import QImage, QPixmap
import numpy as np

# Allow relative imports when being executed as script.
if __name__ == "__main__" and (__package__ is None or __package__ == ''):
    from pathlib import Path
    import sys

    project_dir = str(Path(__file__).parents[1])
    sys.path.insert(0, project_dir)
    # import utility.sub_region_detector  # noqa: F401
    __package__ = "pystdplane"

from common.render.VideoRenderWidget import VideoRenderWidget


def qtpixmap_to_cvimg(qtpixmap):

    # qimg = qtpixmap.toImage()
    # temp_shape = (qimg.height(), qimg.bytesPerLine() * 8 // qimg.depth())
    # temp_shape += (4,)
    # ptr = qimg.bits()
    # ptr.setsize(qimg.byteCount())
    # result = np.array(ptr, dtype=np.uint8).reshape(temp_shape)
    # result = result[..., :3]

    qimg = qtpixmap.toImage()
    result_shape = (qimg.height(), qimg.width(), 4)

    ptr = qimg.bits()
    # ptr.setsize(qimg.byteCount())
    result = np.array(ptr, dtype=np.uint8).reshape(result_shape)
    result = result[..., :3]

    return result


def cvimg_to_pixmap(image):
    height, width, channels = image.shape
    bytes_per_line = width * channels
    qImg = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)

    if qImg.isNull():
        logger.error('Failed to covert image from opencv to Qimage')
        return
    pixmap = QPixmap.fromImage(qImg.rgbSwapped())
    return pixmap


def draw_image_annotations(image, anno_set, frame_idx):
    if not image.data.contiguous:
        image = np.ascontiguousarray(image)

    cv2.putText(image, str(frame_idx), (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255))
    if anno_set is None:
        return image

    # convert to QPixmap
    pixmap = cvimg_to_pixmap(image)
    if pixmap is not None:
        pixmap = VideoRenderWidget.draw_offline_annotations(pixmap, anno_set, 'hadlock', False, show_ruler_info=True)

        # convert to cv2
        image = qtpixmap_to_cvimg(pixmap)

    return image
