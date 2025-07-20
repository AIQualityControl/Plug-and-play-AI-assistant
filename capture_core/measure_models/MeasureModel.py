#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@description:
@date       : 2022/04/14 22:28:49
@author     : Guanghua Tan
@email      : guanghuatan@hnu.edu.cn
@version    : 1.0
'''

import json
import cv2
import os
from ..QcDetection.qc_models.QcModel import QcModel
from common.model.AnnotationSet import AnnotationSet
from PIL import Image
import numpy as np
from time import time
# from ..QcDetection.qc_models.OnnxModel import OnnxModel
from loguru import logger


class MeasureModel(QcModel):
    def __init__(self, model_file_name, class_mapping_file, config, load_model=True,
                 gpu_id=0, model_dir=r'/data/QC_python/model/'):
        '''constructor'''
        super(MeasureModel, self).__init__(model_file_name, class_mapping_file, config,
                                           load_model, gpu_id, model_dir)
        self.measure_mode = config.get('measure_mode', 'hadlock')

    @logger.catch
    def measure_biometry(self, image_info):
        """
        compute roi --> segmentation --> measurement
        """

        # 1. roi 取图片的扇形部分
        roi_image, offset = self.get_roi_image_offset(image_info)  # 获取大框的bbox内容和offset
        self.offset = offset

        # convert to gray
        # if len(roi_image.shape) > 2:
        #     gray_image = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
        # else:
        #     gray_image = roi_image

        # cv2.imshow('roi_image', roi_image)

        # 2. segmentation 对图片进行分割
        start = time()

        if roi_image is None or roi_image.shape[0] == 0 or roi_image.shape[1] == 0:
            mask = None
        elif self.model is None and image_info:
            mask = self.do_segment_with_image_info(roi_image, image_info)
        else:
            mask_list = self.do_segment([roi_image], [image_info])
            mask = mask_list[0] if mask_list else None

        logger.debug(f'>>>> segment time: {time() - start}')

        # ------------- test display -----------
        # if mask is not None:
        #     if isinstance(mask, dict):
        #         display_image = roi_image.copy()
        #         channel = 0
        #         for info_list in mask.values():
        #             for mask_info in info_list:
        #                 display_image = self.blend_with_bin_mask(display_image, mask_info['mask'],
        #                                                          channel, False)
        #             channel = (channel + 1) % 3
        #
        #     else:
        #         display_image = self.blend_with_mask(roi_image, mask)
        #
        #     # print(image_info)
        #     cv2.imshow('segment with mask', display_image)

        # --------------------------------------

        # 3. measurement
        # if mask is None:
        #     return

        start = time()

        info = self.do_measure(mask, roi_image, image_info)

        if info is not None:
            info.translate(offset)

        logger.debug(f'>>>> measure time: {time() - start}')

        return info

    def diagnose_disease(self, measure_info, image_info):
        return []

    def get_roi_image_offset(self, image_info):
        """
        return roi_image and offset
        """
        if self.is_detect_with_roi():

            bbox = self.compute_roi(image_info, enlarge_bbox=True)

            if image_info.detection_results is not None:
                image_info.detection_results['seg_roi'] = bbox
            elif image_info.anno_set is not None:
                image_info.anno_set.seg_roi = bbox

            # compute the roi
            roi_image = image_info.roi_image(bbox)
            offset = bbox[:2] if bbox is not None else image_info.offset()
        else:
            roi_image = image_info.roi_image()
            # roi左上角坐标
            offset = image_info.offset()

        return roi_image, offset

    @classmethod
    def compute_roi(cls, image_info, enlarge_bbox=True):
        """
        bounding box of all the annotations
        return bbox: [x, y, w, h]
        """
        annotation_list = image_info.detection_results['annotations'] \
            if image_info.detection_results is not None else image_info.anno_set
        if not annotation_list:
            return

        left, top, right, bottom = 10000, 10000, 0, 0

        if isinstance(annotation_list, AnnotationSet):
            annotation_list = annotation_list.annotations
            if not annotation_list:
                return
            for anno in annotation_list:
                if anno.name == '关键区域':
                    continue

                pt_start = anno.start_point()
                left = min(pt_start[0], left)
                top = min(pt_start[1], top)

                pt_end = anno.end_point()
                right = max(pt_end[0], right)
                bottom = max(pt_end[1], bottom)

            offset = image_info.offset()
            left -= offset[0]
            top -= offset[1]
            right -= offset[0]
            bottom -= offset[1]
        else:
            for anno in annotation_list:
                if isinstance(anno, str):
                    anno = json.loads(anno)
                # anno = ast.literal_eval(anno)
                vertex0, vertex1 = anno['vertex']

                # split by ,
                # v_list = vertex.split(',')

                # bbox
                left = min(float(vertex0[0]), left)
                top = min(float(vertex0[1]), top)
                right = max(float(vertex1[0]), right)
                bottom = max(float(vertex1[1]), bottom)

        bbox = [left, top, right - left, bottom - top]
        if enlarge_bbox:
            # enlarge bbox
            # width
            delta_w = bbox[2] // 10
            bbox[0] -= delta_w
            bbox[2] += delta_w * 2

            # height
            delta_h = bbox[3] // 10
            bbox[1] -= delta_h
            bbox[3] += delta_h * 2

        return bbox

    def detect(self, image_list, image_info):
        return self.do_segment(image_list, image_info)

    def do_segment(self, image_list, image_info_list):
        raise NotImplementedError('Please define a do_segment method')

    def do_segment_with_image_info(self, roi_image, image_info):
        if image_info.anno_set is None:
            return
        annotations = image_info.anno_set.get_polygon_annotations()

        type2mask = {}
        for anno in annotations:

            polygon = np.array(anno.points, np.int32)

            image_shape = image_info.image.shape[:2]
            mask = np.zeros(image_shape, np.uint8)

            cv2.fillPoly(mask, [polygon], (255,))

            h, w = roi_image.shape[:2]
            mask = mask[self.offset[1]: self.offset[1] + h, self.offset[0]:self.offset[0] + w]

            # x, y, w, h
            bbox = cv2.boundingRect(polygon)
            x = bbox[0] - self.offset[0]
            y = bbox[1] - self.offset[1]
            bbox = [max(x, 1), max(y, 1), min(x + bbox[2], w), min(y + bbox[3], h)]

            # draw on mask
            mask_info = {
                'mask': mask,
                'box': bbox,
                'score': 1.0,
                'polygon': None
            }
            if anno.name in type2mask:
                type2mask[anno.name].append(mask_info)
            else:
                type2mask[anno.name] = [mask_info]
        return type2mask if type2mask else None

    def do_measure(self, mask, roi_image):
        raise NotImplementedError('Please define a do_measure method')

    @classmethod
    def cv2_padding_resize(cls, img, base_size, padding_value=0):
        """
        return: (resized_image, padding_roi)
                 padding_roi is in format with xywh
        """
        start = time()

        ih, iw = img.shape[0:2]
        ew, eh = base_size
        scale = min(eh / ih, ew / iw)
        nh = int(ih * scale)
        nw = int(iw * scale)

        image = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)

        if isinstance(padding_value, list):
            padding_value = tuple(padding_value)
        elif not isinstance(padding_value, tuple):
            padding_value = (padding_value, padding_value, padding_value)

        top = (eh - nh) // 2
        bottom = eh - nh - top
        left = (ew - nw) // 2
        right = ew - nw - left

        new_img = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=padding_value)

        logger.debug(f'cv2_resize-time: {time() - start}')

        # cv2.imshow('resize', new_img)
        return new_img, (left, top, nw, nh)

    @classmethod
    def PIL_padding_resize(cls, image, size, padding_value=0):

        start = time()

        # time-consuming
        image = Image.fromarray(image)

        logger.debug(f'PIL_convert-time: {time() - start}')

        iw, ih = image.size
        w, h = size

        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)

        image = image.resize((nw, nh), Image.BILINEAR)

        if isinstance(padding_value, list):
            padding_value = tuple(padding_value)
        elif not isinstance(padding_value, tuple):
            padding_value = (padding_value, padding_value, padding_value)
        new_image = Image.new('RGB', size, padding_value)

        top = (h - nh) // 2
        left = (w - nw) // 2

        new_image.paste(image, (left, top))

        new_image = np.asarray(new_image)

        # cv2.imshow('resize', new_image)
        return new_image, (left, top, nw, nh)

    def get_cfg_path(self):
        cfg_path = os.path.join(self.model_dir, self.config['config_path'])
        if not os.path.exists(cfg_path):
            normal_cfg_path = cfg_path
            cfg_path = os.path.join(self.model_dir, 'detectron2', self.config['config_path'])
            if not os.path.exists(cfg_path):
                logger.warning('config path does not exist: ' + normal_cfg_path)
                return
        return cfg_path

    @classmethod
    def blend_with_mask(cls, image, mask, channel=2, copy_image=True):
        """
        blend image and mask with specified channel
        """
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif copy_image:
            image = image.copy()

        _, bin_mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
        bg_exclude_roi = cv2.bitwise_and(image, image, mask=cv2.bitwise_not(bin_mask))
        # cv2.imshow('bg exclude roi', bg_exclude_roi)

        # image with roi
        image[:, :, channel] = cv2.addWeighted(image[:, :, 2], 0.8, mask, 0.2, 0)
        weighted_roi = cv2.bitwise_and(image, image, mask=bin_mask)

        # cv2.imshow('weighted_roi', weighted_roi)

        image = cv2.add(bg_exclude_roi, weighted_roi)

        return image

    @classmethod
    def blend_with_bin_mask(cls, image, bin_mask, channel=2, copy_image=True):
        """
        blend image and mask with specified channel
        """
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif copy_image:
            image = image.copy()
        if image.shape[:2] != bin_mask.shape[:2]:
            bin_mask = QcModel.scale_mask(bin_mask, image.shape[:2])
        bg_exclude_roi = cv2.bitwise_and(image, image, mask=cv2.bitwise_not(bin_mask))
        # cv2.imshow('bg exclude roi', bg_exclude_roi)

        # image with roi
        image[:, :, channel] = cv2.addWeighted(image[:, :, 2], 0.8, bin_mask, 60, 0)
        weighted_roi = cv2.bitwise_and(image, image, mask=bin_mask)

        # cv2.imshow('weighted_roi', weighted_roi)

        image = cv2.add(bg_exclude_roi, weighted_roi)

        return image

    def get_part_bbox(self, image_info, part_name):
        """
        获取结构部位的bbox
        image_info:
        return bbox: [left_upper, right_bottom]
        """
        if part_name:
            bbox = image_info.get_part_bbox(part_name)
        else:
            bbox = self.compute_roi(image_info, False)
            if bbox:
                bbox = [[bbox[0], bbox[1]], [bbox[0] + bbox[2], bbox[1] + bbox[3]]]

        if bbox:
            offset = self.get_relative_offset(image_info)

            pt_start, pt_end = bbox
            pt_start = [pt_start[0] - offset[0], pt_start[1] - offset[1]]
            pt_end = [pt_end[0] - offset[0], pt_end[1] - offset[1]]

            bbox = [pt_start, pt_end]

        return bbox

    def get_part_bbox_list(self, image_info, part_name):
        """
        image_info:
        return bbox: [[left_upper, right_bottom]、......、[left_upper, right_bottom]]
        """
        bbox_list = []
        if part_name:
            bbox_list = image_info.get_part_bbox(part_name, only_one=False)
        else:
            bbox = self.compute_roi(image_info, False)
            if bbox:
                bbox = [[bbox[0], bbox[1]], [bbox[0] + bbox[2], bbox[1] + bbox[3]]]
                bbox_list = [bbox]

        if bbox_list:
            offset = self.get_relative_offset(image_info)
            for i, this_bbox in enumerate(bbox_list):
                pt_start, pt_end = this_bbox
                pt_start = [pt_start[0] - offset[0], pt_start[1] - offset[1]]
                pt_end = [pt_end[0] - offset[0], pt_end[1] - offset[1]]

                bbox_list[i] = [pt_start, pt_end]

        return bbox_list

    def get_relative_offset(self, image_info):
        """
        bbox像想对于ROI的offset
        """
        if self.is_detect_with_roi():
            offset = image_info.offset()  # 若为dicom，则返回：0,0
            # 返回
            return self.offset[0] - offset[0], self.offset[1] - offset[1]
        else:
            return 0, 0
