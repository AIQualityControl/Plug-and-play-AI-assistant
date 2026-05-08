#!/usr/bin/env python
# -*- encoding: utf-8 -*-


import os
import sys
import cv2
from common.model.measure_info import MeasureInfo

# Allow relative imports when being executed as script.
if __name__ == "__main__" and (__package__ is None or __package__ == ''):
    project_dir = os.path.join(os.path.dirname(__file__), '..', '..')
    sys.path.insert(0, project_dir)
    # import utility.sub_region_detector  # noqa: F401
    __package__ = "capture_core.biometry_measure"

# import torch
from ..biometry_measure.CerebMeasure import CerebMeasure
from ..biometry_measure.LVMeasure import LVMeasure
from common.model.CenaoMeasureInfo import CenaoMeasureInfo
from common.model.HcMeasureInfo import HcMeasureInfo
from common.model.EllipseAnnotation import EllipseAnnotation
from common.model.image_info import ImageInfo
from common.model.LineAnnotation import LineAnnotation
from common.model.XiaonaoMeasureInfo import XiaonaoMeasureInfo
from common.FetalBiometry import FetalBiometry
import torch
import numpy as np
import math
# from .Yolov8Model import Yolov8Model
# class LunaoMeasureModel(Yolov8Model):
# from .MaskRCNNModel import MaskRCNNModel
# from capture_core.measure_models.Yolov8Model import Yolov8Model
from capture_core.measure_models.SOLOV2Model import SOLOV2Model


class LunaoMeasureModel(SOLOV2Model):
    def __init__(self, model_file_name, class_mapping_file, config, load_model=True,
                 gpu_id=0, model_dir=r'/data/QC_python/model/'):
        '''constructor'''
        super(LunaoMeasureModel, self).__init__(model_file_name, class_mapping_file, config,
                                                load_model, gpu_id, model_dir)

    def measure_biometry(self, image_info: ImageInfo):
        info = super().measure_biometry(image_info)

        if image_info.anno_set is not None:
            image_info.anno_set.plane_type = self.plane_type

        return info

    def scale_image(self, masks, image_shape, ratio_pad=None):
        im1_shape = masks.shape
        if ratio_pad is None:  # calculate from im@ shape
            gain = min(im1_shape[0] / image_shape[0], im1_shape[1] / image_shape[1])
            pad = (im1_shape[1] - image_shape[1] * gain) / 2, (im1_shape[0] - image_shape[0] * gain) / 2
        else:
            gain = ratio_pad[0][0]
            pad = ratio_pad[1]
        top, left = int(pad[1]), int(pad[0])
        bottom, right = int(im1_shape[0] - pad[1]), int(im1_shape[1] - pad[0])
        masks = masks[top: bottom, left:right]
        masks = cv2.resize(masks, (image_shape[1], image_shape[0]), interpolation=cv2.INTER_NEAREST)
        return masks

    def clip_boxes(self, boxes, shape):
        """
        Takes a list of bounding boxes and a shape (height, width) and clips the bounding boxes to the shape.

        Args:
            boxes (torch.Tensor): the bounding boxes to clip
            shape (tuple): the shape of the image

        Returns:
            (torch.Tensor | numpy.ndarray): Clipped boxes
        """
        if isinstance(boxes, torch.Tensor):  # faster individually (WARNING: inplace .clamp_() Apple MPS bug)
            boxes[..., 0] = boxes[..., 0].clamp(0, shape[1])  # x1
            boxes[..., 1] = boxes[..., 1].clamp(0, shape[0])  # y1
            boxes[..., 2] = boxes[..., 2].clamp(0, shape[1])  # x2
            boxes[..., 3] = boxes[..., 3].clamp(0, shape[0])  # y2
        else:  # np.array (faster grouped)
            boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
            boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2
        return boxes

    def scale_boxes(self, img1_shape, boxes, img0_shape, ratio_pad=None, padding=True, xywh=False):
        """
        Rescales bounding boxes (in the format of xyxy by default) from the shape of the image they were originally
        specified in (img1_shape) to the shape of a different image (img0_shape).

        Args:
            img1_shape (tuple): The shape of the image that the bounding boxes are for, in the format of (height, width).
            boxes (torch.Tensor): the bounding boxes of the objects in the image, in the format of (x1, y1, x2, y2)
            img0_shape (tuple): the shape of the target image, in the format of (height, width).
            ratio_pad (tuple): a tuple of (ratio, pad) for scaling the boxes. If not provided, the ratio and pad will be
                calculated based on the size difference between the two images.
            padding (bool): If True, assuming the boxes is based on image augmented by yolo style. If False then do regular
                rescaling.
            xywh (bool): The box format is xywh or not, default=False.

        Returns:
            boxes (torch.Tensor): The scaled bounding boxes, in the format of (x1, y1, x2, y2)
        """
        if ratio_pad is None:  # calculate from img0_shape
            gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
            pad = (
                round((img1_shape[1] - img0_shape[1] * gain) / 2 - 0.1),
                round((img1_shape[0] - img0_shape[0] * gain) / 2 - 0.1),
            )  # wh padding
        else:
            gain = ratio_pad[0][0]
            pad = ratio_pad[1]

        if padding:
            boxes[..., 0] -= pad[0]  # x padding
            boxes[..., 1] -= pad[1]  # y padding
            if not xywh:
                boxes[..., 2] -= pad[0]  # x padding
                boxes[..., 3] -= pad[1]  # y padding
        boxes[..., :4] /= gain
        return self.clip_boxes(boxes, img0_shape)

    def find_contour_mask_intersection(self, mask, lugu_mask, distance_threshold=10):
        # Find contours
        mask = cv2.resize(mask, (int(mask.shape[1] / 10), int(mask.shape[0] / 10)))
        lugu_mask = cv2.resize(lugu_mask, (int(lugu_mask.shape[1] / 10), int(lugu_mask.shape[0] / 10)))
        contour1, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour2, _ = cv2.findContours(lugu_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Create masks for each contour
        mask1 = np.zeros_like(mask, dtype=np.uint8)
        mask2 = np.zeros_like(mask, dtype=np.uint8)

        cv2.drawContours(mask1, contour1, -1, (255))
        cv2.drawContours(mask2, contour2, -1, (255))
        # cv2.imshow('1111',mask1)
        # cv2.imshow('2222', mask2)
        # Find intersection between the two masks
        intersection = cv2.bitwise_and(mask1, mask2)

        # Find the coordinates of non-zero pixels in the intersection
        intersection_points = np.column_stack(np.where(intersection > 0))

        # Filter out points that are too close to each other
        filtered_points = []

        for point in intersection_points:
            if all(np.linalg.norm(np.array(point) - np.array(existing_point)) > distance_threshold for existing_point in
                   filtered_points):
                filtered_points.append(point)
        return filtered_points

    def compute_tidu_score(self, ptStart, ptEnd, base_image):
        ptStart, ptEnd = (ptStart, ptEnd) if ptStart[1] <= ptEnd[1] else (ptEnd, ptStart)
        image1 = base_image[ptStart[1] - 15:ptStart[1] + 15, ptStart[0] - 20:ptStart[0] + 20]
        image2 = base_image[ptEnd[1] - 15:ptEnd[1] + 15, ptEnd[0] - 20:ptEnd[0] + 20]
        image1, image2 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY), cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        ct_start, ct_end = np.mean(image1, axis=1), np.mean(image2, axis=1)
        grad_start, grad_end = np.diff(ct_start), np.diff(ct_end)
        differences_left = abs(np.sum(grad_start[grad_start > 0]))
        differences_right = abs(np.sum(grad_end[grad_end < 0]))
        differences_left, differences_right = round(differences_left / 150, 2), round(differences_right / 150, 2)
        clear_score = round((differences_left + differences_right) / 2 * 100, 2)
        return clear_score, differences_left, differences_right

    def compute_total_score(self, image_info, biometry_info, image, type2mask):
        score_dict = {
            "thalamic transverse plane": {
                "skull halo": 5,
                "cerebral falx": 5,
                "cerebral parenchyma": 5,
                "thalamic": 5,
                "third ventricle": 5,
                "posterior horn of lateral ventricle": 5,
                "anterior horn of lateral ventricle": 5,
                "choroid plexus": 5,
                "lateral cerebral sulcus": 5,
                "cavum septi pellucidi": 5,
                "septum pellucidum": 5,
                "column of fornix": 5,
                "corpus callosum": 5,
                "insular lobe": 5,
                "cerebellar hemisphere": -5,
                "cerebellar vermis": 5,
                "midbrain": 5,
                "cerebral aqueduct": 5,
                "posterior fossa cistern": 5
            },
            "lateral ventricular transverse plane": {
                "skull halo": 5,
                "cerebral falx": 5,
                "cerebral parenchyma": 5,
                "thalamic": 5,
                "third ventricle": 5,
                "posterior horn of lateral ventricle": 35,
                "anterior horn of lateral ventricle": 5,
                "choroid plexus": 5,
                "abnormal lateral cerebral sulcus": 5,
                "lateral cerebral sulcus": 5,
                "cavum septi pellucidi": 5,
                "septum pellucidum": 5,
                "column of fornix": 5,
                "corpus callosum": 5,
                "insular lobe": 5
            },
            "cerebellar transverse plane": {
                "skull halo": 5,
                "cerebral falx": 5,
                "cerebral parenchyma": 5,
                "thalamic": 5,
                "third ventricle": 5,
                "anterior horn of lateral ventricle": 5,
                "lateral cerebral sulcus": 5,
                "cavum septi pellucidi": 5,
                "septum pellucidum": 5,
                "column of fornix": 5,
                "corpus callosum": 5,
                "insular lobe": 5,
                "cerebellar hemisphere": 30,
                "cerebellar vermis": 5,
                "midbrain": 5,
                "cerebral aqueduct": 5,
                "posterior fossa cistern": 5
            }
        }

        total_score = {'lateral ventricular transverse plane': 135, 'cerebellar transverse plane': 170, 'thalamic transverse plane': 100}
        score_sum = 0
        intersection_list = []
        seg_score = {'structure_score': 0, 'tidu_score': 0, 'sector_score': 0, 'size_score': 0}

        for name, value in type2mask.items():
            if name == 'fan-shaped region':
                sector_mask = type2mask['fan-shaped region'][0]['mask']
                if "skull Halo" in type2mask:
                    hc_mask = type2mask["skull Halo"][0]['mask']
                    intersection_list = self.find_contour_mask_intersection(sector_mask, hc_mask)
                else:
                    score_sum -= 15

            for v in value:
                if name in score_dict[self.plane_type]:
                    score_sum += score_dict[self.plane_type][name] * float(v['score'])

        seg_score['structure_score'] = score_sum / total_score[self.plane_type] * 100
        seg_score['sector_score'] = -5 if len(intersection_list) > 0 else 5
        if seg_score['sector_score'] >= 0 and "skull Halo" in type2mask:
            lugh_box = type2mask["skull Halo"][0]['box']
            w = lugh_box[2] - lugh_box[0]
            h = lugh_box[3] - lugh_box[1]
            image_h, image_w = image_info.image.shape[:2]
            h_scale = h / image_h
            w_scale = w / image_w
            scale = (h_scale + w_scale) / 2
            original_min = 0.3
            original_max = 0.8
            target_min = 0
            target_max = 10
            slope = (target_max - target_min) / (original_max - original_min)
            intercept = target_min - slope * original_min

            seg_score['size_score'] = slope * scale + intercept

        measure_results = biometry_info
        base_image = image.copy()
        tcd_flag, lvw_flag, bpd_flag = False, False, False
        ptEnd, ptStart = None, None
        lvw_mean, tcd_mean = [], []
        if measure_results is not None:
            measure_results.update_ga()

        if self.plane_type == 'lateral ventricular transverse plane':
            lvw = round(measure_results.lvw, 2)
            lvw_flag = True
            lvw_mean.append(lvw)
            ptEnd, ptStart = [int(v) for v in measure_results.lvw_anno.ptEnd], [int(v) for v in
                                                                                measure_results.lvw_anno.ptStart]
        elif self.plane_type == 'cerebellar transverse plane':
            tcd = round(measure_results.tcd, 2)
            tcd_mean.append(tcd)

            tcd_flag = True
            ptEnd, ptStart = [int(v) for v in measure_results.tcd_anno.ptEnd], [int(v) for v in
                                                                                measure_results.tcd_anno.ptStart]
        elif self.plane_type == 'thalamic transverse plane':
            bpd_flag, hc_flag = True, True
            ptEnd, ptStart = ([int(v) for v in measure_results.intergrowth_21st_bpd_annotation.ptEnd],
                              [int(v) for v in measure_results.intergrowth_21st_bpd_annotation.ptStart])

        if ptStart and ptEnd:
            clear_score, differences_left, differences_right = self.compute_tidu_score(ptStart, ptEnd, base_image)
            seg_score['tidu_score'] = clear_score
            if image_info.anno_set is not None:
                detection_score = image_info.anno_set.score
            else:
                detection_score = image_info.auto_score
            seg_score['detection_score'] = detection_score
            if seg_score['tidu_score'] > 100:
                seg_score['tidu_score'] = 100
            total_score = round(
                seg_score['tidu_score'] * 0.20 + seg_score['structure_score'] * 0.3 + seg_score[
                    'detection_score'] * 0.5 + seg_score['sector_score'] + seg_score['size_score'], 2)

            distance = math.sqrt((ptEnd[0] - ptStart[0]) ** 2 + (ptEnd[1] - ptStart[1]) ** 2)
            if self.history_measure_queue is not None:
                q_len = len(self.history_measure_queue)
                curr_type = self.history_measure_queue[-1].auto_type if q_len else None
                good_count = 0
                for i in range(q_len - 1):
                    his = self.history_measure_queue[i]
                    if his.auto_type == curr_type:
                        if his is not None and hasattr(his, 'measure_results') and hasattr(his.measure_results,
                                                                                           'ptEnd'):
                            if self.plane_type == 'thalamic transverse plane':
                                ptEnd, ptStart = his.measure_results.ptEnd, his.measure_results.ptStart
                            elif self.plane_type == 'lateral ventricular transverse plane':
                                ptEnd, ptStart = his.measure_results.ptEnd, his.measure_results.ptStart
                            elif self.plane_type == 'cerebellar transverse plane':
                                ptEnd, ptStart = his.measure_results.ptEnd, his.measure_results.ptStart
                            old_distance = math.sqrt((ptEnd[0] - ptStart[0]) ** 2 + (ptEnd[1] - ptStart[1]) ** 2)
                            if lvw_flag is True and abs(old_distance - distance) < 15:
                                good_count = good_count + 1
                            if tcd_flag is True and abs(old_distance - distance) < 15:
                                good_count = good_count + 1
                            if bpd_flag is True and hc_flag is True and abs(old_distance - distance) < 15:
                                good_count = good_count + 1
                if q_len > 10 and good_count > q_len / 3:
                    total_score = total_score + 5
            if total_score > 95:
                total_score = 95
            # cv2.putText(image,
            #             f"{differences_left},{differences_right},{clear_score},{round(seg_score['structure_score'],2)},{total_score}",
            #             (100, 200), cv2.FONT_HERSHEY_SIMPLEX,
            #             fontScale=2, color=(0, 0, 255), thickness=2)
            # if lvw_flag:
            #     lvw_mean_num, lvw = sum(lvw_mean) / len(lvw_mean), round(lvw, 2)
            #     cv2.putText(image, f"{lvw_mean_num},{lvw}", (100, 150), cv2.FONT_HERSHEY_SIMPLEX,
            #                 fontScale=2, color=(0, 0, 255), thickness=2)
            # if tcd_flag:
            #     cv2.putText(image, f"{tcd_mean_num},{tcd}", (100, 150), cv2.FONT_HERSHEY_SIMPLEX,
            #                 fontScale=2, color=(0, 0, 255), thickness=2)
            # cv2.imshow('1', image)
            # cv2.waitKey(0)
            return total_score

    def do_measure(self, type2mask, roi_image, image_info):
        """
            args:
            type2mask - segmentation results
            roi_image - roi Image
            image_info - detection results

            return: MeasureInfo
        """
        # Get detection info for each structure {'name': 'cerebellum', 'vertex': [[...], [...]], 'score': 0.9404296875}
        detection_info = image_info.get_detection_info()

        # Use detection results if available; otherwise determine plane type from segmentation results
        if not detection_info:
            self.plane_type = self.plane_type_from_mask(type2mask)
        else:
            # Get detection results
            self.plane_type = image_info.anno_set.plane_type

        if hasattr(image_info, 'anno_set') and hasattr(image_info.anno_set,
                                                       'plane_type') and image_info.anno_set.plane_type:
            self.plane_type = image_info.anno_set.plane_type

        # start_time = time.time()
        for name, value in type2mask.items():
            for i, v in enumerate(type2mask[name]):
                if self.plane_type == 'lateral ventricular transverse plane':
                    if name in ['fan-shaped region', "skull Halo", "posterior horn of the lateral ventricle", "choroid plexus"]:
                        type2mask[name][i]['mask'] = self.scale_image(type2mask[name][i]['mask'], roi_image.shape)
                if self.plane_type == 'thalamic transverse plane':
                    if name in ['fan-shaped region', "skull Halo"]:
                        type2mask[name][i]['mask'] = self.scale_image(type2mask[name][i]['mask'], roi_image.shape)
                if self.plane_type == 'cerebellar transverse plane':
                    if name in ['fan-shaped region', "skull Halo", 'cerebellar hemisphere', 'third ventricle', 'cerebral aqueduct']:
                        type2mask[name][i]['mask'] = self.scale_image(type2mask[name][i]['mask'], roi_image.shape)

        # start_time = time.time()
        info = None
        if detection_info is not None:
            for anno in detection_info:
                if anno['name'] == 'abnormal lateral cerebral sulcus':
                    info = MeasureInfo()
                    info.disease_name_list = ['abnormal lateral cerebral sulcus']
        # if detection_info is None:
        #     return info
        if self.plane_type == 'lateral ventricular transverse plane':
            # Head circumference, biparietal diameter, lateral ventricle width
            hc, bpds, lvw = LVMeasure.do_measure(type2mask, roi_image, detection_info,
                                                 self.measure_mode, not FetalBiometry.is_hc_plane_detected)
            # print('segment lvw:', lvw)

            error_type = ''
            is_default_value = False
            if lvw is None:
                # No lateral ventricle structure from segmentation; estimate roughly using detection box
                lvw = self.default_lvw_measure(roi_image, image_info)
                error_type = 'LVW error'
                is_default_value = True

            lvw_anno = LineAnnotation(lvw[0], lvw[1], is_default_value=is_default_value) if lvw else None
            hc_anno = EllipseAnnotation(hc[0], hc[1], hc[2]) if hc else None

            intergrowth_21st_bpd_anno = LineAnnotation(bpds["intergrowth_21st"][0], bpds["intergrowth_21st"][
                1]) if bpds and "intergrowth_21st" in bpds else None

            hadlock_bpd_anno = LineAnnotation(
                bpds["hadlock"][0], bpds["hadlock"][1]) if bpds and "hadlock" in bpds else None

            if hc_anno or intergrowth_21st_bpd_anno or hadlock_bpd_anno or lvw_anno:
                info = CenaoMeasureInfo(hc_anno, intergrowth_21st_bpd_anno, hadlock_bpd_anno, lvw_anno=lvw_anno)
                info.error_type = error_type

        elif self.plane_type == 'cerebellar transverse plane':
            tcd = CerebMeasure.do_measure(type2mask, roi_image, detection_info)
            # print('segment tcd:', tcd)
            tcd_anno = LineAnnotation(tcd[0], tcd[1])
            info = XiaonaoMeasureInfo(tcd_anno=tcd_anno)
        elif self.plane_type == 'thalamic transverse plane':
            # Head circumference, biparietal diameter, lateral ventricle width

            hc, bpds, lvw = LVMeasure.do_measure(type2mask, roi_image, detection_info,
                                                 self.measure_mode, True, True)

            hc_anno = EllipseAnnotation(hc[0], hc[1], hc[2]) if hc else None

            intergrowth_21st_bpd_anno = LineAnnotation(bpds["intergrowth_21st"][0], bpds["intergrowth_21st"][
                1]) if bpds and "intergrowth_21st" in bpds else None

            hadlock_bpd_anno = LineAnnotation(
                bpds["hadlock"][0], bpds["hadlock"][1]) if bpds and "hadlock" in bpds else None

            info = HcMeasureInfo(hc_anno, intergrowth_21st_bpd_anno, hadlock_bpd_anno)
        else:
            print(f'No need to do measure for {self.plane_type}')

        return info

    def postprocess(self, boxes_batch, scores_batch, labels_batch, score_threshold=0.2, max_detections=10):
        boxes_result = []
        scores_result = []
        labels_result = []
        for boxes, scores, labels in zip(boxes_batch, scores_batch, labels_batch):
            if score_threshold > 0:
                # select indices which have a score above the threshold
                indices = np.where(scores > score_threshold)[0]

                # select those scores
                scores = scores[indices]

                # find the order with which to sort the scores
                scores_sort = np.argsort(-scores)[:max_detections]

                # select detections
                image_boxes = boxes[indices[scores_sort], :]
                image_scores = scores[scores_sort]
                image_labels = labels[indices[scores_sort]]
            else:
                # find the order with which to sort the scores
                scores_sort = np.argsort(-scores)[:max_detections]

                # select detections
                image_boxes = boxes[scores_sort, :]
                image_scores = scores[scores_sort]
                image_labels = labels[scores_sort]

            # unique except for BM has two pieces, if has Xiaonao, BM has one pieces at most
            if max_detections > 1:
                labels, scores, boxes = self.postprocess_special(image_labels, image_scores, image_boxes)

            else:
                labels = image_labels
                scores = image_scores
                boxes = image_boxes

            labels_result.append(labels)
            scores_result.append(scores)
            boxes_result.append(boxes)

        return boxes_result, scores_result, labels_result

    def plane_type_from_mask(self, type2mask):
        if 'cerebellar hemisphere' in type2mask or 'cerebellar vermis' in type2mask:
            plane_type = "cerebellar transverse plane"
        elif "posterior horn of the lateral ventricle" in type2mask:
            plane_type = "lateral ventricular transverse plane"
        elif 'thalamic' in type2mask:
            plane_type = "thalamic transverse plane"
        elif 'genu of corpus callosum' in type2mask or 'splenium of corpus callosum' in type2mask:
            plane_type = "cavum septi pellucidi horizontal transverse plane"
        else:
            plane_type = "cranial vault transverse plane"

        return plane_type

    def default_lvw_measure(self, roi_image, image_info):
        '''Estimate lateral ventricle width using detection results when segmentation is missing'''

        bbox = self.get_part_bbox(image_info, "posterior horn of the lateral ventricle")
        # bbox = None
        if bbox:
            x = int((bbox[0][0] + bbox[1][0]) * 0.5)
            h = abs(bbox[0][1] - bbox[1][1])
            pt_start = [x, int(bbox[0][1] + 0.2 * h)]
            pt_end = [x, int(bbox[1][1] - 0.2 * h)]
        else:
            height, width = roi_image.shape[:2]
            # No segmentation or detection results
            pt_start = [int(width * 0.5), int(height * 0.45)]
            pt_end = [int(width * 0.5), int(height * 0.55)]

        lvw_info = [pt_start, pt_end]
        return lvw_info

    def default_tcd_measure(self, roi_image, image_info):
        '''Estimate cerebellum diameter using detection results when segmentation is missing'''

        bbox = self.get_part_bbox(image_info, '小脑')
        # bbox = None
        if bbox:
            x = int((bbox[0][0] + bbox[1][0]) * 0.5)
            pt_start = [x, int(bbox[0][1])]
            pt_end = [x, int(bbox[1][1])]
        else:
            height, width = roi_image.shape[:2]

            # No segmentation or detection results
            pt_start = [int(width * 0.5), int(height * 0.4)]
            pt_end = [int(width * 0.5), int(height * 0.6)]

        cere_info = [pt_start, pt_end]
        return cere_info

    def diagnose_disease(self, biometry_info, image_info, image_type):
        diseas_list = biometry_info.disease_name_list

        if image_type == 'lateral ventricular transverse plane':
            lvw_length = biometry_info.lvw_anno.length() * biometry_info.ruler_unit
            if lvw_length > 1.0:
                diseas_list.append('Expansion - lateral ventricular transverse plane')
            elif lvw_length > 1.5:
                diseas_list.append('Hydrocephalus - lateral ventricular transverse plane')
        return diseas_list


