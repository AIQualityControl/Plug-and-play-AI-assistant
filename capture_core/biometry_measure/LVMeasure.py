#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@description:
@date       : 2022/04/22 00:00:18
@author     : Guanghua Tan
@email      : guanghuatan@hnu.edu.cn
@version    : 1.0
'''

from .EllipseFitting import EllipseFitting
from .BPDMeasure import BPDMeasure
from common.model.EllipseAnnotation import EllipseAnnotation
from ..QcDetection.utility import math_util, draw_util
from . import measure_util
import time

import numpy as np
import cv2

key_point_offset_ratio = 0.2
binary_threshold = 85
wrong_ratio = 0.4

_FOR_DEBUG_ = False


class LVMeasure:
    """
    measurement for Lateral Ventricle (LV)
    """

    def __init__(self):
        pass

    @classmethod
    def do_measure(cls, type2mask, image, detection_info, measure_mode='hadlock', measure_hc_bpd=True, qn_flag=False):

        hc_info, bpd_infos = None, None
        if measure_hc_bpd:
            start = time.time()

            hc_info, bpd_infos = cls.hc_and_bpd(type2mask, image, measure_mode)

            print('hc and bpd time: ', time.time() - start)
            if qn_flag:
                return hc_info, bpd_infos, None

        if '侧脑室后角' not in type2mask:
            return hc_info, bpd_infos, None

        start = time.time()

        # detection_info = image_info.get_detection_info()

        cns_contours = measure_util.match_detection_info(type2mask, detection_info, '侧脑室后角', smooth_ksize=15)
        if not cns_contours:
            return hc_info, bpd_infos, None
        cns_contour = cns_contours[0]

        mlc_contour = None
        if '脉络丛' in type2mask:
            mlc_contours = measure_util.match_detection_info(type2mask, detection_info, '脉络丛', smooth_ksize=15)
            if mlc_contours and len(mlc_contours) == 1:
                mlc_contour = mlc_contours[0]

        # print('match time: ', time.time() - start)

        # start = time.time()
        lvw_info = cls.lv_width_area_perimeter(cns_contour, mlc_contour, image)

        if _FOR_DEBUG_:
            print('lv time: ', time.time() - start)

            display_image = draw_util.draw_contours(image, [cns_contour, mlc_contour])
            display_image = draw_util.draw_lineseg(display_image, lvw_info)
            cv2.imshow('mlc/cns contours', display_image)
            cv2.waitKey(0)

        return hc_info, bpd_infos, lvw_info

    @classmethod
    def hc_and_bpd(cls, type2mask, image, measure_mode='hadlock'):
        #
        if '颅骨光环' not in type2mask or len(type2mask['颅骨光环']) != 1:
            return None, None

        # 只需要一个颅骨光环实例
        mask = type2mask['颅骨光环'][0]['mask']

        # todo: 颅骨光环的分割结果有断裂: 需要合并两个contour
        contours = measure_util.max_n_contours_of_mask(mask, 2, 'num_points')

        # 如果只有一个轮廓，直接进行椭圆拟合
        if len(contours) == 0:
            return None, None

        bndry_pnts = cls.remove_nonconvex_points(contours)

        hc_info = EllipseFitting.fit_ellipse_by_points(bndry_pnts)

        if _FOR_DEBUG_:
            display_image = draw_util.draw_contours(image, contours)
            if hc_info:
                draw_util.draw_ellipse(display_image, hc_info, inplace=True)
            else:
                cv2.imshow('lggh', display_image)

        if not hc_info:
            return None, None

        # bpd
        # convert to gray
        if len(image.shape) > 2 and image.shape[2] > 1:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image

        hc_anno = EllipseAnnotation(hc_info[0], hc_info[1], hc_info[2])
        end_points = hc_anno.minor_radius_points()
        bpd_infos = BPDMeasure.do_measure(gray_image, mask, end_points, measure_mode, is_bin_mask=True)

        # if _FOR_DEBUG_:
        #     draw_util.draw_lineseg(display_image, bpd_info, (0, 255, 255))
        #     cv2.imshow('hc and bpd', display_image)

        return hc_info, bpd_infos

    @classmethod
    def remove_nonconvex_points(cls, contours):
        # 计算每个连通域的凸包
        convex_hulls = []
        contours_2D = []
        for contour in contours:
            if len(contour) < 3:
                continue
            convex_hull = cv2.convexHull(contour)
            convex_hulls.append(convex_hull)
            contours_2D.extend(contour.tolist())
        # 合并所有连通域的凸包
        merged_convex_hull = cv2.convexHull(np.concatenate(convex_hulls))
        # contours = np.squeeze(contours)
        new_contour = []
        for point in contours_2D:
            # int_point = [int(x) for x in point]
            if cv2.pointPolygonTest(merged_convex_hull, point, True) <= 3:
                new_contour.append(point)
        new_contour = np.squeeze(new_contour)
        return new_contour

    @classmethod
    def lv_width_area_perimeter(cls, cns_contour, mlc_contour, image):

        # 脉络丛不存在，则以侧脑室后角的外接矩形的中垂线定位
        if mlc_contour is None:
            line = cls.get_line_from_contour_and_point(cns_contour, None)
        # 脉络丛存在，则以脉络丛为定位结构
        else:
            mlcMax_xy = np.argmax(mlc_contour, axis=0)
            mlcMax_x = mlc_contour[mlcMax_xy[0]]
            mlcMin_xy = np.argmin(mlc_contour, axis=0)
            mlcMin_x = mlc_contour[mlcMin_xy[0]]

            cnsMax_xy = np.argmax(cns_contour, axis=0)
            cnsMax_x = cns_contour[cnsMax_xy[0]]
            cnsMin_xy = np.argmin(cns_contour, axis=0)
            cnsMin_x = cns_contour[cnsMin_xy[0]]

            # 判断"脉络丛"在侧脑室后角的左侧还是右侧
            if (abs(cnsMax_x[0] - mlcMax_x[0]) > abs(cnsMin_x[0] - mlcMin_x[0])):
                mlc_point = [mlcMax_x[0] + key_point_offset_ratio * (cnsMax_x[0] - mlcMax_x[0]),
                             mlcMax_x[1] + key_point_offset_ratio * (cnsMax_x[1] - mlcMax_x[1])]  # 脉络丛在左侧
            else:
                mlc_point = [mlcMin_x[0] + key_point_offset_ratio * (cnsMin_x[0] - mlcMin_x[0]),
                             mlcMin_x[1] + key_point_offset_ratio * (cnsMin_x[1] - mlcMin_x[1])]  # 脉络丛在右侧

            line = cls.get_line_from_contour_and_point(cns_contour, mlc_point)

        points_cns = math_util.line_intersect_with_polygon(line, cns_contour, keep_two_max=True)
        if not points_cns:
            return

        cns_mask = np.zeros(image.shape[:2], np.uint8)
        # 填充特征区域
        cv2.fillPoly(cns_mask, np.array([cns_contour], dtype=np.int32), (255,))

        end_points = cls.refine_end_points(image, cns_mask, points_cns, line)

        return end_points

    @classmethod
    def get_line_from_contour_and_point(cls, contour, point):
        center, (w, h), angle = cv2.minAreaRect(contour)
        # box = cv2.boxPoints(rect)

        # get the shortest edge
        # w, h = rect[1]
        if w > h:
            dir = math_util.rotate_vec([0, 1], angle)
        else:
            dir = math_util.rotate_vec([1, 0], angle)

        # box = np.int0(box)
        # box = cls.order_points(box)

        # midpoint1 = math_util.mid_point(box[0], box[1])
        # midpoint2 = math_util.mid_point(box[2], box[3])

        if point is None:
            point = center

        return point, dir

    @classmethod
    def refine_end_points(cls, image, mask, end_points, line):
        if end_points[0][1] > end_points[1][1]:
            upper_point = end_points[1]
            lower_point = end_points[0]
        else:
            upper_point = end_points[0]
            lower_point = end_points[1]
        x0, y0 = upper_point
        x1, y1 = lower_point

        height, width = image.shape[:2]
        if len(image.shape) > 2:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image

        if not (0 < y0 < height and 0 < x0 < width and 0 < y1 < height and 0 < x1 < width):
            return end_points

        # roi: 左上点x, 左上y, 水平宽，垂直高
        upper_roi = BPDMeasure.get_roi_image(upper_point, [width, height], 20, True)
        lower_roi = BPDMeasure.get_roi_image(lower_point, [width, height], 20, False)
        roi = [upper_roi[0], upper_roi[1], abs((lower_roi[0] + lower_roi[2]) - upper_roi[0]),
               abs((lower_roi[1] + lower_roi[3]) - upper_roi[1])]
        # 原图roi
        roi_image = gray_image[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2]]

        # 根据阈值二值化，阈值需要实验
        if roi_image.size == 0 or roi_image.shape[0] == 0 or roi_image.shape[1] == 0:
            return end_points

        avg_gray, max_gray = np.mean(roi_image), np.max(roi_image)
        binary_threshold = avg_gray + int((max_gray - avg_gray) * 0.25)
        ret, ori_mask = cv2.threshold(gray_image, binary_threshold, 255, cv2.THRESH_BINARY)
        roi_ori_mask = ori_mask[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2]]

        # 分割结果mask
        roi_contours_mask = 255 - mask[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2]]

        # 得到侧脑室后角内部mask
        or_mask = cv2.bitwise_or(roi_ori_mask, roi_contours_mask)
        # cv2.imshow('or_mask', cv2.resize(or_mask, (int(or_mask.shape[1] * 2), int(or_mask.shape[0] * 2))))
        # 找到内沿轮廓，去除噪声保存最大轮廓
        new_contours, _ = cv2.findContours(255 - or_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if new_contours == ():
            new_contours, _ = cv2.findContours(255 - roi_contours_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        max_contour = measure_util.max_contour(new_contours)
        new_contour = measure_util.smooth_contour(roi_image, max_contour, ksize=15)

        # cv2.drawContours(roi_image, np.array([new_contour], dtype=np.int32), -1, (255, 0, 255), 1)
        # cv2.imshow('roi_image',cv2.resize(roi_image, (int(roi_image.shape[1] * 2), int(roi_image.shape[0] * 2))))

        line = [list(li) for li in line]
        line[0][0] -= roi[0]
        line[0][1] -= roi[1]

        # 新点是原线段斜率交于新轮廓的点
        new_roi_point = math_util.line_intersect_with_polygon(line, new_contour)
        if not new_roi_point:
            return end_points

        refined_point = [math_util.vec_add(roi, new_roi_point[0]), math_util.vec_add(roi, new_roi_point[1])]

        dist = math_util.distance_between(end_points[0], end_points[1])
        if abs(math_util.distance_between(refined_point[0], refined_point[1]) - dist) / dist <= wrong_ratio:
            return refined_point

        return end_points


if __name__ == '__main__':
    image_file = r'F:\hnu\lab\backbonePic\test\seg_test\123\05244a67da7644229771a239a49e9ae6.jpg'
    image = cv2.imread(image_file)
