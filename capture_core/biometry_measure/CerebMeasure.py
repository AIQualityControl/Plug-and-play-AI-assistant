#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@description:
@date       : 2022/04/22 00:32:08
@author     : Guanghua Tan
@email      : guanghuatan@hnu.edu.cn
@version    : 1.0
'''

import math
import cv2
import numpy as np
from ..QcDetection.utility import math_util, draw_util
from . import measure_util


_FOR_DEBUG_ = False


class CerebMeasure:
    """
    测量小脑横径: measurement for Transverse Cerebellar Diameter (TCD)
    """

    def __init__(self):
        '''constructor'''
        pass

    @classmethod
    def do_measure(cls, type2mask, image, detection_info):

        box_center = []
        # 1、取第三脑室、中脑导水管的分割结果的中心点
        for type in ['第三脑室', '中脑导水管']:
            if type not in type2mask:
                continue
            for type_Gmask in type2mask[type]:
                # 计算mask的最大轮廓contour
                contour = measure_util.max_contour_of_GenericMask(type_Gmask)
                # 如果contour小于3（3点才能确定多边形）
                if contour is None or len(contour) <= 3:
                    continue
                # 通过得到的轮廓计算质心
                M = cv2.moments(contour)
                center_x = M['m10'] / M['m00']
                center_y = M['m01'] / M['m00']
                box_center.append([center_x, center_y])
        # 2、取小脑、大脑镰、丘脑目标检测结果的中心点
        qn_center_point = []
        for info in detection_info:
            if info['name'] in ['小脑', '大脑镰', '脑中线']:
                box_center.append(math_util.mid_point(info['vertex'][0], info['vertex'][1]))
            if info['name'] in ['丘脑']:
                qn_center_point.append(math_util.mid_point(info['vertex'][0], info['vertex'][1]))
        if len(qn_center_point) == 2:
            box_center.append(math_util.mid_point(qn_center_point[0], qn_center_point[1]))

        brain_line = []
        if len(box_center) >= 2:
            # 提取这些点的 x 坐标和 y 坐标，分别存储在 x_points 和 y_points 列表中
            x_points = [pt[0] for pt in box_center]
            y_points = [pt[1] for pt in box_center]
            # 通过中心点拟合一条直线，brain_line 是一个多项式系数数组，它描述了拟合的直线的斜率和截距。
            brain_line = np.polyfit(x_points, y_points, 1)

            # 显示拟合的中线
            show = False
            if show:
                cls.show_brain_line(brain_line, image)
        # 有中线但是没有分割结果 或者没有中线,那么就根据抓取结果来判断
        if '小脑半球' not in type2mask or len(brain_line) < 2:
            point_CEREB = cls.default_tcd_measure(image, detection_info, brain_line)
            return point_CEREB

        # 得到两个小脑半球mask的最大轮廓放在xnbq_contours列表中，并按轮廓大小进行排序，使得面积最大的轮廓排在列表的前面
        xnbq_contours = [measure_util.max_contour_of_GenericMask(xnbq_info) for xnbq_info in type2mask['小脑半球']]
        xnbq_contours.sort(key=lambda x: x.size, reverse=True)

        # 判断有几个小脑，两个就走正常的tcd_measure，一个就走single_tcd_measure
        if len(xnbq_contours) >= 2:
            point_CEREB = cls.tcd_measure(xnbq_contours[:2], brain_line, image)

            # xnbq_info['mask']是小脑半球的掩膜，xnbq_info['box']是小脑半球的掩膜的框
            mask_box_list = [[xnbq_info['mask'], xnbq_info['box']] for xnbq_info in type2mask['小脑半球']]
            point_CEREB = cls.refine_points(image, point_CEREB, mask_box_list, detection_info, brain_line)
        elif len(xnbq_contours) == 1:
            point_CEREB = cls.single_tcd_measure(xnbq_contours[0], brain_line, image)

        if _FOR_DEBUG_:
            display_image = image.copy()
            display_image = draw_util.draw_contours(display_image, xnbq_contours)
            # display_image = draw_util.draw_contours(display_image, brain_contours, inplace=True)

            if len(brain_line) > 0:
                display_image = draw_util.draw_line(display_image, brain_line)
                display_image = draw_util.draw_points(display_image, box_center)

            if point_CEREB:
                display_image = draw_util.draw_lineseg(display_image, point_CEREB, (255, 0, 0))
            cv2.imshow('xnbq', display_image)
            cv2.waitKey(0)

        return point_CEREB

    @classmethod
    def tcd_measure(self, xnbq_contours, brain_line, image):
        '''
        1.初始化一个空列表 xnbq_vertex 用于存储轮廓的顶点坐标，并初始化变量 idx 为 -1。
        2.遍历输入的轮廓列表 xnbq_contours:
            使用 cv2.approxPolyDP() 对每个轮廓进行多边形逼近。
            使用 np.squeeze() 函数去除多余的维度，得到平坦的轮廓坐标。
            如果 idx 为负数，将其设为当前轮廓的顶点数量。
            将当前轮廓的顶点坐标添加到 xnbq_vertex 中。
        3.使用 cv2.minAreaRect() 函数计算轮廓的最小外接矩形，得到矩形的中心坐标、宽度、高度和旋转角度。
        4.从 brain_line 中获取直线的斜率 k 和截距 b,并计算逆时针旋转角度 re_angle。
        5.使用 math_util.rotate_points() 函数对 xnbq_vertex 中的顶点进行逆时针旋转。
        6.创建一个和输入图像相同大小的空白掩模(纯黑) mask,然后在掩模上绘制填充了两个轮廓的区域。
        7.使用 math_util.boundingbox() 函数获取 xnbq_vertex 最小的x,y和最大的x,y(即距离最远的两个点)
        8.使用 measure_util.points_with_max_vertical_dist() 函数从 mask 中找到具有最大垂直距离的两个点，作为 end_points。
        9.如果 _FOR_DEBUG_ 为真，将显示旋转后的轮廓和两个最远点的图像。
        10.使用 math_util.rotate_points() 函数对 end_points 进行逆时针旋转，并返回旋转后的结果 points_CEREB。
        '''
        xnbq_vertex = []
        idx = -1
        for contour in xnbq_contours:
            # 使用cv2.approxPolyDP减少点数，减少计算量
            contour = cv2.approxPolyDP(contour, 1, True)
            contour = np.squeeze(contour)

            # idx 用于记录第一个轮廓的点数，以便在后续操作中区分两个轮廓
            if idx < 0:
                idx = len(contour)

            # 将简化后的轮廓点存储在 xnbq_vertex 列表中
            xnbq_vertex.extend(contour)

        # 使用 cv2.minAreaRect 计算包含所有轮廓点的最小外接矩形，并返回矩形的中心点 center、宽高 (w, h) 以及旋转角度 angle
        center, (w, h), angle = cv2.minAreaRect(np.array(xnbq_vertex))

        # 计算需要旋转的角度 re_angle，使轮廓点对齐基准线
        k, b = brain_line
        re_angle = -math.atan(k)

        # 使用 math_util.rotate_points 以 center 为中心，将 xnbq_vertex 旋转 re_angle 角度。
        xnbq_vertex = math_util.rotate_points(xnbq_vertex, center, re_angle, in_degree=False)

        # 创建一个与输入图像 image 大小相同的全零掩码，将 xnbq_vertex 转换为两个轮廓，并使用 cv2.fillPoly 将这些轮廓填充到掩码中
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        # convert to int
        contours = [np.array(xnbq_vertex[:idx], dtype=np.int32),
                    np.array(xnbq_vertex[idx:], dtype=np.int32)]
        mask = cv2.fillPoly(mask, contours, (255,))

        # 计算包含旋转后轮廓点的外接矩形 bbox，使用 measure_util.points_with_max_vertical_dist 在掩码中找到垂直距离最大的两点 end_points
        bbox = math_util.boundingbox(xnbq_vertex)

        end_points = measure_util.points_with_max_vertical_dist(mask, bbox)
        if not end_points:
            return

        if _FOR_DEBUG_:
            display_image = draw_util.draw_lineseg(mask, end_points, inplace=False)
            cv2.imshow('rotated contour', display_image)
            cv2.waitKey(0)

        # 使用 math_util.rotate_points 将 end_points 反向旋转回原始角度，以获得最终的小脑半球测量点 points_CEREB
        points_CEREB = math_util.rotate_points(end_points, center, -re_angle, in_degree=False)
        return points_CEREB

    @classmethod
    def single_tcd_measure(cls, xnbq_contour, brain_line, image):

        xnbq_points = np.squeeze(xnbq_contour)
        max_point = cls.max_distance_of_point2line(brain_line, xnbq_points)

        k, b = brain_line
        # (0, b) - (-b/k， 0)  = (b / k, b) = (1/k, 1) = (1, k)
        # kx - y + b  = 0
        line = (0, b), (1, k)
        symmetry_point = math_util.mirror_along_line(max_point, line)

        points_CEREB = [max_point, symmetry_point]

        return points_CEREB

    @classmethod
    def max_distance_of_point2line(cls, line, points):
        k, b = line
        denom = math.sqrt(k * k + 1)

        distance_list = []
        for p in points:
            dist = abs(k * p[0] - p[1] + b) / denom
            distance_list.append(dist)

        idx = np.argmax(distance_list)
        return points[idx]

    @classmethod
    def default_tcd_measure(cls, roi_image, detection_info, brain_line):
        """不存在小脑半球的分割结果，根据抓取结果估计"""
        # 用于存储小脑的边界框顶点
        bbox = None

        # 遍历 detection_info，查找检测结果中是否有小脑，如果找到，则将其边界框顶点赋值给bbox
        for info in detection_info:
            if info['name'] == '小脑':
                bbox = info['vertex']

        # 没有检测到小脑：如果 bbox 为 None，则根据图像尺寸估计一个默认的测量点对。
        if bbox is None:
            height, width = roi_image.shape[:2]
            # 不存在分割结果和抓取结果
            pt_start = [int(width * 0.5), int(height * 0.4)]
            pt_end = [int(width * 0.5), int(height * 0.6)]
            return [pt_start, pt_end]

        # 没有 brain_line：如果 brain_line 长度小于2，表示没有有效的直线拟合结果，直接计算边界框的中点和顶点，即bbox的中间的那条线作为测量
        elif len(brain_line) < 2:
            x = int((bbox[0][0] + bbox[1][0]) * 0.5)
            pt_start = [x, int(bbox[0][1])]
            pt_end = [x, int(bbox[1][1])]
            return [pt_start, pt_end]

        # return [(point1[0] + point2[0]) / 2, (point1[1] + point2[1]) / 2]
        # 即bbox的中点作为小脑的中点
        xn_center = math_util.mid_point(bbox[0], bbox[1])

        # 计算垂直于 brain_line 的直线
        k, b = brain_line

        # slope_xn：垂直线的斜率，b_xn：垂直线的截距，line_xn：垂直线
        slope_xn = -1 / k

        # 判断slope_xn是否不是正无穷大，如果是，意味着直线是垂直的，不能用通常的方法计算直线方程和交点
        if slope_xn != float('inf'):
            # 计算截距
            b_xn = xn_center[1] - slope_xn * xn_center[0]

            # 得到脑中线的垂直线的方程
            # 函数 kb_to_point_dir 将标准形式的直线方程y = kx + b转换为参数形式的直线方程p(t) = p0 + t * dir，p0是直线上的一点，dir是直线的方向向量
            line_xn = math_util.kb_to_point_dir(slope_xn, b_xn)

            # 计算边界框上下边界线，也是参数形式的直线
            line_upper = math_util.kb_to_point_dir(0, bbox[0][1])
            line_under = math_util.kb_to_point_dir(0, bbox[1][1])

            # 计算垂直线与上下边界线的交点
            point_upper = math_util.line_intersect_with_line(line_xn, line_upper)
            point_under = math_util.line_intersect_with_line(line_xn, line_under)

            # distance1 是边界框的垂直距离，distance2 是垂直线与上下边界线交点之间的距离，distance 是二者的平均距离。
            distance1 = abs(bbox[1][1] - bbox[0][1])
            distance2 = math_util.distance_between(point_upper, point_under)
            distance = (distance1 + distance2) / 2

            # horizontal_distance 是计算得到的水平距离，给定垂直距离 distance 的情况下，沿着与原始拟合直线垂直的方向的水平分量
            horizontal_distance = distance / (2 * ((1 + slope_xn ** 2) ** 0.5))

            # 最终结算出测量的起始点pt_start和终止点pt_end
            x1 = xn_center[0] + horizontal_distance
            x2 = xn_center[0] - horizontal_distance
            y1 = xn_center[1] + slope_xn * horizontal_distance
            y2 = xn_center[1] - slope_xn * horizontal_distance

            pt_start = [x1, y1]
            pt_end = [x2, y2]
        # 如果直线是垂直的，则直接计算中点和顶点
        else:
            x = int((bbox[0][0] + bbox[1][0]) * 0.5)
            pt_start = [x, int(bbox[0][1])]
            pt_end = [x, int(bbox[1][1])]

        return [pt_start, pt_end]

    @classmethod
    def refine_points(cls, image, point_CEREB, mask_box_list, detection_info, brain_line):
        # 对 point_CEREB 进行细化，以确保最终的点符合预期的标准
        # image: 图像数据，point_CEREB: 初步测量得到的点，mask_box_list: 掩码和边界框列表，detection_info: 目标检测信息。brain_line: 拟合的大脑中线

        # 第一次细化
        refine_points_1 = cls.refine_single_points(image, point_CEREB, mask_box_list, detection_info, True)
        if not cls.is_stdandard(refine_points_1, brain_line):
            refine_points_1 = point_CEREB

        refine_points_2 = cls.refine_single_points(image, refine_points_1, mask_box_list, detection_info, False)
        if not cls.is_stdandard(refine_points_2, brain_line):
            refine_points_2 = refine_points_1

        return refine_points_2

    @classmethod
    def is_stdandard(cls, refine_points, brain_line):
        brain_line_pd = math_util.kb_to_point_dir(brain_line[0], brain_line[1])
        distance_up = math_util.dist_of_point_to_line(refine_points[0], brain_line_pd)
        distance_down = math_util.dist_of_point_to_line(refine_points[1], brain_line_pd)
        distance_standard = distance_down if distance_down > distance_up else distance_up
        if abs(distance_up - distance_down) > distance_standard * 0.06:
            # print('更改之后差别太大，撤销此次改动')
            return False
        return True

    @classmethod
    def refine_single_points(cls, image, point_CEREB, mask_box_list, detection_info, down):
        # mask_box_list 根据掩码的边界框的右下角y（x[1][3]）进行升序排序，然后根据 down 参数选择掩码和边界框， down 为True对下面那个小脑半球进行处理
        mask_box_list.sort(key=lambda x: x[1][3])
        mask_box = mask_box_list[1] if down else mask_box_list[0]
        mask = mask_box[0]
        box = mask_box[1]

        # point_CEREB按每个点的 y 坐标进行升序排序
        point_CEREB.sort(key=lambda y: y[1])

        # 提取小脑检测框的顶点坐标，并转换为一维数组
        xn_box = [info['vertex'] for info in detection_info if info['name'] == '小脑']
        xn_box = np.array(xn_box).reshape(-1)

        # 将图像转换为灰度图像
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        height, width = gray_image.shape[:2]

        # 将 xn_box 和 mask_box 的坐标值四舍五入
        # xn_box是检测的框，mask_box是分割的框（最小外接矩形）
        xn_box = [round(x) for x in xn_box]
        mask_box = [round(x) for x in box]

        # 如果 down 为 True，处理下面那个小脑半球，向下扩展 mask_box 的底部（增加高度）和 xn_box 的底部。
        # 如果 down 为 False，处理上面那个小脑半球，向上扩展 mask_box 的顶部（减少高度）和 xn_box 的底部。
        if down:
            mask_box[3] += 10
            xn_box[3] += 5
        else:
            mask_box[1] -= 10
            xn_box[3] -= 5
        mask = mask * 255

        # 创建两个大小与输入图像相同的全零（黑色）掩膜 mask_test_1 和 mask_test_2，这些掩膜用于存放两个不同区域的二值化结果
        mask_test_1 = np.zeros((height, width), dtype=np.uint8)
        mask_test_2 = np.zeros((height, width), dtype=np.uint8)

        # 从灰度图像 gray_image 中提取出 mask_box 和 xn_box 对应的子区域图像，分别赋值给 mask_box_image 和 xn_box_image
        mask_box_image = gray_image[mask_box[1]:mask_box[3], mask_box[0]:mask_box[2]]
        xn_box_image = gray_image[xn_box[1]:xn_box[3], xn_box[0]:xn_box[2]]

        # 使用 Otsu 阈值法对 mask_box_image 和 xn_box_image 进行二值化处理，生成 binary_mask_box_image 和 binary_xn_box_image
        # 将二值化结果binary_mask_box_image 和 binary_xn_box_image分别填充到 mask_test_1 和 mask_test_2 对应的位置。
        _, binary_mask_box_image = cv2.threshold(mask_box_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        mask_test_1[mask_box[1]:mask_box[3], mask_box[0]:mask_box[2]] = binary_mask_box_image
        _, binary_xn_box_image = cv2.threshold(xn_box_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        mask_test_2[xn_box[1]:xn_box[3], xn_box[0]:xn_box[2]] = binary_xn_box_image

        # 使用按位与操作将 mask_test_1 和 mask_test_2 结合，生成 temp_mask。这个操作仅保留两个掩膜都为白色的区域，从而得到二者的交集部分。
        temp_mask = cv2.bitwise_and(mask_test_1, mask_test_2)

        # 将交集与原始掩膜 mask 结合生成最终的掩膜 final_mask
        final_mask = cv2.bitwise_and(temp_mask, mask)

        # 怎么判断是测在内侧，还是外侧
        # cv2.countNonZero(temp_mask): 计算 temp_mask 中非零像素的数量，即交集掩膜的大小,mask是原始分割的大小
        if cv2.countNonZero(temp_mask) <= cv2.countNonZero(mask) * 0.2 or \
                cv2.countNonZero(temp_mask) >= cv2.countNonZero(mask) * 1.3:
            # 如果二值化的部分太少，可能是噪声，或者边缘很不明显，这部分不做调整
            # 如果二值化的部分过多，可能是看不到边缘，很难确定边缘，这部分不做调整
            # print('很难判断，不做调整')
            # 不做调整，直接返回原始的point_CEREB
            return point_CEREB
        elif cv2.countNonZero(final_mask) <= cv2.countNonZero(temp_mask) * 0.3:
            # 交集太少，边缘可能在外侧，分割到里面了
            # 使用 cv2.bitwise_or(temp_mask, mask) 计算新的掩膜 final_mask，即 temp_mask 和 mask 的并集
            final_mask = cv2.bitwise_or(temp_mask, mask)

            # 找到 final_mask 的边界框并提取其区域（最小矩形边界框）
            x, y, w, h = cv2.boundingRect(final_mask)
            cut_final_mask = final_mask[y:y + h, x:x + w]

            # cv2.imshow('Final ', image)
            # cv2.imshow('Final Mk', cut_final_mask)
            # cv2.imshow('Final Mask', final_mask)
            # cv2.waitKey(0)

            # 中值滤波
            cut_binary_image = cv2.medianBlur(cut_final_mask, 7)

            # 将中值滤波处理后的图像 cut_binary_image 放回 final_mask 的对应位置，这样，final_mask 中只有边界框内的部分被处理过，边界框外的部分保持不变
            final_mask[y:y + h, x:x + w] = cut_binary_image

            # cv2.imshow('Fin', final_mask)
            # cv2.imshow('Fi', cut_binary_image)
            # cv2.waitKey(0)

            # 在处理后的 final_mask 中查找轮廓
            new_mask_contours, _ = cv2.findContours(final_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # 从找到的轮廓中选择最大的轮廓
            new_contour = measure_util.max_contour(new_mask_contours)
            # print('判断测小了，边缘应该还在外面')

        elif cv2.countNonZero(final_mask) >= cv2.countNonZero(temp_mask) * 0.5:
            # 交集很大，很可能边缘应该处于mask的内侧
            final_mask = cv2.bitwise_and(temp_mask, mask)

            # cv2.imshow('Fin', final_mask)
            # cv2.waitKey(0)

            # 对final_mask做膨胀和腐蚀
            final_mask = cls.dilate_then_erode(final_mask)

            # cv2.imshow('Fin', final_mask)
            # cv2.waitKey(0)

            # time1 = time.time()
            x, y, w, h = cv2.boundingRect(final_mask)
            cut_final_mask = final_mask[y:y + h, x:x + w]
            cut_binary_image = cv2.medianBlur(cut_final_mask, 7)
            final_mask[y:y + h, x:x + w] = cut_binary_image
            # cv2.imshow('final_mask', final_mask)
            # cv2.waitKey(0)
            new_mask_contours, _ = cv2.findContours(final_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            new_contour = cls.all_mask_contourHull(new_mask_contours)
            # print('判断测大了，边缘应该在里面')
        else:
            return point_CEREB

        # 过point_CEREB[0]和point_CEREB[1]的直线
        line = np.array(point_CEREB[0]), np.array(point_CEREB[0]) - np.array(point_CEREB[1])

        # 计算直线与新轮廓的交点
        new_points = math_util.line_intersect_with_polygon(line, new_contour)

        # 如果 down 为 True，则按 y 坐标降序排序，否则按 y 坐标升序排序
        if down:
            new_points.sort(key=lambda y: y[1], reverse=True)
        else:
            new_points.sort(key=lambda y: y[1])
        if len(new_points) == 0:
            return point_CEREB
        new_point = new_points[0]
        # display_image = image.copy()
        # display_image = draw_util.draw_points(display_image, [new_point], big=3)
        # display_image = draw_util.draw_points(display_image, point_CEREB, (0, 255, 0))
        # dist_newdownpoint = math_util.distance_between(new_down_point, point_CEREB[1])
        # cv2.imshow('display_image', display_image)
        # cv2.waitKey(0)
        if down:
            return [point_CEREB[0], new_point]
        else:
            return [new_point, point_CEREB[1]]

    @classmethod
    def all_mask_contourHull(cls, all_contours):
        # 计算每个连通域的凸包
        convex_hulls = []
        contours_2D = []
        for contour in all_contours:
            if len(contour) < 3:
                continue
            convex_hull = cv2.convexHull(contour)
            convex_hulls.append(convex_hull)
            contours_2D.extend(contour.tolist())
        # 合并所有连通域的凸包
        merged_convex_hull = cv2.convexHull(np.concatenate(convex_hulls))
        return merged_convex_hull

    @classmethod
    def dilate_then_erode(cls, mask, kernel_size=5, iterations=1):
        # 构建膨胀和腐蚀的内核
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        # 膨胀操作
        dilated_mask = cv2.dilate(mask, kernel, iterations=iterations)
        # 腐蚀操作
        eroded_mask = cv2.erode(dilated_mask, kernel, iterations=iterations)

        return eroded_mask

    @classmethod
    def show_brain_line(cls, brain_line, image):
        # 显示脑中线
        import matplotlib.pyplot as plt
        k, b = brain_line
        height, width = image.shape[:2]

        # 计算直线在图像边界上的交点
        # 直线方程为 y = kx + b
        # 当 x = 0 时，y = b
        y0 = int(b)
        # 当 x = width 时，y = k*width + b
        y1 = int(k * width + b)

        # 确保 y0 和 y1 在图像高度范围内
        y0 = max(0, min(height - 1, y0))
        y1 = max(0, min(height - 1, y1))

        # 在图像上绘制直线
        image_with_line = cv2.line(image.copy(), (0, y0), (width, y1), (0, 255, 0), 2)  # 绿色线，线宽2
        plt.imshow(cv2.cvtColor(image_with_line, cv2.COLOR_BGR2RGB))
        plt.title("Image with Fitted Line")
        plt.axis("off")
        plt.show()


if __name__ == '__main__':

    xnbq_contours = []
    image_path = r'F:\hnu\lab\backbonePic\test\seg_test\123\f1216f33794f4606bf2c050bc6c92d86.jpg'
    image = cv2.imread(image_path)

    tcd_measure = CerebMeasure()

    for xnbq_contour in xnbq_contours:
        cv2.drawContours(image, np.array([xnbq_contour], dtype=np.int32), -1, (44, 238, 44), 1)
        cv2.imshow('new', image)
        cv2.waitKey()
