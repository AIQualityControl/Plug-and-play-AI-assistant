#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@description:
@date       : 2022/03/25 12:15:16
@author     : Guanghua Tan
@email      : guanghuatan@hnu.edu.cn
@version    : 1.0
'''

import cv2
import math
import numpy as np


def compare_points(a):
    return len(a)


class EllipseFitting:
    def __init__(self):
        '''constructor'''
        pass

    @classmethod
    def fit_ellipse(cls, gray_image, is_bin=False):
        """
        return ellipse in (top_left, bottom_right, degree)
        """
        bndry_points = cls.get_bndry_points(gray_image, is_bin)
        if bndry_points is None or len(bndry_points) == 0:
            return

        return cls.fit_ellipse_by_points(bndry_points)

    @classmethod
    def fit_ellipse_by_points(cls, bndry_points):
        ellipse = cv2.fitEllipseDirect(bndry_points)

        # ------ debug---------
        # cls.__show_bndry_points(gray_image, bndry_points, ellipse, is_bin)

        clean_points = cls.filter_noise(bndry_points, ellipse)
        if len(clean_points) != len(bndry_points) and len(clean_points) >= 5:
            ellipse = cv2.fitEllipseDirect(np.array(clean_points))

        center, sizes, angles = ellipse

        # convert to ellipse annotation
        return ([center[0] - sizes[0] / 2, center[1] - sizes[1] / 2],
                [center[0] + sizes[0] / 2, center[1] + sizes[1] / 2],
                angles)

    @classmethod
    def __show_bndry_points(cls, gray_image, bndry_points, ellipse=None, is_bin=False):
        if is_bin:
            gray_image *= 255
        color_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)

        for pt in bndry_points:
            pt = pt[0]
            color_image[int(pt[1]), int(pt[0])] = (255, 0, 0)

        if ellipse is not None:
            center, size, angle = ellipse
            cv2.ellipse(color_image, (round(center[0]), round(center[1])), (round(size[0] / 2), round(size[1] / 2)),
                        angle, 0, 360, (0, 0, 255))

        cv2.imshow('ellipse', color_image)

    @classmethod
    def get_bndry_points(cls, gray_image, is_bin=False):
        # gray_image[gray_image <= 150] = 0
        # bin_image = gray_image
        if is_bin:
            bin_image = gray_image
        else:
            _, bin_image = cv2.threshold(gray_image, 150, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(bin_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if len(contours) == 0:
            return

        # find contour with max points
        points_of_contours = [len(contour) for contour in contours]
        idx = np.argmax(points_of_contours)
        return np.squeeze(contours[idx])

    @staticmethod
    def DEGREE_TO_RAD(x):
        return 0.01745329252 * x

    @staticmethod
    def RAD_TO_DEGREE(x):
        return 57.295779513 * x

    @classmethod
    def filter_noise(cls, points, ellipse):

        # unroate to local coordinate system
        center, size, angle = ellipse
        angle = -EllipseFitting.DEGREE_TO_RAD(angle)
        c = math.cos(angle)
        s = math.sin(angle)

        a2 = size[0] * size[0] * 0.25
        b2 = size[1] * size[1] * 0.25

        outer_points = []
        inner_points = []
        on_points = []
        on_points_idx = [[]]

        for i, pt in enumerate(points):
            # pt = point[0]

            x = pt[0] - center[0]
            y = pt[1] - center[1]

            # un-rotate
            x1 = c * x - s * y
            y1 = s * x + c * y

            r2 = x1 * x1 / a2 + y1 * y1 / b2

            if r2 > 1.16:          # outer
                outer_points.append(pt)
                if len(on_points_idx[-1]) > 0:
                    on_points_idx.append([])
            elif r2 < 0.85:       # inner
                inner_points.append(pt)
                if len(on_points_idx[-1]) > 0:
                    on_points_idx.append([])
            else:
                on_points.append(pt)
                on_points_idx[-1].append(i)

            # unroated_points.append([(x1+center[0], y1+center[1])])
        # cls.__show_bndry_points(gray_image, unroated_points, (center, size, 0))

        # clean
        if not outer_points and not inner_points:
            return points

        # number of noise points is few
        thresh = len(points) / 8
        if len(outer_points) < thresh and len(inner_points) < thresh:
            return on_points

        thresh = int(len(points) * 0.05)
        thresh = max(thresh, 20)
        # ---
        if len(on_points_idx[-1]) == 0:
            on_points_idx = on_points_idx[:-1]

        if len(on_points_idx) > 2:
            # first one
            points_idx = on_points_idx[0]
            on_points = list(points[points_idx[0]:points_idx[-1] + 1])
            # for idx in on_points_idx[0]:
            #     on_points.append(points[idx])

            for i in range(1, len(on_points_idx) - 1):
                prev_pnts = on_points_idx[i][0] - on_points_idx[i - 1][-1]
                after_pnts = on_points_idx[i + 1][0] - on_points_idx[i][-1]

                points_idx = on_points_idx[i]
                if len(points_idx) < 5 or len(points_idx) < thresh and \
                   (prev_pnts > thresh or after_pnts > thresh):
                    continue

                on_points.extend(points[points_idx[0]:points_idx[-1] + 1])
                # for idx in on_points_idx[i]:
                #     on_points.append(points[idx])

            # last one
            points_idx = on_points_idx[-1]
            on_points.extend(points[points_idx[0]:points_idx[-1] + 1])
            # for idx in on_points_idx[-1]:
            #     on_points.append(points[idx])

        if len(inner_points) > len(points) / 8:
            on_points.extend(outer_points)
        elif len(outer_points) > len(points) / 8:
            on_points.extend(inner_points)

        # if len(inner_points) < len(points) / 8:
        #     on_points.extend(inner_points)
        # if len(outer_points) < len(points) / 8:
        #     on_points.extend(outer_points)

        # ellipse = cv2.fitEllipseDirect(np.array(on_points))
        if len(on_points) >= 5:
            ellipse = cv2.fitEllipseDirect(np.array(on_points))
        return cls.filter_noise_simple(points, ellipse)

    @classmethod
    def filter_noise_simple(cls, points, ellipse):
        # sourcery skip: merge-duplicate-blocks, remove-pass-elif, remove-redundant-if
        # unroate to local coordinate system
        center, size, angle = ellipse
        angle = -EllipseFitting.DEGREE_TO_RAD(angle)
        c = math.cos(angle)
        s = math.sin(angle)

        a2 = size[0] * size[0] * 0.25
        b2 = size[1] * size[1] * 0.25

        on_points_idx = [[]]
        for i, pt in enumerate(points):
            # pt = pt[0]

            x = pt[0] - center[0]
            y = pt[1] - center[1]

            # un-rotate
            x1 = c * x - s * y
            y1 = s * x + c * y

            r2 = x1 * x1 / a2 + y1 * y1 / b2
            if 0.85 < r2 < 1.20:    # on
                on_points_idx[-1].append(i)
            elif len(on_points_idx[-1]) == 0:
                continue
            elif len(on_points_idx[-1]) < 10:
                on_points_idx[-1] = []
            else:
                on_points_idx.append([])

        if len(on_points_idx[-1]) == 0:
            on_points_idx = on_points_idx[:-1]

        if len(on_points_idx) == 0:
            return points

        #
        thresh = max(len(points) * 0.05, 20)

        clean_points = []
        for i in range(len(on_points_idx)):
            points_idx = on_points_idx[i]
            if len(points_idx) < 4:
                continue

            # prev
            prev_pnts = 0
            if i == 0:
                if points_idx[0] > on_points_idx[-1][-1]:
                    prev_pnts = points_idx[0] - on_points_idx[-1][-1]
                else:
                    prev_pnts = len(points) - on_points_idx[-1][-1] + points_idx[0]
            else:
                prev_pnts = points_idx[0] - on_points_idx[i - 1][-1]

            start_idx = 0
            if prev_pnts > thresh:
                prev_sign = 0
                while start_idx < len(points_idx) and start_idx < thresh * 0.8:
                    idx = points_idx[start_idx]
                    pt = points[idx]

                    x = pt[0] - center[0]
                    y = pt[1] - center[1]

                    x1 = c * x - s * y
                    y1 = s * x + c * y

                    r2 = x1 * x1 / a2 + y1 * y1 / b2

                    if r2 < 1:
                        if prev_sign == 0:
                            prev_sign = -1
                        elif prev_sign > 0:
                            break
                    elif r2 > 1:
                        if prev_sign == 0:
                            prev_sign = 1
                        elif prev_sign < 0:
                            break

                    start_idx += 1

            # after
            after_pnts = 0
            if i == len(on_points_idx) - 1:
                if on_points_idx[0][0] > points_idx[-1]:
                    after_pnts = on_points_idx[0][0] - points_idx[-1]
                else:
                    after_pnts = len(points) - points_idx[-1] + on_points_idx[0][0]
            else:
                after_pnts = on_points_idx[i + 1][0] - points_idx[-1]

            end_idx = len(points_idx)
            if after_pnts > thresh:
                prev_sign = 0
                while end_idx > start_idx and len(points_idx) - end_idx < thresh * 0.8:

                    idx = points_idx[end_idx - 1]
                    pt = points[idx]

                    x = pt[0] - center[0]
                    y = pt[1] - center[1]

                    x1 = c * x - s * y
                    y1 = s * x + c * y

                    r2 = x1 * x1 / a2 + y1 * y1 / b2

                    if r2 < 1:
                        if prev_sign == 0:
                            prev_sign = -1
                        elif prev_sign > 0:
                            break
                    elif r2 > 1:
                        if prev_sign == 0:
                            prev_sign = 1
                        elif prev_sign < 0:
                            break

                    end_idx -= 1

            if start_idx < end_idx and end_idx > 0 and start_idx < len(points_idx):
                clean_points.extend(points[points_idx[start_idx]:points_idx[end_idx - 1] + 1])

        return clean_points


if __name__ == '__main__':
    mask = cv2.imread('E:/mask.jpg')
    gray_image = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    anno = EllipseFitting.fit_ellipse(gray_image)

    cv2.ellipse(mask, anno.center_point(), anno.half_size(), -anno.degree(), 0, 360, (0, 0, 255))

    cv2.imshow('fitting', mask)

    cv2.waitKey()
