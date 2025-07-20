#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@description:
@date       : 2022/03/25 12:20:20
@author     : Guanghua Tan
@email      : guanghuatan@hnu.edu.cn
@version    : 1.0
'''
import cv2
import numpy as np

LOW_CONFIDENCE = 50
HIGH_CONFIDENCE = 150
MIN_DIST_TRANSFORM = 60
is_analysis = False  # True#


def compare_points(a):
    return len(a)


class LineFitting:
    def __init__(self):
        """constructor"""
        pass

    @classmethod
    def fit_line(cls, gray_image, mask):
        if len(gray_image.shape) > 2:
            gray_image = cv2.cvtColor(gray_image, cv2.COLOR_BGR2GRAY)

        _, bin_image = cv2.threshold(gray_image, 180, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        bin_image[mask <= LOW_CONFIDENCE] = 0
        bin_image[mask > HIGH_CONFIDENCE] = 255

        # bin_mask = cls._optimize_mask(bin_image)
        bin_mask = bin_image

        # boundary points
        bndry_points, bbox, _, rotated_mask, M = cls._boundary_points(bin_mask)
        if bndry_points is None or len(bndry_points) == 0:
            return

        bin_rotated = np.zeros_like(rotated_mask)
        # 复制矩形框内的值
        bin_rotated[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]
                    ] = rotated_mask[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]
        dst_image = cv2.distanceTransform(rotated_mask, cv2.DIST_L2, cv2.DIST_MASK_3)
        cv2.normalize(dst_image, dst_image, 255, 0, cv2.NORM_MINMAX)
        dst_image = dst_image.astype(np.uint8)
        max_coord = np.argmax(dst_image, axis=0)

        left_point = []

        dll = int(bbox[2] * 0.06)  # 去掉0.06尖端的位置
        for i in range(bbox[0] + dll, bbox[0] + int(bbox[2] * 0.2), 3):  # 取2端0.2的点进行直线拟合
            left_point.append([i, max_coord[i]])
        left_line = cv2.fitLine(np.array(left_point), cv2.DIST_L2, 0, 0.01, 0.01)

        right_point = []
        for i in range(bbox[0] + int(bbox[2] * 0.8), bbox[0] + bbox[2] - dll, 3):
            right_point.append([i, max_coord[i]])
        right_line = cv2.fitLine(np.array(right_point), cv2.DIST_L2, 0, 0.01, 0.01)

        ptLeft = cls._gradient_points(gray_image, M, bbox, left_line, True)
        ptRight = cls._gradient_points(gray_image, M, bbox, right_line, False)

        # rotate back
        if M is not None:
            reverseMatRotation = cv2.invertAffineTransform(M)
            ptLeft = np.dot(reverseMatRotation, np.array([ptLeft[0], ptLeft[1], 1]))
            ptRight = np.dot(reverseMatRotation, np.array([ptRight[0], ptRight[1], 1]))

        if is_analysis:
            gray_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)
            # bin_image = cv2.warpAffine(bin_image, M, (w, h))
            # bin_image = cv2.warpAffine(bin_mask, M, (w, h))
            bin_image = cv2.cvtColor(bin_image, cv2.COLOR_GRAY2RGB)
            bin_image[:, :, 0] = 0
            bin_image[:, :, 1] = 0
            alpha = 0.1
            gray_image = cv2.addWeighted(gray_image, 1, bin_image, alpha, 0)

            ptLeft = [int(i) for i in ptLeft]
            ptRight = [int(i) for i in ptRight]
            cv2.circle(gray_image, ptLeft, 2, (0, 0, 250), 1)
            cv2.circle(gray_image, ptRight, 2, (0, 0, 250), 1)

            cv2.imshow('new_gray_img', gray_image)

            k = cv2.waitKey(0)
            if k == 27:  # 键盘上Esc键的键值
                cv2.destroyAllWindows()

        return (ptLeft, ptRight)

    @classmethod
    def _gradient_points(cls, gray_image, M, bbox, line, is_left):
        # 同时旋转原图
        h, w = gray_image.shape[:2]
        gray_image = cv2.warpAffine(gray_image, M, (w, h))

        dis = 25
        over = 15
        vx, vy, x0, y0 = line[0], line[1], line[2], line[3]
        r = vy / vx
        if is_left:
            s_x = bbox[0] - dis
            gts = []
            pts = []
            for i in range(s_x, bbox[0] + over, 2):
                m_y = int(y0 + (i - x0) * r)
                patch = gray_image[m_y - 5:m_y + 5, i - 2:i + 2]
                # gts.append(np.sum(patch))
                pts.append((i, m_y))

                gradient_x = abs(cv2.Sobel(patch, cv2.CV_64F, 1, 0, ksize=3)).sum()
                gradient_y = 0  # abs(cv2.Sobel(patch, cv2.CV_64F, 0, 1, ksize=3)).sum()

                gts.append(gradient_x+gradient_y)

            # 计算水平梯度
            # differences_row = np.diff(np.array(gts))
            max_ind = np.argmax(gts)

            return pts[max_ind]
        else:
            gts = []
            pts = []
            s_x = bbox[0] + bbox[2] - over
            for i in range(s_x, bbox[0] + bbox[2] + dis, 2):
                m_y = int(y0 + (i - x0) * r)
                patch = gray_image[m_y - 5:m_y + 5, i - 2:i + 2]
                # gts.append(np.sum(patch))
                pts.append((i, m_y))

                gradient_x = abs(cv2.Sobel(patch, cv2.CV_64F, 1, 0, ksize=3)).sum()
                gradient_y = 0  # abs(cv2.Sobel(patch, cv2.CV_64F, 0, 1, ksize=3)).sum()

                gts.append(gradient_x + gradient_y)
            # 计算水平梯度
            # differences_row = np.diff(np.array(gts))
            max_ind = np.argmax(gts)
            return pts[max_ind]

    @classmethod
    def _boundary_points(cls, bin_image):
        contours, _ = cv2.findContours(bin_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if len(contours) == 0:
            return None, None, None, None, None

        max_contour = max(contours, key=cv2.contourArea)
        center, (rect_w, rect_h), rotate_angle = cv2.minAreaRect(max_contour)
        if rect_w < rect_h:
            rotate_angle -= 90

        h, w = bin_image.shape[:2]
        # rotate_angle : positive value mean counter-clockwise
        M = cv2.getRotationMatrix2D((w // 2, h // 2), rotate_angle, 1)  # 旋转矩阵
        bin_image = cv2.warpAffine(bin_image, M, (w, h))  # 仿射变换

        if is_analysis:
            cv2.imshow('rotated', bin_image)
            k = cv2.waitKey(0)
            if k == 27:  # 键盘上Esc键的键值
                cv2.destroyAllWindows()

        contours, _ = cv2.findContours(bin_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # if is_analysis:
        #     max_bbox = cv2.boundingRect(contours[0])
        #     cv2.rectangle(bin_image, (max_bbox[0], max_bbox[1]), (max_bbox[0] +
        #                                                           max_bbox[2], max_bbox[1] + max_bbox[3]),
        #                   (255, 255, 255), 2)
        #     cv2.imshow('rotated', bin_image)
        #     k = cv2.waitKey(0)
        #     if k == 27:  # 键盘上Esc键的键值
        #         cv2.destroyAllWindows()

        max_contour = max(contours, key=cv2.contourArea)
        max_bbox = cv2.boundingRect(max_contour)

        if len(contours) == 1:
            return contours[0], cv2.boundingRect(contours[0]), None, bin_image, M

        return max_contour, max_bbox, None, bin_image, M


if __name__ == '__main__':
    # pass
    from PIL import Image
    ms = LineFitting()

    img = Image.open(r"D:\filetake\python_workspace\polyp骨头分割\TestDataset\old_images\20231212_090521_32834.jpg")
    img = np.array(img)
    msk = Image.open(r"D:\filetake\python_workspace\polyp骨头分割\result_map\PolypPVT\CVC-300\20231212_090521_32834.png")
    msk = np.array(msk)

    print(img.shape, msk.shape)
    a, b = ms.fit_line(img, msk)
    print(a, b)
