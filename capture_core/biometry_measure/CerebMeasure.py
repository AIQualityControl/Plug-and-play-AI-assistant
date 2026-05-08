#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import math
import cv2
import numpy as np
from ..QcDetection.utility import math_util, draw_util
from . import measure_util


_FOR_DEBUG_ = False


class CerebMeasure:
    """
    Measurement of Transverse Cerebellar Diameter (TCD)
    """

    def __init__(self):
        '''constructor'''
        pass

    @classmethod
    def do_measure(cls, type2mask, image, detection_info):

        box_center = []
        # 1. Get the center points of the segmentation results for the third ventricle and cerebral aqueduct
        for type in ['third ventricle', 'cerebral aqueduct']:
            if type not in type2mask:
                continue
            for type_Gmask in type2mask[type]:
                # Calculate the maximum contour of the mask
                contour = measure_util.max_contour_of_GenericMask(type_Gmask)
                # If the contour has fewer than 3 points (3 points are required to form a polygon)
                if contour is None or len(contour) <= 3:
                    continue
                # Calculate the centroid from the obtained contour
                M = cv2.moments(contour)
                center_x = M['m10'] / M['m00']
                center_y = M['m01'] / M['m00']
                box_center.append([center_x, center_y])
        # 2. Get the center points of the detection results for cerebellum, cerebral falx, and thalamus
        qn_center_point = []
        for info in detection_info:
            if info['name'] in ['cerebellum', 'cerebral falx', 'brain midline']:
                box_center.append(math_util.mid_point(info['vertex'][0], info['vertex'][1]))
            if info['name'] in ['thalamic']:
                qn_center_point.append(math_util.mid_point(info['vertex'][0], info['vertex'][1]))
        if len(qn_center_point) == 2:
            box_center.append(math_util.mid_point(qn_center_point[0], qn_center_point[1]))

        brain_line = []
        if len(box_center) >= 2:
            # Extract the x and y coordinates of these points and store them in x_points and y_points lists respectively
            x_points = [pt[0] for pt in box_center]
            y_points = [pt[1] for pt in box_center]
            # Fit a line through the center points; brain_line is an array of polynomial coefficients describing the slope and intercept of the fitted line
            brain_line = np.polyfit(x_points, y_points, 1)

            # Display the fitted midline
            show = False
            if show:
                cls.show_brain_line(brain_line, image)
        # If there is no cerebellar hemisphere segmentation or no midline, estimate based on detection results
        if 'cerebellar hemisphere' not in type2mask or len(brain_line) < 2:
            point_CEREB = cls.default_tcd_measure(image, detection_info, brain_line)
            return point_CEREB

        # Get the maximum contours of the two cerebellar hemisphere masks, store them in xnbq_contours, and sort by contour size in descending order
        xnbq_contours = [measure_util.max_contour_of_GenericMask(xnbq_info) for xnbq_info in type2mask['cerebellar hemisphere']]
        xnbq_contours.sort(key=lambda x: x.size, reverse=True)

        # Determine the number of cerebellar hemispheres: use normal tcd_measure for two, single_tcd_measure for one
        if len(xnbq_contours) >= 2:
            point_CEREB = cls.tcd_measure(xnbq_contours[:2], brain_line, image)

            # xnbq_info['mask'] is the cerebellar hemisphere mask; xnbq_info['box'] is the bounding box of the mask
            mask_box_list = [[xnbq_info['mask'], xnbq_info['box']] for xnbq_info in type2mask['cerebellar hemisphere']]
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
        1.Initialize an empty list xnbq_vertex to store contour vertex coordinates and a variable idx set to -1.
        2.Iterate over the input contour list xnbq_contours:
            Use cv2.approxPolyDP() to perform polygon approximation on each contour.
            Use np.squeeze() to remove redundant dimensions and get flattened contour coordinates.
            If idx is negative, set it to the number of vertices of the current contour.
            Append the vertex coordinates of the current contour to xnbq_vertex.
        3.Use cv2.minAreaRect() to calculate the minimum enclosing rectangle of the contour, obtaining the rectangle's center coordinates, width, height, and rotation angle.
        4.Get the slope k and intercept b from brain_line, and calculate the counterclockwise rotation angle re_angle.
        5.Use math_util.rotate_points() to rotate the vertices in xnbq_vertex counterclockwise.
        6.Create a blank mask (pure black) with the same size as the input image, then draw and fill the two contour regions on the mask.
        7.Use math_util.boundingbox() to get the minimum x,y and maximum x,y of xnbq_vertex (i.e., the two farthest points).
        8.Use measure_util.points_with_max_vertical_dist() to find two points with the maximum vertical distance from the mask as end_points.
        9.If _FOR_DEBUG_ is true, display the image with rotated contours and the two farthest points.
        10.Use math_util.rotate_points() to rotate end_points counterclockwise and return the rotated result points_CEREB.
        '''
        xnbq_vertex = []
        idx = -1
        for contour in xnbq_contours:
            # Use cv2.approxPolyDP to reduce the number of points and computation
            contour = cv2.approxPolyDP(contour, 1, True)
            contour = np.squeeze(contour)

            # idx records the point count of the first contour to distinguish the two contours in subsequent operations
            if idx < 0:
                idx = len(contour)

            # Store the simplified contour points in the xnbq_vertex list
            xnbq_vertex.extend(contour)

        # Use cv2.minAreaRect to calculate the minimum enclosing rectangle containing all contour points, returning center, (w, h), and angle
        center, (w, h), angle = cv2.minAreaRect(np.array(xnbq_vertex))

        # Calculate the rotation angle re_angle to align the contour points with the baseline
        k, b = brain_line
        re_angle = -math.atan(k)

        # Use math_util.rotate_points to rotate xnbq_vertex by re_angle around center
        xnbq_vertex = math_util.rotate_points(xnbq_vertex, center, re_angle, in_degree=False)

        # Create an all-zero mask with the same size as the input image, convert xnbq_vertex to two contours, and fill them into the mask with cv2.fillPoly
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        # convert to int
        contours = [np.array(xnbq_vertex[:idx], dtype=np.int32),
                    np.array(xnbq_vertex[idx:], dtype=np.int32)]
        mask = cv2.fillPoly(mask, contours, (255,))

        # Calculate the bounding box bbox containing the rotated contour points, and find two points with maximum vertical distance in the mask using measure_util.points_with_max_vertical_dist
        bbox = math_util.boundingbox(xnbq_vertex)

        end_points = measure_util.points_with_max_vertical_dist(mask, bbox)
        if not end_points:
            return

        if _FOR_DEBUG_:
            display_image = draw_util.draw_lineseg(mask, end_points, inplace=False)
            cv2.imshow('rotated contour', display_image)
            cv2.waitKey(0)

        # Use math_util.rotate_points to rotate end_points back to the original angle to get the final cerebellar hemisphere measurement points_CEREB
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
        """Estimate based on detection results when no cerebellar hemisphere segmentation exists"""
        # Used to store the bounding box vertices of the cerebellum
        bbox = None

        # Traverse detection_info to find the cerebellum; if found, assign its bounding box vertices to bbox
        for info in detection_info:
            if info['name'] == '小脑':
                bbox = info['vertex']

        # No cerebellum detected: if bbox is None, estimate default measurement points based on image dimensions
        if bbox is None:
            height, width = roi_image.shape[:2]
            # No segmentation or detection results
            pt_start = [int(width * 0.5), int(height * 0.4)]
            pt_end = [int(width * 0.5), int(height * 0.6)]
            return [pt_start, pt_end]

        # No valid brain_line: if brain_line length < 2, directly calculate the midpoint and vertices of the bounding box as the measurement line
        elif len(brain_line) < 2:
            x = int((bbox[0][0] + bbox[1][0]) * 0.5)
            pt_start = [x, int(bbox[0][1])]
            pt_end = [x, int(bbox[1][1])]
            return [pt_start, pt_end]

        # Use the midpoint of bbox as the cerebellum center
        xn_center = math_util.mid_point(bbox[0], bbox[1])

        # Calculate the line perpendicular to brain_line
        k, b = brain_line

        # slope_xn: slope of the perpendicular line; b_xn: intercept of the perpendicular line; line_xn: the perpendicular line
        slope_xn = -1 / k

        # Check if slope_xn is not infinity (meaning the line is vertical and cannot use standard line equations)
        if slope_xn != float('inf'):
            # Calculate the intercept
            b_xn = xn_center[1] - slope_xn * xn_center[0]

            # Get the equation of the perpendicular line to the brain midline
            # Convert standard line equation y = kx + b to parametric form p(t) = p0 + t * dir
            line_xn = math_util.kb_to_point_dir(slope_xn, b_xn)

            # Calculate upper and lower boundary lines of the bounding box in parametric form
            line_upper = math_util.kb_to_point_dir(0, bbox[0][1])
            line_under = math_util.kb_to_point_dir(0, bbox[1][1])

            # Calculate intersection points of the perpendicular line with upper and lower boundaries
            point_upper = math_util.line_intersect_with_line(line_xn, line_upper)
            point_under = math_util.line_intersect_with_line(line_xn, line_under)

            # distance1: vertical distance of the bounding box; distance2: distance between intersection points; distance: average of the two
            distance1 = abs(bbox[1][1] - bbox[0][1])
            distance2 = math_util.distance_between(point_upper, point_under)
            distance = (distance1 + distance2) / 2

            # horizontal_distance: horizontal component along the direction perpendicular to the fitted line for the given vertical distance
            horizontal_distance = distance / (2 * ((1 + slope_xn ** 2) ** 0.5))

            # Calculate the final start and end measurement points
            x1 = xn_center[0] + horizontal_distance
            x2 = xn_center[0] - horizontal_distance
            y1 = xn_center[1] + slope_xn * horizontal_distance
            y2 = xn_center[1] - slope_xn * horizontal_distance

            pt_start = [x1, y1]
            pt_end = [x2, y2]
        # If the line is vertical, directly calculate midpoint and vertices
        else:
            x = int((bbox[0][0] + bbox[1][0]) * 0.5)
            pt_start = [x, int(bbox[0][1])]
            pt_end = [x, int(bbox[1][1])]

        return [pt_start, pt_end]

    @classmethod
    def refine_points(cls, image, point_CEREB, mask_box_list, detection_info, brain_line):
        # Refine point_CEREB to ensure the final points meet expected standards
        # image: input image; point_CEREB: preliminary measurement points; mask_box_list: list of masks and bounding boxes; detection_info: object detection info; brain_line: fitted brain midline

        # First refinement
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
            # The difference is too large after modification, cancel the change
            return False
        return True

    @classmethod
    def refine_single_points(cls, image, point_CEREB, mask_box_list, detection_info, down):
        # Sort mask_box_list by the bottom y-coordinate of the mask bounding box in ascending order, then select mask and box based on the down parameter
        # down=True processes the lower cerebellar hemisphere
        mask_box_list.sort(key=lambda x: x[1][3])
        mask_box = mask_box_list[1] if down else mask_box_list[0]
        mask = mask_box[0]
        box = mask_box[1]

        # Sort point_CEREB by y-coordinate in ascending order
        point_CEREB.sort(key=lambda y: y[1])

        # Extract cerebellum detection box vertices and convert to 1D array
        xn_box = [info['vertex'] for info in detection_info if info['name'] == '小脑']
        xn_box = np.array(xn_box).reshape(-1)

        # Convert image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        height, width = gray_image.shape[:2]

        # Round coordinate values of xn_box and mask_box
        # xn_box: detection box; mask_box: segmentation box (minimum enclosing rectangle)
        xn_box = [round(x) for x in xn_box]
        mask_box = [round(x) for x in box]

        # Expand the bounding boxes: extend downward for lower hemisphere, upward for upper hemisphere
        if down:
            mask_box[3] += 10
            xn_box[3] += 5
        else:
            mask_box[1] -= 10
            xn_box[3] -= 5
        mask = mask * 255

        # Create two zero masks with the same size as input image for binary results
        mask_test_1 = np.zeros((height, width), dtype=np.uint8)
        mask_test_2 = np.zeros((height, width), dtype=np.uint8)

        # Extract sub-regions from grayscale image corresponding to mask_box and xn_box
        mask_box_image = gray_image[mask_box[1]:mask_box[3], mask_box[0]:mask_box[2]]
        xn_box_image = gray_image[xn_box[1]:xn_box[3], xn_box[0]:xn_box[2]]

        # Apply Otsu's thresholding to binarize the sub-regions
        _, binary_mask_box_image = cv2.threshold(mask_box_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        mask_test_1[mask_box[1]:mask_box[3], mask_box[0]:mask_box[2]] = binary_mask_box_image
        _, binary_xn_box_image = cv2.threshold(xn_box_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        mask_test_2[xn_box[1]:xn_box[3], xn_box[0]:xn_box[2]] = binary_xn_box_image

        # Compute intersection of the two masks using bitwise AND
        temp_mask = cv2.bitwise_and(mask_test_1, mask_test_2)

        # Combine intersection with original mask to get final_mask
        final_mask = cv2.bitwise_and(temp_mask, mask)

        # Determine whether to measure the inner or outer edge
        # cv2.countNonZero(temp_mask): count non-zero pixels in the intersection mask
        if cv2.countNonZero(temp_mask) <= cv2.countNonZero(mask) * 0.2 or \
                cv2.countNonZero(temp_mask) >= cv2.countNonZero(mask) * 1.3:
            # Too few binarized pixels (noise) or too many (indistinct edge), skip adjustment
            return point_CEREB
        elif cv2.countNonZero(final_mask) <= cv2.countNonZero(temp_mask) * 0.3:
            # Too small intersection: edge is likely outside the segmentation mask
            # Compute union of temp_mask and mask
            final_mask = cv2.bitwise_or(temp_mask, mask)

            # Find bounding box and extract the region
            x, y, w, h = cv2.boundingRect(final_mask)
            cut_final_mask = final_mask[y:y + h, x:x + w]

            # Median filtering
            cut_binary_image = cv2.medianBlur(cut_final_mask, 7)

            # Put the processed region back to the original position
            final_mask[y:y + h, x:x + w] = cut_binary_image

            # Find contours in the processed mask
            new_mask_contours, _ = cv2.findContours(final_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # Select the largest contour
            new_contour = measure_util.max_contour(new_mask_contours)

        elif cv2.countNonZero(final_mask) >= cv2.countNonZero(temp_mask) * 0.5:
            # Large intersection: edge is likely inside the mask
            final_mask = cv2.bitwise_and(temp_mask, mask)

            # Apply dilation followed by erosion
            final_mask = cls.dilate_then_erode(final_mask)

            # Process the mask with bounding box and median filtering
            x, y, w, h = cv2.boundingRect(final_mask)
            cut_final_mask = final_mask[y:y + h, x:x + w]
            cut_binary_image = cv2.medianBlur(cut_final_mask, 7)
            final_mask[y:y + h, x:x + w] = cut_binary_image
            new_mask_contours, _ = cv2.findContours(final_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            new_contour = cls.all_mask_contourHull(new_mask_contours)
        else:
            return point_CEREB

        # Line passing through point_CEREB[0] and point_CEREB[1]
        line = np.array(point_CEREB[0]), np.array(point_CEREB[0]) - np.array(point_CEREB[1])

        # Calculate intersection points between the line and the new contour
        new_points = math_util.line_intersect_with_polygon(line, new_contour)

        # Sort points by y-coordinate: descending for lower hemisphere, ascending for upper hemisphere
        if down:
            new_points.sort(key=lambda y: y[1], reverse=True)
        else:
            new_points.sort(key=lambda y: y[1])
        if len(new_points) == 0:
            return point_CEREB
        new_point = new_points[0]

        if down:
            return [point_CEREB[0], new_point]
        else:
            return [new_point, point_CEREB[1]]

    @classmethod
    def all_mask_contourHull(cls, all_contours):
        # Calculate convex hull for each connected component
        convex_hulls = []
        contours_2D = []
        for contour in all_contours:
            if len(contour) < 3:
                continue
            convex_hull = cv2.convexHull(contour)
            convex_hulls.append(convex_hull)
            contours_2D.extend(contour.tolist())
        # Merge convex hulls of all connected components
        merged_convex_hull = cv2.convexHull(np.concatenate(convex_hulls))
        return merged_convex_hull

    @classmethod
    def dilate_then_erode(cls, mask, kernel_size=5, iterations=1):
        # Create kernel for dilation and erosion
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        # Dilation operation
        dilated_mask = cv2.dilate(mask, kernel, iterations=iterations)
        # Erosion operation
        eroded_mask = cv2.erode(dilated_mask, kernel, iterations=iterations)

        return eroded_mask

    @classmethod
    def show_brain_line(cls, brain_line, image):
        # Display the brain midline
        import matplotlib.pyplot as plt
        k, b = brain_line
        height, width = image.shape[:2]

        # Calculate intersection points of the line with image boundaries
        # Line equation: y = kx + b
        # y = b when x = 0
        y0 = int(b)
        # y = k*width + b when x = width
        y1 = int(k * width + b)

        # Ensure y0 and y1 are within image height range
        y0 = max(0, min(height - 1, y0))
        y1 = max(0, min(height - 1, y1))

        # Draw the line on the image
        image_with_line = cv2.line(image.copy(), (0, y0), (width, y1), (0, 255, 0), 2)  # Green line, thickness 2
        plt.imshow(cv2.cvtColor(image_with_line, cv2.COLOR_BGR2RGB))
        plt.title("Image with Fitted Line")
        plt.axis("off")
        plt.show()