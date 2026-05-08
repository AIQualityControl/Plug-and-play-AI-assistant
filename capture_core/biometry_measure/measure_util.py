#!/usr/bin/env python
# -*- encoding: utf-8 -*-


import cv2
import numpy as np
import math
from ..QcDetection.utility import math_util


def center_of_contour(contour):
    '''get contour center'''
    M = cv2.moments(contour)
    if M['m00'] == 0:
        M['m00'] = 1
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])

    return (cx, cy)


def mask_is_intersect(mask1, mask2):
    if not math_util.is_box_intersect(mask1['box'], mask2['box']):
        return False

    bit_mask1 = mask1['mask']
    bit_mask2 = mask2['mask']

    is_intersection = np.logical_and(bit_mask1, bit_mask2)
    return is_intersection.any()


def max_contour_of_mask(mask, critia='area', approx_epsilon=0, smooth_ksize=0):
    '''
    critia:  area | arclength | num_points
    approx_epsilon: simplify contour with specified aprrox_epsilon
                    if approx_epsilon = 0, no simplification
    '''
    # smooth
    if smooth_ksize > 0:
        mask = cv2.medianBlur(mask, smooth_ksize)

    method = cv2.CHAIN_APPROX_NONE if critia == 'num_points' else cv2.CHAIN_APPROX_SIMPLE
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, method)

    return max_contour(contours, critia, approx_epsilon)


def max_GenericMask(mask_info_list):
    """
    mask_info_list: [{'mask':, 'box':, 'score':}]
    return mask_info with maximum pixels
    """
    if not mask_info_list:
        return

    idx = 0
    if len(mask_info_list) > 1:
        nzs_list = []
        for mask_info in mask_info_list:
            mask = mask_info['mask']
            bbox = mask_info['box']
            mask = mask[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

            nzs_list.append(np.count_nonzero(mask))

        # nzs = [np.count_nonzero(mask_info['mask']) for mask_info in mask_info_list]
        idx = np.argmax(nzs_list)

    mask_info = mask_info_list[idx]
    return mask_info


def max_mask(mask_info_list):
    if not mask_info_list:
        return

    idx = 0
    if len(mask_info_list) > 1:
        nzs = [np.count_nonzero(mask_info['mask']) for mask_info in mask_info_list]
        idx = np.argmax(nzs)

    return mask_info_list[idx]['mask']


def max_contour_of_mask_list(mask_info_list, critia='area', approx_epsilon=0):
    """
    """
    if not mask_info_list:
        return

    idx = 0
    if len(mask_info_list) > 1:
        nzs = [np.count_nonzero(mask_info['mask']) for mask_info in mask_info_list]
        idx = np.argmax(nzs)

    mask_info = mask_info_list[idx]
    return max_contour_of_mask(mask_info['mask'], critia, approx_epsilon)


def max_contour(contours, critia='area', approx_epsilon=0):
    '''
    critia:  area | arclength | num_points
    '''
    if len(contours) == 0:
        return None

    if len(contours) == 1:
        contour = cv2.approxPolyDP(contours[0], approx_epsilon, True) if approx_epsilon > 0 else contours[0]
        return np.squeeze(contour)

    critia_func = cv2.contourArea
    if critia == 'arclength':
        critia_func = cv2.arcLength
    elif critia == 'num_points':
        critia_func = len

    # Find the maximum contour
    critia_values = [critia_func(contour) for contour in contours]
    idx = np.argmax(critia_values)

    contour = cv2.approxPolyDP(contours[idx], approx_epsilon, True) if approx_epsilon > 0 else contours[idx]
    return np.squeeze(contour)


def max_n_contours_of_mask(mask, num_contours=1, critia='area'):
    '''
    find the largest num_countour contours
    critia:  area | arclength | num_points
    '''
    method = cv2.CHAIN_APPROX_NONE if critia == 'num_points' else cv2.CHAIN_APPROX_SIMPLE
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, method)

    if len(contours) == 0:
        return None

    if len(contours) == 1:
        return [np.squeeze(contours[0])]

    critia_func = cv2.contourArea
    if critia == 'arclength':
        critia_func = cv2.arcLength
    elif critia == 'num_points':
        critia_func = len

    # get maximum contours
    critia_values = [[critia_func(contour), contour] for contour in contours]
    critia_values.sort(key=lambda x: x[0], reverse=True)

    num_contours = max(num_contours, len(critia_values))
    return [np.squeeze(contour) for _, contour in critia_values[:num_contours]]


def smooth_contour(image, contour, ksize=15):
    height, width = image.shape[:2]
    mask = np.zeros([height, width], np.uint8)

    cv2.fillPoly(mask, np.array([contour], dtype=np.int32), (255, ))

    mask = cv2.medianBlur(mask, ksize)

    smooth_contour = max_contour_of_mask(mask)
    if smooth_contour is None or len(smooth_contour) == 0:
        return contour

    return smooth_contour


def max_contour_of_GenericMask(Gmask, approx_epsilon=0, smooth_ksize=0):
    '''
    Gmask: GenericMask or list of GenericMask
    smooth_ksize > 0: mediumBlur, usually set to be 15
    approxepsilon > 0: approxPolyDP, usually set to be 1
    '''
    if not Gmask:
        return

    '''
    First, check if Gmask is a list, indicating multiple GenericMask objects.
    If the list is empty, return directly.
    If there is only one GenericMask object in the list, assign it to Gmask.
    If there are multiple GenericMask objects, call max_GenericMask()
    to select the one with the largest area and assign it to Gmask.
    '''
    if isinstance(Gmask, list):
        if len(Gmask) == 0:
            return
        if len(Gmask) == 1:
            Gmask = Gmask[0]
        else:
            # find one with largest
            Gmask = max_GenericMask(Gmask)

    # If the contour is precomputed and stored in 'polygon', return it directly
    if Gmask['polygon'] is not None:
        return Gmask['polygon']

    '''
    1. Gmask['box'] is the bounding box defined by top-left and bottom-right coordinates.
    Coordinates are rounded to integers using int(x + 0.5) to ensure valid integer indices.
    Extract the ROI from the mask data based on computed coordinates.
    2. Use Python slicing syntax mask[start_row:end_row, start_col:end_col] to extract the sub-region.
    max(roi[1] - 1, 0) and max(roi[0] - 1, 0) ensure indices are non-negative to avoid out-of-bounds.
    roi[3] + 2 and roi[2] + 2 extend the range slightly.
    '''
    roi = [int(x + 0.5) for x in Gmask['box']]
    mask = Gmask['mask']
    # Crop the segmented region; coordinates will be adjusted back later
    mask = mask[max(roi[1] - 1, 0):roi[3] + 2, max(roi[0] - 1, 0):roi[2] + 2]

    '''
    1. If smooth_ksize > 0, apply median blur to the mask for noise reduction
    and more accurate contour detection.
    2. Use cv2.findContours() to find contours in the smoothed mask with cv2.RETR_EXTERNAL
    to detect only outer contours.
    If no contours are found (len(contours) == 0), return empty list.
    If contours are found, iterate to find the one with maximum area.
    If no valid max contour is found, return empty list.
    '''
    if smooth_ksize > 0:
        mask = cv2.medianBlur(mask, smooth_ksize)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        return []

    max_area = 0
    max_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            max_contour = contour

    if max_contour is None:
        return []

    '''
    1. If approx_epsilon > 0, apply polygon approximation to the max contour
    using cv2.approxPolyDP() to reduce points and simplify the contour.
    2. Use np.squeeze() to reduce dimensions of max_contour from (n, 1, 2) to (n, 2).
    Adjust coordinates of each point to map back to the original image coordinate space.
    Store the adjusted contour in Gmask['polygon'] for later use.
    Finally, return the adjusted max_contour.
    '''
    if approx_epsilon > 0:
        max_contour = cv2.approxPolyDP(max_contour, approx_epsilon, True)

    max_contour = np.squeeze(max_contour)
    for pt in max_contour:
        # Adjust coordinates back to the original image
        pt[0] += roi[0] - 1
        pt[1] += roi[1] - 1

    Gmask['polygon'] = max_contour
    return max_contour


def match_detection_info(type2mask, detection_info, strc_name, smooth_ksize=0):
    # get contours
    if strc_name not in type2mask:
        return None

    if not detection_info:
        # get lowest mask
        Gmask = max(type2mask[strc_name], key=lambda x: x['box'][3])
        return [max_contour_of_GenericMask(Gmask, 1, smooth_ksize)]

    #
    boxes = [info['vertex'] for info in detection_info if info['name'] == strc_name]
    if not boxes:
        return None

    # expand box
    for i, box in enumerate(boxes):
        boxes[i] = [int(box[0][0] - 50), int(box[0][1] - 50),
                    int(box[1][0] + 50), int(box[1][1] + 50)]

    results = []
    for Gmask in type2mask[strc_name]:
        for box in boxes:
            if math_util.is_box_intersect(box, Gmask['box']):
                contour = max_contour_of_GenericMask(Gmask, 1, smooth_ksize)
                results.append(contour)
                break

    return results


def get_lowest_contour(contours):
    '''Get the lowest contour'''
    if len(contours) == 0:
        return

    if len(contours) == 1:
        return contours[0]

    y_list = []
    for contour in contours:
        max_y = max([pt[1] for pt in contour])
        y_list.append(max_y)

    # sort according to y
    idx = np.argmax(y_list)
    return contours[idx]


def match_detection_contours(contours, detection_info, strc_name):
    '''Check if segmentation and detection results match'''
    # contours - a list of contours
    # to match the results of segmentation and detection results

    if not detection_info:
        return [get_lowest_contour(contours)]

    boxes = []
    for result in detection_info:
        # result = eval(result)
        if result['name'] == strc_name:
            vertex = result['vertex']
            box = [int(vertex[0][0] - 50), int(vertex[0][1] - 50),
                   int(vertex[1][0] + 50), int(vertex[1][1] + 50)]
            boxes.append(box)

    if len(boxes) == 0:
        return None

    # computer mass center
    center_list = [center_of_contour(contour) for contour in contours]
    results = []
    for box in boxes:
        for pt_center, contour in zip(center_list, contours):
            if math_util.point_is_contained_by(pt_center, box):
                results.append(contour)

    return results


def rotate_contour_to_horizontal(contour, center=None, angle=None, in_degree=True, mask_size=None):
    """
    mask_size: (h, w), if mask_size is not None, convert contour to mask

    return: contour_or_mask, rotate_center, rotate_angle_in_degree
    """
    if contour is None or len(contour) < 2:
        return contour

    # get rotation angle
    if center is None or angle is None:
        center, (w, h), angle = cv2.minAreaRect(contour)
        angle = -angle if w > h else 90 - angle
        in_degree = True

    contour = math_util.rotate_points(contour, center, angle, in_degree)
    if not mask_size:
        return contour, center, angle

    # convert contour to mask
    mask = np.zeros(mask_size, dtype=np.uint8)
    cv2.fillPoly(mask, np.array([contour], dtype=np.int32), (255,))

    return mask, center, angle


def points_with_max_vertical_dist(mask, bbox=None):
    """
    bbox: [x, y, w, h] or [(x1, y1), (x2, y2)]
    """
    h, w = mask.shape[:2]
    if bbox is None:
        # if bouding box is not specified, use the whole image
        bbox = [(0, 0), (w, h)]
    elif type(bbox[0]) is int or type(bbox[0]) is float:
        # Convert [x, y, w, h] format to [(x1, y1), (x2, y2)] format
        bbox = [bbox[0:2], [bbox[0] + bbox[2], bbox[1] + bbox[3]]]

    pt_min, pt_max = bbox

    start_x = max(0, int(pt_min[0]))
    start_y = int(pt_min[1])
    end_x = min(int(pt_max[0]), w)

    max_dist = 0
    max_x = -1
    y = []
    # iterate each column
    for x in range(start_x, end_x):
        # Extract column vector for each column (from start row to last row)
        col = mask[start_y: int(pt_max[1] + 2), x]
        # Find indices of non-zero elements
        nz_idx = np.nonzero(col)[0]
        if len(nz_idx) < 2:
            continue
        # Calculate distance between last and first non-zero index (max vertical distance)
        dist = nz_idx[-1] - nz_idx[0]
        if dist > max_dist:
            max_dist = dist
            max_x = x
            y = [nz_idx[0], nz_idx[-1]]
    # If max_x < 0, no column with >=2 non-zero elements found, return None
    if max_x < 0:
        return

    pt_min = [max_x, y[0] + start_y]
    pt_max = [max_x, y[1] + start_y]
    # Return coordinates of two points with maximum vertical distance
    return [pt_min, pt_max]


def points_with_max_horiz_dist(mask, bbox=None):
    """
    bbox: [x, y, w, h] or [(x1, y1), (x2, y2)]
    """
    h, w = mask.shape[:2]
    if bbox is None:
        # if bouding box is not specified, use the whole image
        bbox = [(0, 0), (w, h)]
    elif type(bbox[0]) is int or type(bbox[0]) is float:
        # [x, y, w, h] -> [(x1, y1), (x2, y2)]
        bbox = [bbox[0:2], [bbox[0] + bbox[2], bbox[1] + bbox[3]]]

    pt_min, pt_max = bbox

    start_x = max(0, int(pt_min[0]))
    start_y = max(0, int(pt_min[1]))
    end_x = int(pt_max[0])
    end_y = min(int(pt_max[1]), h)

    max_dist = 0
    max_y = -1
    x = []
    # iterate each row
    for y in range(start_y, end_y):
        row = mask[y, start_x: end_x + 2]
        nz_idx = np.nonzero(row)[0]
        if len(nz_idx) < 2:
            continue

        dist = nz_idx[-1] - nz_idx[0]
        if dist > max_dist:
            max_dist = dist
            max_y = y
            x = [nz_idx[0], nz_idx[-1]]

    if max_y < 0:
        return

    pt_min = [x[0] + start_x, max_y]
    pt_max = [x[1] + start_x, max_y]

    return [pt_min, pt_max]


def major_axis_along_dir(contour, dir):
    # rotate to horizontal
    angle = math.atan2(dir[1], dir[0])
    center = contour[0]
    rotated_contour = math_util.rotate_points(contour, center, -angle, in_degree=False)

    # fill contour
    pt_min, pt_max = math_util.boundingbox(rotated_contour)
    w, h = int(pt_max[0] - pt_min[0]) + 1, int(pt_max[1] - pt_min[1]) + 1
    mask = np.zeros([h, w], dtype=np.uint8)

    translated_contour = [math_util.vec_subtract(pt, pt_min) for pt in rotated_contour]
    translated_contour = np.array(translated_contour, dtype=np.int32)
    # cv2.fillPoly(mask, [translated_contour], (255,))
    cv2.polylines(mask, [translated_contour], True, (255,))

    # cv2.imshow('mask', mask)

    # find two points with max horizontal distance
    end_points = points_with_max_horiz_dist(mask)
    if end_points:
        end_points = [math_util.vec_add(pt, pt_min) for pt in end_points]
        # unrotate
        end_points = math_util.rotate_points(end_points, center, angle, in_degree=False)

    return end_points


def scale_image_with_pad(image, new_shape, padding=0):
    """
    scale image to specified shape 'new_shape', padding the short edge with value
    """
    shape = image.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    if shape == new_shape:
        return image

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    # ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[0] * r)), int(round(shape[1] * r))
    dw, dh = new_shape[1] - new_unpad[1], new_shape[0] - new_unpad[0]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape != new_unpad:  # resize
        image = cv2.resize(image, (new_unpad[1], new_unpad[0]), interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=padding)  # add border
    return im


def _scale_mask_(masks, image_shape):
    # Rescale coordinates (xyxy) from im1_shape to im0_shape
    im1_shape = masks.shape[:2]
    if im1_shape == image_shape:
        return masks

    # calculate from im0_shape
    gain = min(im1_shape[0] / image_shape[0], im1_shape[1] / image_shape[1])  # gain  = old / new
    pad = (im1_shape[1] - image_shape[1] * gain) / 2, (im1_shape[0] - image_shape[0] * gain) / 2  # wh padding

    top, left = int(pad[1]), int(pad[0])  # y, x
    bottom, right = int(im1_shape[0] - pad[1]), int(im1_shape[1] - pad[0])

    if len(masks.shape) < 2:
        raise ValueError(f'"len of masks shape" should be 2 or 3, but got {len(masks.shape)}')
    masks = masks[top:bottom, left:right]

    masks = cv2.resize(masks, (int(gain * im1_shape[1]), int(gain * image_shape[0])))

    # padding
    if len(masks.shape) == 2:
        masks = np.expand_dims(masks, 2)

    return masks


def scale_mask(masks, image_shape):
    """
    scale mask to specified image shape in reverse, removing the padding value
    """
    if len(masks) > 2:
        # (b, h, w) -> (h, w, b)
        masks = masks.transpose(1, 2, 0)
        masks = _scale_mask_(masks, image_shape)
        masks = masks.transpose(2, 0, 1)
    else:
        masks = _scale_mask_(masks, image_shape)

    return masks


def get_scale_offset(mask, image_shape):
    """
    scale and offset for scaling mask to the original image shape
    """
    im1_shape = mask.shape
    if len(im1_shape) > 2:
        im1_shape = im1_shape[1:]

    gain = min(im1_shape[0] / image_shape[0], im1_shape[1] / image_shape[1])  # gain  = old / new
    pad = (im1_shape[1] - image_shape[1] * gain) / 2, (im1_shape[0] - image_shape[0] * gain) / 2  # wh padding

    return 1.0 / gain, [-pad[0], -pad[1]]