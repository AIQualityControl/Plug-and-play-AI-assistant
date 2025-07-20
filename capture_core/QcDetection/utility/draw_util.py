import cv2
import numpy as np


def draw_mask_or_contour(image, mask_or_contour, include_contour=False, alpha=0.15, inplace=False):
    """
    polygon: contour of the mask
    """
    contour = None
    mask = mask_or_contour
    if isinstance(mask_or_contour, map):
        if 'vertex' in mask_or_contour:
            contour = mask_or_contour['vertex']
        else:
            return
    elif isinstance(mask_or_contour, list) or mask_or_contour.shape[1] < 3:
        contour = np.squeeze(mask_or_contour)

    if contour is not None:
        contours = [np.array(contour, dtype=np.int32)]

        # mask
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        cv2.fillPoly(mask, contours, color=(255,))
    elif include_contour:
        # find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # blend
    image = blend_with_mask(image, mask, alpha, inplace)

    if include_contour:
        cv2.polylines(image, contours, isClosed=True, color=(255, 0, 0))

        for contour in contours:
            contour = np.squeeze(contour)
            for pt in contour:
                cv2.circle(image, pt, 2, color=(255, 0, 0), thickness=cv2.FILLED)

    return image


def blend_with_mask(image, mask, alpha=0.15, inplace=False):
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif not inplace:
        image = image.copy()

    _, bin_mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
    bg_exclude_roi = cv2.bitwise_and(image, image, mask=cv2.bitwise_not(bin_mask))
    # cv2.imshow('bg exclude roi', bg_exclude_roi)

    # image with roi
    image[:, :, 2] = cv2.addWeighted(image[:, :, 2], 1 - alpha, mask, alpha, 0)
    weighted_roi = cv2.bitwise_and(image, image, mask=bin_mask)

    # cv2.imshow('weighted_roi', weighted_roi)

    image = cv2.add(bg_exclude_roi, weighted_roi)

    # cv2.imshow('blend_result', image)

    return image


def blend_with_bin_mask(image, bin_mask, alpha=0.15, inplace=False):
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif not inplace:
        image = image.copy()

    bg_exclude_roi = cv2.bitwise_and(image, image, mask=cv2.bitwise_not(bin_mask))
    # cv2.imshow('bg exclude roi', bg_exclude_roi)

    # image with roi
    image[:, :, 2] = cv2.addWeighted(image[:, :, 2], 1 - alpha, bin_mask, alpha * 255, 0)
    weighted_roi = cv2.bitwise_and(image, image, mask=bin_mask)

    # cv2.imshow('weighted_roi', weighted_roi)

    image = cv2.add(bg_exclude_roi, weighted_roi)

    return image


def draw_contours(image, contours, color=(244, 38, 244), thickness=1, inplace=False, draw_point=False, point_color=(0, 0, 255),
                  is_closed=True):
    """
    contours: list of contour
    """
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif not inplace:
        image = image.copy()

    if isinstance(contours, np.ndarray):
        if len(contours.shape) == 2 or len(contours.shape) == 3 and contours.shape[1] == 1:
            contours = [contours]
    elif len(contours[0]) == 2:
        contours = [contours]

    contours = [np.array(contour, dtype=np.int32) for contour in contours if contour is not None]
    cv2.polylines(image, contours, is_closed, color, thickness=thickness)

    if draw_point:
        for contour in contours:
            for pt in contour:
                cv2.circle(image, pt, 2, point_color, cv2.FILLED, cv2.LINE_AA)

    return image


def draw_polyline(image, polyline, color=(244, 38, 244), thickness=1, inplace=False, draw_point=False, point_color=(0, 0, 255)):
    return draw_contours(image, polyline, color, thickness, inplace, draw_point, point_color, is_closed=False)


def draw_line(image, line, color=(0, 0, 255), thickness=1, inplace=True):
    """
    line can be in format with (k, b) or (pt, dir)
    """
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif not inplace:
        image = image.copy()

    if line is None or len(line) < 2:
        return image

    pt, dir = line
    if pt is None or pt is None:
        return image

    if not isinstance(pt, tuple) and not isinstance(pt, list) and not isinstance(pt, np.ndarray):
        k, b = pt, dir
        #
        pt0 = (0, int(b))
        pt1 = (1200, int(k * 1200 + b))
    else:
        pt0 = (int(pt[0] - 2000 * dir[0]), int(pt[1] - 2000 * dir[1]))
        pt1 = (int(pt[0] + 2000 * dir[0]), int(pt[1] + 2000 * dir[1]))

    cv2.line(image, pt0, pt1, color, thickness, lineType=cv2.LINE_AA)
    return image


def draw_lineseg(image, lineseg, color=(0, 0, 255), inplace=True):
    """
    lineseg: (start_point, end_point)
    """
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif not inplace:
        image = image.copy()

    if lineseg is None or len(lineseg) != 2:
        return image

    pt0, pt1 = lineseg
    if pt0 is None or pt1 is None:
        return image

    pt0 = (int(pt0[0]), int(pt0[1]))
    pt1 = (int(pt1[0]), int(pt1[1]))

    cv2.line(image, pt0, pt1, color, lineType=cv2.LINE_AA)
    return image


def draw_ellipse(image, ellipse, color=(0, 255, 255), inplace=False, is_cv2_format=False):
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif not inplace:
        image = image.copy()

    if is_cv2_format:
        center, axes, angle = ellipse
        center = (int(center[0]), int(center[1]))
        axes = (int(axes[0] / 2), int(axes[1] / 2))
    else:
        upper_left, bottom_right, angle = ellipse
        center = [int((upper_left[0] + bottom_right[0]) / 2), int((upper_left[1] + bottom_right[1]) / 2)]
        axes = (int((bottom_right[0] - upper_left[0]) / 2), int((bottom_right[1] - upper_left[1]) / 2))
    cv2.ellipse(image, center, axes, angle, 0, 360, color, lineType=cv2.LINE_AA)

    return image


def draw_points(image, points, color=(255, 0, 0), inplace=False):
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif not inplace:
        image = image.copy()

    for pt in points:
        pt = list(map(round, pt))
        cv2.circle(image, pt, 2, color, cv2.FILLED, cv2.LINE_AA)

    return image


def draw_bidiameter(image, bidiameter, inplace=False):
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif not inplace:
        image = image.copy()

    if not bidiameter:
        return image

    if isinstance(bidiameter, (list, tuple)):
        major_axis, minor_axis = bidiameter
    else:
        # BiDiameterAnnotation
        major_axis, minor_axis = bidiameter.major_axis, bidiameter.minor_axis

    if major_axis:
        start, end = major_axis
        cv2.line(image, [int(start[0]), int(start[1])], [int(end[0]), int(end[1])], (0, 255, 0))

    if minor_axis:
        start, end = minor_axis
        cv2.line(image, [int(start[0]), int(start[1])], [int(end[0]), int(end[1])], (255, 0, 0))

    return image


def draw_box(image, box, color=(0, 255, 0), inplace=False):
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif not inplace:
        image = image.copy()

    x1, y1 = round(box[0][0]), round(box[0][1])
    x2, y2 = round(box[1][0]), round(box[1][1])

    cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)

    return image
