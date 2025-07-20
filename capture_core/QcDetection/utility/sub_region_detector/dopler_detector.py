import os
import sys
import cv2
import numpy as np

# Allow relative imports when being executed as script.
if __name__ == "__main__" and (__package__ is None or __package__ == ''):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    # print(sys.path)
    # import utility.sub_region_detector  # noqa: F401
    __package__ = "utility.sub_region_detector"

from .image_utility import ImageUtility


def take_first(elem):
    return elem[0]


def compare_y(elem):
    return elem[1]


class DoplerDetector:
    def __init__(self):
        '''constructor'''
        pass

    @classmethod
    def is_dopler_with_spectrum(cls, bin_image, color_image, check_middle_empty=True):
        """
        docstring
        """
        height, width = bin_image.shape[:2]
        # if cls._is_color_bar(right_remain_image, scale):
        #     nzs = np.count_nonzero(bin_image[0:height//2,:])
        #     if nzs < height * width * 0.4:
        #         return True
        #     return True

        if height > width * 1.2:
            return False

        roi_height = max(10, height // 3)
        y_start = height // 2 - 16
        offset = min(20, width // 10)
        # offset = width // 10
        roi = bin_image[y_start: height // 2 + roi_height, offset:-offset]

        # cv2.imshow('dopler hzone roi', roi)
        # 1 - black zone between top image and bottom spectrum
        black_zone = ImageUtility.detect_horizontal_black_zone(roi, sort_by_gap=False, nzs_thresh=7)
        y_split = (y_start + black_zone[0], y_start + black_zone[1]) if black_zone is not None else None
        if cls.is_color_dopler_with_spectrum(color_image, bin_image, y_split):
            return True

        if black_zone is None:
            return False

        if y_start + (black_zone[0] + black_zone[1]) / 2 < height * 0.45:
            return False

        if black_zone[1] - black_zone[0] > 15:
            nzs = np.count_nonzero(bin_image)
            if nzs > height * width * 0.65:
                return False

        # top is full
        roi_image = bin_image[:height // 3, :]
        nzs = np.count_nonzero(roi_image)
        if black_zone[1] - black_zone[0] < 10 and \
                (black_zone[1] < roi.shape[0] * 0.6 and nzs > width * height * 0.25 or
                 roi.shape[0] * 0.6 < black_zone[1] < roi.shape[0] * 0.7 and nzs > width * height * 0.3):
            return False

        if y_split[0] > height // 2 and black_zone[1] - black_zone[0] < 15:
            nzs = np.count_nonzero(bin_image[:int(height * 0.4), :])
            if nzs > width * height * 0.3:
                return False

        # black zone is at middle, may be 4-sub graph
        if black_zone[0] <= 16 and black_zone[1] - black_zone[0] < 10:
            y = y_start + black_zone[1]
            roi = bin_image[y:, :]
            nzs = np.count_nonzero(roi)
            # cv2.imshow('dopler roi', roi)
            if nzs > roi.shape[0] * roi.shape[1] * 0.6:
                return False

        # top usually is not full
        if black_zone[1] - black_zone[0] < 12 or black_zone[0] < 20:
            roi_top = bin_image[0:y_start + black_zone[0], :]
            nzs = np.count_nonzero(roi_top)
            if nzs > (y_start + black_zone[0]) * width * 0.75:
                return False
        elif y_start + black_zone[0] > height * 0.6:
            roi_top = bin_image[0:height // 2, :]
            nzs = np.count_nonzero(roi_top)
            if nzs > width * height * 0.4:
                return False

        # last two columns
        nzs = np.count_nonzero(roi[:, -2:])
        end_cont = nzs >= roi.shape[0] * 2 - 5

        # horizontal bar
        if black_zone[1] < 10 and black_zone[1] + 13 < roi_height:
            horiz_roi = roi[black_zone[1] + 5:black_zone[1] + 13, :]
            nzs = np.count_nonzero(horiz_roi)
            if nzs > horiz_roi.shape[1] * 7.5:
                return False

        # middle is not empty
        roi = bin_image[y_start + black_zone[1]:, width // 10:-width // 10]
        # print(np.count_nonzero(roi))
        if check_middle_empty and (black_zone[1] - black_zone[0] < roi_height // 3 or
                                   np.count_nonzero(roi) > roi.shape[0] * roi.shape[1] * 0.4):
            roi_top = bin_image[0:(y_start + black_zone[0]) // 2, int(width * 0.4):int(width * 0.6)]
            nzs = np.count_nonzero(roi_top)
            totoal_pixels = roi_top.shape[0] * roi_top.shape[1]
            if nzs < totoal_pixels * 0.1:
                if nzs < totoal_pixels * 0.08:
                    return False

                # right should not be empty
                roi_top_right = bin_image[0:(y_start + black_zone[0]) // 2, int(width * 0.4):]
                nzs_right = np.count_nonzero(roi_top_right)
                if nzs_right > min(roi_top_right.shape[0] * roi_top_right.shape[1] * 0.13, totoal_pixels * 0.5):
                    return False

        # top is full
        y_start += black_zone[0]
        if width > y_start * 2 and y_start > height * 0.7:
            nzs = np.count_nonzero(bin_image[0:y_start, :])
            if nzs > y_start * width * 0.7:
                return False

        # bottom is pulse
        for i in range(roi.shape[0]):
            row = roi[i, :]
            if np.count_nonzero(row) > 0:
                roi = roi[i:, :]
                break

        # kernel = np.ones((3,3), dtype=np.uint8)
        # roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel)

        # too many pixels with large height
        if roi.shape[0] > height * 0.4 and np.count_nonzero(roi) > roi.size * 0.75:
            return False

        return cls.is_bottom_spectrum(roi[:-3, :], bin_image, end_cont, y_start, black_zone[1] - black_zone[0])

    @classmethod
    def is_color_dopler_with_spectrum(cls, color_image, gray_bin_image, y_split=None):
        if len(color_image.shape) < 3:
            return False

        bin_image = ImageUtility.convert_to_bin_image(color_image[:, 10:-5])
        # cv2.imshow('color dopler', bin_image)

        height, width = bin_image.shape[:2]
        # too many color points
        nzs = np.count_nonzero(bin_image)
        if nzs > height * width // 2:
            return False

        if y_split is not None and height * 0.7 < y_split[1] < height * 0.85:
            y_split_top, y_split_bottom = y_split
            # case 1: two different color
            nzs_top = np.count_nonzero(bin_image[0:y_split_top, :])
            nzs_bottom = np.count_nonzero(bin_image[y_split_bottom:, :])
            # cv2.imshow('bottom dopler', bin_image[y_split_bottom:, :])

            # top is color and bottom is gray
            if nzs_top > y_split_top * width // 5 and nzs_bottom < width * (height - y_split_bottom) * 0.04:
                nzs = np.count_nonzero(gray_bin_image[y_split_bottom:, :])
                if nzs > width * 5:
                    return True

            # top is gray and bottom is color
            if nzs_top < y_split_top * width * 0.04 and nzs_bottom > width * (height - y_split_bottom) * 0.4:
                return True

        contours, _ = cv2.findContours(bin_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        bbox_list = []
        small_bbox_list = []
        for contour in contours:
            bbox = cv2.boundingRect(contour)
            # height should be less than half
            if bbox[3] > height // 2:
                return False

            # specturm is at the bottom half
            if bbox[1] < height * 0.4 or bbox[3] < height // 10:
                continue

            # cross the whole width
            if bbox[2] > width * 0.76:
                bbox_list.append(bbox)
            elif bbox[2] > width // 10:
                small_bbox_list.append(bbox)

        if len(small_bbox_list) + len(bbox_list) >= 4:
            # many color pixels
            if nzs > height * width * 0.3:
                return False

            for bbox in small_bbox_list:
                if bbox[2] > width * 0.6:
                    return False

        # whether can be merged into one
        if len(bbox_list) == 2:
            top, bottom = bbox_list
            if top[1] > bottom[1]:
                bottom, top = bbox_list

            if abs(top[1] + top[3] - bottom[1]) < 4:
                return True

        if len(bbox_list) != 1:
            return False

        bbox = bbox_list[0]
        if bbox[3] > height * 0.3:
            # top contains only few color pixels
            roi_image = bin_image[0:bbox[1], :]
            nzs = np.count_nonzero(roi_image)

            ratio = 0.1 if bbox[3] > height * 0.45 else 0.15
            if nzs > height * width * ratio:
                return False

        return True

    @classmethod
    def is_bottom_spectrum(cls, roi, bin_image, end_cont, top, gap=0):

        roi_height, roi_width = roi.shape[:2]
        nzs = np.count_nonzero(roi[-6:, :])
        if nzs > roi_width * 4.5:
            roi = roi[: -6, :]

        # cv2.imshow('bottom spectrum', roi)

        pulse_list = []
        contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        img_height, img_width = bin_image.shape[:2]

        small_list = []
        # extra_small_list = []
        largest_bbox = None
        for contour in contours:
            if len(contour) < 15:
                continue
            bbox = cv2.boundingRect(contour)
            if bbox[3] < 10:
                continue

            if bbox[2] > roi_width * 0.9 or \
                    bbox[2] > roi_width * 0.85 and cls._is_black(roi, bbox):
                # area = cv2.contourArea(contour)
                if bbox[3] > img_height * 0.4:
                    area = cv2.contourArea(contour)
                    thresh = 0.8 if bbox[3] > img_height * 0.5 else 0.85
                    if area > bbox[2] * bbox[3] * thresh:
                        nzs = np.count_nonzero(bin_image[0:top, 0:img_width // 4])
                        if nzs > img_width // 4:
                            continue
                elif bbox[3] < img_height * 0.3 and bbox[1] + bbox[3] > roi_height - 3:
                    pulse_image = roi[bbox[1]: bbox[1] + bbox[3], bbox[0]: bbox[0] + bbox[2]]
                    nzs = np.count_nonzero(pulse_image)
                    if bbox[3] < img_height * 0.2 and nzs > bbox[2] * bbox[3] * 0.85:
                        continue

                    if nzs > bbox[2] * bbox[3] * 0.6 and end_cont:
                        nzs = np.count_nonzero(bin_image[img_height // 10:, -img_width // 5:-img_width // 10])
                        if nzs > img_height * img_width * 0.072:
                            continue
                elif bbox[1] < 10 and bbox[3] < roi_height * 0.3:
                    continue

                pulse_list.append(bbox)

            elif roi_height * 0.5 < bbox[3] < img_height * 0.4:
                small_list.append(bbox)
                if largest_bbox is None:
                    largest_bbox = bbox
                elif largest_bbox[2] < bbox[2]:
                    largest_bbox = bbox
            # elif bbox[3] > roi_height // 2 and bbox[3] < img_height * 0.4:
            #     extra_small_list.append(bbox)

        if len(pulse_list) == 1:
            bbox = pulse_list[0]
            if bbox[1] > roi_height // 10 and bbox[2] > roi_width - 4 and bbox[3] > img_height * 0.25 and \
                    bbox[1] + bbox[3] > roi_height - 4:
                roi_image = ImageUtility.get_roi_image(roi, bbox)
                nzs = np.count_nonzero(roi_image)
                if nzs > roi_image.size * 0.7:
                    return False

            return True
        elif len(pulse_list) == 2:
            # whether can be joined together
            if pulse_list[0][1] < pulse_list[1][1]:
                if abs(pulse_list[0][1] + pulse_list[0][3] - pulse_list[1][1]) < 2:
                    return True
            elif abs(pulse_list[1][1] + pulse_list[1][3] - pulse_list[0][1]) < 2:
                return True
            else:
                # bottom is status bar
                top_pulse, bottom_pulse = pulse_list
                if top_pulse[1] > bottom_pulse[1]:
                    bottom_pulse, top_pulse = pulse_list
                if top_pulse[1] + top_pulse[3] > bottom_pulse[1]:
                    return False

                if bottom_pulse[3] < top_pulse[3] // 2 and bottom_pulse[3] < roi_height // 5 and \
                        bottom_pulse[1] + bottom_pulse[3] > roi_height - 5:
                    return True

        elif len(pulse_list) == 0 and len(small_list) > 1 and largest_bbox[2] > roi_width * 0.45:
            ignore = False
            if largest_bbox[3] > img_height * 0.35:
                roi_image = roi[largest_bbox[1]:, ]
                nzs = np.count_nonzero(roi_image)
                if nzs > roi_image.shape[0] * roi_image.shape[1] * 0.75:
                    ignore = True

            if not ignore and cls._is_pulse_with_multiple_pieces(roi, small_list, largest_bbox, img_height):
                return True
        # elif gap > roi_height // 2 and len(small_list) == 1 and \
        #         small_list[0][2] > roi_width * 0.7 and small_list[0][3] > roi_height * 0.8:
        #     # exclude status bar
        #     small_bbox = small_list[0]
        #     roi_image = ImageUtility.get_roi_image(roi, small_bbox)
        #     nzs = np.count_nonzero(roi_image)
        #     if nzs < small_bbox[2] * small_bbox[3] * 0.8:
        #         return True

        return False

    @classmethod
    def _is_color_bar(cls, right_remain_image, scale):
        if right_remain_image is None or len(right_remain_image.shape) < 3:
            return False

        height, width = right_remain_image.shape[:2]
        if width < 10:
            return False

        roi = right_remain_image[0:height // 2, :]

        # cv2.imshow('dopler roi', roi)
        return DoplerDetector.is_dopler_by_color_bar(roi, scale, False)

    @classmethod
    def has_horizontal_line(cls, contour, bbox):

        prev_pt = contour[-1][0]
        for pt in contour:
            pt = pt[0]

            if pt[1] > bbox[1] + bbox[3] - 4:
                prev_pt = pt
                continue

            if abs(prev_pt[1] - pt[1]) > 1:
                if abs(pt[0] - prev_pt[0]) > bbox[2] * 0.9:
                    return True

                prev_pt = pt
        return False

    @classmethod
    def _is_pulse_with_multiple_pieces(cls, roi, small_list, largest_bbox, img_height):
        small_list.sort(key=take_first)

        # 首尾相接
        length = 0
        for i in range(len(small_list) - 1):
            if small_list[i + 1][0] < small_list[i][0] + small_list[i][2] - 5 or \
                    small_list[i][1] < largest_bbox[1] - 1 or \
                    small_list[i][1] + small_list[i][3] > largest_bbox[1] + largest_bbox[3] + 1:
                return False

            length += small_list[i][2]

        length += small_list[-1][2]
        if small_list[-1][1] < largest_bbox[1] - 1 or \
                small_list[-1][1] + small_list[-1][3] > largest_bbox[1] + largest_bbox[3] + 1:
            return False

        height, width = roi.shape[:2]
        if length > width * 0.9 or \
                largest_bbox[1] + largest_bbox[3] > height * 0.9 and \
                length > width * 0.65 and largest_bbox[3] < img_height // 4:
            return True

        return False

    @classmethod
    def _is_black(cls, bin_image, bbox):
        # cv2.imshow('black', roi)
        height, width = bin_image.shape[:2]
        roi = bin_image[bbox[1]:bbox[1] + bbox[3], bbox[0] + bbox[2]:int(width * 0.95)]
        nzs = np.count_nonzero(roi)
        if nzs < 10:
            return True

        if nzs < 15:
            roi = bin_image[bbox[1]:int(bbox[1] + bbox[3] * 0.95), bbox[0] + bbox[2]:width]
            nzs = np.count_nonzero(roi)
            if nzs < 10:
                return True

        return False

    @classmethod
    def _is_valid_dopler(cls, image, y_split):

        # cv2.imshow('right dopler', image)

        roi = image[:y_split, :]
        kernel = np.ones((3, 3))
        roi = cv2.morphologyEx(roi, cv2.MORPH_OPEN, kernel)

        if np.count_nonzero(roi) < 15:
            return True

        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bbox_list = []
        for contour in contours:
            if len(contour) < 16:
                continue

            bbox = cv2.boundingRect(contour)
            bbox_list.append(bbox)

        if len(bbox_list) == 1:
            bbox = bbox_list[0]
            if bbox[0] == 0 and bbox[1] < y_split - 1:
                return False

        return True

    @classmethod
    def is_dopler(cls, color_image):
        height, width = color_image.shape[:2]
        x_offset = int(width * 0.1)
        y_offset = int(height * 0.1)

        # convert to hls
        roi = color_image[y_offset: height - y_offset, x_offset: width - x_offset]
        hls = cv2.cvtColor(roi, cv2.COLOR_BGR2HLS)

        # historgram
        count_map = [0] * 52
        height, width = hls.shape[:2]
        for row in range(height):
            for col in range(width):
                pixel = hls[row, col]
                if pixel[1] < 40 or pixel[2] > 200 or pixel[2] < 40 or pixel[2] > 200:
                    continue

                count_map[pixel[0] // 5] += 1

        # print(count_map)

    @classmethod
    def detect_dopler_sector(cls, color_image, origin_bin_image, scale=1):
        '''
        return value: (rois, is_dopler)
        '''
        if len(color_image.shape) < 3:
            return

        # convert to binary image
        bin_image = cls.convert_green_to_bin_image(color_image)
        # cv2.imshow('green_to_bin', bin_image)

        nzs_col = np.count_nonzero(bin_image, axis=0)
        nzs = sum(nzs_col)

        height, width = color_image.shape[:2]
        if nzs < max(width, height) or nzs > (width + height) * 3:
            return

        # one or two sector:
        sector_list = cls.get_sector_list(nzs_col, width // 6, max(5, width // 15))
        if len(sector_list) == 0:
            return

        if len(sector_list) == 2 and sector_list[1][0] - sector_list[0][1] < 10:
            sector_list[0][1] = sector_list[1][1]
            if sector_list[1][2] > sector_list[0][2]:
                sector_list[0][2] = sector_list[1][2]
                sector_list[0][3] = sector_list[1][3]
            sector_list.pop()

        # print(sector_list)
        if len(sector_list) == 1:
            # x_range = sector_list[0]
            x_start, x_end, max_nzs, x_max = sector_list[0]

            x_span = x_end - x_start
            if x_span > width * 0.35 and x_span < width * 0.75:
                if x_end <= width * 0.4 and x_start < max(width // 20, 5) or \
                        x_start >= width * 0.6 and x_end > min(width - 6, width - width // 20):
                    pass
                elif width * 0.33 < x_max < width * 0.6 - 2:
                    if max_nzs > height // 3:
                        cnt = 0
                        for i in range(x_max, x_max + 10):
                            if abs(nzs_col[i] - nzs_col[x_max]) < 5:
                                cnt += 1
                        if cnt > 7:
                            return None, True

                        return [[0, 0, width - 1, height - 1]], True
                    elif max_nzs > height // 4:
                        nzs_row = np.count_nonzero(bin_image[:, x_max - 3:x_max + 4], axis=1)
                        # max gap
                        max_gap = 0
                        start_idx = -1
                        for i in range(0, len(nzs_row)):
                            if nzs_row[i] == 0:
                                if start_idx >= 0:
                                    gap = i - start_idx
                                    if gap > max_gap:
                                        max_gap = gap
                                    start_idx = -1
                            elif start_idx < 0:
                                start_idx = i

                        if max_gap < height // 8:
                            return [[0, 0, width - 1, height - 1]], True

            elif x_end < width // 5 or x_start > width * 0.8 or x_span > width * 0.8:
                return None, False

            # vertical
            nzs_row = np.count_nonzero(bin_image[:, x_start:x_end], axis=1)
            sector_list = cls.get_sector_list(nzs_row, height // 6)

            if len(sector_list) == 1:
                max_green_points = np.max(nzs_col[x_start:(x_start + x_end) // 2])
                if max_green_points < height * 0.1:
                    return None, True

                y_range = sector_list[0]
                y_span = y_range[1] - y_range[0]
                nzs = sum(nzs_row[y_range[0]:y_range[1]])
                if y_span < height // 6 or y_span > height * 0.8 or nzs > x_span * y_span * 0.1:
                    return None, False

                nzs_inner = np.count_nonzero(bin_image[y_range[0] + y_span // 4: y_range[1] + int(y_span * 0.75),
                                             x_start + x_span // 4: x_end + int(x_span * 0.75)])
                if nzs_inner > nzs * 0.5:
                    return None, False

                return None, True

            return None, False

        if len(sector_list) == 2:
            nzs = np.count_nonzero(nzs_col[sector_list[0][1]:sector_list[1][0]])
            zs = sector_list[1][0] - sector_list[0][1] - nzs

            # too short
            if sector_list[1][1] - sector_list[0][0] < width // 2:
                return

            #
            if nzs > (sector_list[1][0] - sector_list[0][1]) * 0.5:
                return

            if zs < width // 10:
                # if sector_list[1][1] - sector_list[0][0] > width * 0.7:
                #     return [[0, 0, width - 1, height -1]], True
                return
            else:

                span0 = sector_list[0][1] - sector_list[0][0]
                span1 = sector_list[1][1] - sector_list[1][0]

                if span1 > span0 * 1.8 or span0 > span1 * 1.8 or span0 < width // 5 or span1 < width // 5:
                    return

                # vertical
                nzs_row = np.count_nonzero(bin_image[:, sector_list[0][0]:sector_list[0][1]], axis=1)
                vert_sector_list = cls.get_sector_list(nzs_row, height // 6, max(5, height // 10))

                if len(vert_sector_list) == 0 or len(vert_sector_list) > 2:
                    return
                left = vert_sector_list[0]

                nzs_row = np.count_nonzero(bin_image[:, sector_list[1][0]:sector_list[1][1]], axis=1)
                vert_sector_list = cls.get_sector_list(nzs_row, height // 6, max(5, height // 10))

                if len(vert_sector_list) == 0 or len(vert_sector_list) > 2:
                    return
                right = vert_sector_list[0]

                overlap = max(right[1], left[1]) - min(right[0], left[0])
                if overlap < height * 0.1:
                    return

                vert_len = vert_sector_list[0][1] - vert_sector_list[0][0]
                if span0 < vert_len // 2 or span1 < vert_len // 2:
                    if sector_list[0][0] > width // 2 - 5 or sector_list[1][1] < width // 2 + 5:
                        return
                    return [[0, 0, width - 1, height - 1]], True

            # get split x
            x_split = cls.get_best_split_x(origin_bin_image, (sector_list[0][1] // scale, sector_list[1][0] // scale))

            if height > x_split * 1.8 or height > (width - x_split) * 1.8 or \
                    x_split > (width - x_split) * 1.8 or width - x_split > x_split * 1.8:
                return

            return [[0, 0, x_split, height - 1], [x_split, 0, width - x_split - 1, height - 1]], True

    @classmethod
    def get_sector_list(cls, nzs_col, len_thresh, max_gap=5):
        '''
        return value: (x_start, x_end, max_nzs, x_max)
        x_start, x_end: start and end of the sector
        x_max: x with maximum # of green points of each column(sector points)
        max_nzs: maximum # of green points of each column
        '''
        start = -1
        cnt = 0
        sector_list = []
        for i in range(0, len(nzs_col)):
            nzs = nzs_col[i]
            if nzs == 0:
                if start < 0:
                    continue
                if cnt > max_gap:
                    end = i - cnt
                    if end - start > len_thresh and sum(nzs_col[start:end]) > len_thresh * 2:
                        blend_nzs = [sum(nzs_col[x:x + 3]) for x in range(start, end - 3)]
                        max_idx = np.argmax(blend_nzs)
                        max_nzs = blend_nzs[max_idx]
                        sector_list.append([start, end, max_nzs, start + max_idx])
                    start = -1
                else:
                    cnt += 1
            else:
                if start < 0:
                    start = i
                cnt = 0

        if start >= 0:
            end = i - cnt
            if end - start > len_thresh and sum(nzs_col[start:end]) > len_thresh * 2:
                blend_nzs = [sum(nzs_col[x:x + 3]) for x in range(start, end - 3)]
                max_idx = np.argmax(blend_nzs)
                max_nzs = blend_nzs[max_idx]
                sector_list.append([start, end, max_nzs, start + max_idx])

        return sector_list

    @classmethod
    def convert_green_to_bin_image(cls, color_image, thresh=40):
        image = color_image.astype(np.int16)
        diff_10 = image[:, :, 1] - image[:, :, 0]
        diff_12 = image[:, :, 1] - image[:, :, 2]

        diff = diff_10 + diff_12
        diff[diff_10 < 15] = 0
        diff[diff_12 < 15] = 0
        diff[image[:, :, 0] > 150] = 0
        diff[image[:, :, 2] > 150] = 0
        # bin_image = bin_image // 3
        #

        _, bin_image = cv2.threshold(diff, thresh, 255, cv2.THRESH_BINARY)
        bin_image = bin_image.astype(np.uint8)

        return bin_image

    @classmethod
    def convert_color_to_bin_image(cls, color_image, thresh=90):

        diff10 = cv2.absdiff(color_image[:, :, 1], color_image[:, :, 0])
        diff12 = cv2.absdiff(color_image[:, :, 1], color_image[:, :, 2])
        diff20 = cv2.absdiff(color_image[:, :, 2], color_image[:, :, 0])

        diff_image = cv2.add(diff10, diff12)
        diff_image = cv2.add(diff_image, diff20)

        _, bin_image = cv2.threshold(diff_image, thresh, 255, cv2.THRESH_BINARY)

        diff = np.max(color_image, axis=2) - np.min(color_image, axis=2)
        low_diff_area = diff < 50
        # 检测过于亮的绿色区域，G值较高，RB值极低
        super_green_area = np.logical_and(
            np.logical_and(color_image[:, :, 1] > 120, color_image[:, :, 1] // 2 > color_image[:, :, 0]),
            color_image[:, :, 1] // 2 > color_image[:, :, 2])
        # 检测R值较高，但GB值较低，且相近的区域
        low_orange_area = np.logical_and(np.logical_and(color_image[:, :, 1] * 1.5 > color_image[:, :, 2],
                                                        color_image[:, :, 2] > color_image[:, :, 1]),
                                         np.abs(color_image[:, :, 2] - color_image[:, :, 2]) < 20)
        zero_area = np.logical_or(low_diff_area, super_green_area)
        zero_area = np.logical_or(zero_area, low_orange_area)
        bin_image[zero_area] = 0

        # cv2.imshow("bin_image", bin_image)
        # cv2.waitKey()
        return bin_image

    @classmethod
    def convert_RB_to_bin_image(cls, color_image, rb_thresh=100, rg_thresh=80, bg_thresh=60):

        diff01 = cv2.subtract(color_image[:, :, 0], color_image[:, :, 1])
        diff21 = cv2.subtract(color_image[:, :, 2], color_image[:, :, 1])
        diff20 = cv2.absdiff(color_image[:, :, 2], color_image[:, :, 0])

        mask = np.logical_or(diff01 > bg_thresh, diff21 > rg_thresh)
        mask = np.logical_and(diff20 > rb_thresh, mask)

        h, w = color_image.shape[:2]
        bin_image = np.zeros((h, w), dtype=np.uint8)

        bin_image[mask] = 255

        return bin_image

    @classmethod
    def get_dopler_mask(cls, color_image, diff_thresh=200, max_dopler_pixels=-1,
                        rb_thresh=150, rg_thresh=100):
        """
        return (mask, nzs):   mask is the pixels with dopler
                              nzs is the number of dopler pixels
        """

        image = color_image.astype(np.int16)
        diff01 = image[:, :, 0] - image[:, :, 1]
        diff21 = image[:, :, 2] - image[:, :, 1]
        diff20 = image[:, :, 2] - image[:, :, 0]

        diff_image = abs(diff01) + abs(diff21) + abs(diff20)
        mask = diff_image > diff_thresh

        nzs = np.count_nonzero(mask)
        if max_dopler_pixels > 0 and nzs > max_dopler_pixels:
            # diff20: max difference
            r_image = np.logical_and(diff20 > rb_thresh, diff21 > rg_thresh)
            b_image = np.logical_and(diff20 < -rb_thresh, diff01 > rg_thresh)
            # g_image = np.logical_and(diff01 < -rg_thresh, diff21 < -rg_thresh)

            mask = np.logical_or(b_image, r_image)
            nzs = np.count_nonzero(mask)

        return mask, nzs

    @classmethod
    def get_best_split_x(cls, bin_image, x_range):
        roi = bin_image[2:, x_range[0]:x_range[1]]

        # cv2.imshow('middle_roi', roi)

        # find x with largest y
        max_y = 0
        x_idx = 0
        height, width = roi.shape[:2]

        black_zone = []
        for i in range(width):
            col = roi[:, i]
            pos = np.nonzero(col)[0]

            if len(pos) == 0:
                black_zone.append(i)
            elif pos[0] > max_y:
                max_y = pos[0]
                x_idx = i

        if len(black_zone) > 0:
            return int(np.mean(black_zone)) + x_range[0]

        return x_idx + x_range[0]

    @classmethod
    def is_dopler_by_color_bar(cls, color_image, scale=-1, check_pseduo_color=True):
        """
        scale > 0: scale the image with the specified scale
        scale < 0: scale the color image automatically
        """
        if len(color_image.shape) < 3:
            return False

        if scale < 0:
            scale = ImageUtility.scale_of_image(color_image)

        # scale
        height, width = color_image.shape[:2]
        if scale != 1:
            height = height // scale
            width = width // scale
            color_image = cv2.resize(color_image, (width, height), interpolation=cv2.INTER_LINEAR)

        bin_image = cls.convert_color_to_bin_image(color_image, thresh=80)
        # cv2.imshow('color image', color_image)
        # cv2.imshow('color to bin', bin_image)
        # cv2.waitKey()

        # find color bar
        bbox_list = cls.find_color_bar(bin_image, color_image, scale, check_pseduo_color)

        return bbox_list is not None and len(bbox_list) == 1

    @classmethod
    def find_color_bar(cls, bin_image, color_image, scale, check_pseudo_color=True):
        height, width = bin_image.shape[:2]
        # is_taken_by_camera = height * scale > 2000 or width * scale > 2000

        bin_roi = bin_image[:height // 2, :]
        bin_roi = cv2.morphologyEx(bin_roi, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        # cv2.imshow('morph open', bin_roi)

        contours, _ = cv2.findContours(bin_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        counter_bbox_list = []
        merged_bbox_list = []
        merged_counter_list = []

        bbox_list = []

        red_bbox_list = []
        blue_bbox_list = []
        # large_bbox = None
        for counter in contours:
            # 先获取所有的counter bbox
            counter_bbox = list(cv2.boundingRect(counter))
            x, y, w, h = counter_bbox
            counter_bbox.extend([x + w // 2, y + h // 2])  # 获取
            counter_bbox_list.append(counter_bbox)

        # 同时对bbox和counters进行排序
        sorted_list = sorted(list(zip(counter_bbox_list, contours)), key=lambda x: x[0][-2])

        if len(sorted_list) == 1:
            merged_bbox_list.append(sorted_list[0][0][:4])
            merged_counter_list.append([sorted_list[0][1]])

        idx = 0
        while idx < (len(sorted_list) - 1):
            # 若len为1，不会进入循环
            merged_bbox = cls.merge_vertical_bbox(sorted_list[idx][0], sorted_list[idx + 1][0], height, width)
            if merged_bbox is not None:
                merged_bbox_list.append(merged_bbox)
                merged_counter_list.append([sorted_list[idx][1], sorted_list[idx + 1][1]])  # list
                idx += 2  # 合并之后应该+2
            else:
                merged_bbox_list.append(sorted_list[idx][0][:4])
                merged_counter_list.append([sorted_list[idx][1]])  # list
                if idx == len(sorted_list) - 1:
                    # 如果最后两个没能合并成功，则最后一个也要加入merged_bbox_list
                    merged_bbox_list.append(sorted_list[idx + 1][0][:4])
                    merged_counter_list.append([sorted_list[idx + 1][1]])
                    idx += 2  # 最后两个没有合并，应该+2
                else:
                    idx += 1  # 否则应该+1
        # print(merged_bbox_list)

        for merged_bbox, merged_counter in zip(merged_bbox_list, merged_counter_list):
            # bbox = cv2.boundingRect(contour)
            origin_bbox_list = None
            if len(merged_bbox) == 6:
                origin_bbox_list = merged_bbox[-2:]

            bbox = merged_bbox[:4]
            # 绘制bbox
            # x, y, w, h = bbox
            # rect_contour = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
            # contour = [rect_contour]
            # color_image_ = color_image.copy()
            # cv2.drawContours(color_image_, contour, -1, (0, 255, 0), -1)
            # cv2.imshow('color image', color_image_)
            # cv2.waitKey()
            # print(bbox)

            if bbox[2] < 3 or bbox[2] > 18 or bbox[3] < bbox[2] * 2:
                continue

            if bbox[1] > height * 0.4 and width * 0.3 < bbox[0] < width * 0.7:
                continue

            # too small color bar
            if bbox[2] < 4 and bbox[3] < bbox[2] * 3 or bbox[2] < 5 and bbox[3] < bbox[2] * 2.5:
                continue

            # bar = bin_roi[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
            # nzs = np.count_nonzero(bar)

            nzs = 0
            bbox_area = 0
            color_bar_list = []

            if isinstance(merged_counter, list):
                for counter in merged_counter:
                    nzs += cv2.contourArea(counter)
            else:
                nzs += cv2.contourArea(merged_counter)

            if origin_bbox_list:
                # merge之前的bbox
                max_with = bbox[2]  # 拼接时候是按照最宽的w来拼接的
                min_x = min(origin_bbox_list[0][0], origin_bbox_list[1][0])
                for origin_bbox in origin_bbox_list:
                    # 计算的时候不使用max_width
                    bbox_area += (origin_bbox[2] - 1) * (origin_bbox[3] - 1)
                    # 但在获取每一个color_bar的时候按照max_width
                    color_bar_list.append(color_image[origin_bbox[1]:origin_bbox[1] + origin_bbox[3],
                                          min_x:min_x + max_with])

            else:
                bbox_area += (bbox[2] - 1) * (bbox[3] - 1)
                color_bar_list.append(color_image[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]])

            if nzs < bbox_area * 0.6:
                continue

            # cv2.imshow('color bar', color_bar)
            # cv2.waitKey()
            # whether has red and blue
            ret = cls.is_valid_color_bar(color_bar_list)
            if ret == 3:
                bbox_list.append(bbox)
            elif ret == 1:
                red_bbox_list.append(bbox)
            elif ret == 2:
                blue_bbox_list.append(bbox)

        if len(bbox_list) == 0:
            if len(red_bbox_list) == 1 and len(blue_bbox_list) == 1:
                bbox = cls.merge_two_bbox(red_bbox_list[0], blue_bbox_list[0])
                if bbox is not None:
                    bbox_list = [bbox]
        elif len(bbox_list) == 1:
            # pseudo color image
            if check_pseudo_color:
                nzs = np.count_nonzero(bin_roi[height // 10:, :])
                # print(nzs / (height * width))
                if nzs > height * width * 0.05:
                    bbox = bbox_list[0]
                    color_bar = color_image[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]
                    if not cls.check_with_pseudo_color(color_bar):
                        return
        elif len(bbox_list) == 2:
            # merge
            bbox = cls.merge_two_bbox(bbox_list[0], bbox_list[1])
            if bbox is not None:
                bbox_list = [bbox]
        elif len(bbox_list) == 3:
            bbox_list.sort(key=take_first)
            if abs(bbox_list[0][0] - bbox_list[1][0]) < 2:
                bbox_list = bbox_list[0:2]
            elif abs(bbox_list[1][0] - bbox_list[2][0]) < 2:
                bbox_list = bbox_list[1:3]
            elif abs(bbox_list[0][0] - bbox_list[2][0]) < 2:
                bbox_list = [bbox_list[0], bbox_list[2]]

            if len(bbox_list) == 2:
                bbox = cls.merge_two_bbox(bbox_list[0], bbox_list[1])
                if bbox:
                    bbox_list = [bbox]

        # /////////// visualization ///////////////////
        # display_image = cv2.cvtColor(bin_roi, cv2.COLOR_GRAY2BGR)
        # if bbox_list is not None:
        #     for bbox in bbox_list:
        #         cv2.rectangle(display_image, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 0, 255))
        # cv2.imshow('color bar', display_image)

        return bbox_list

    @classmethod
    def is_valid_color_bar(cls, color_bar_list):
        """
        return: 1: red pixels only
                2: blue pixels only
                3: red and blue pixels
        """
        if not isinstance(color_bar_list, list):
            color_bar_list = [color_bar_list]

        merged_color_bar = np.concatenate(color_bar_list, axis=0)  # 沿着高度拼接
        red_pixels = 0
        blue_pixels = 0

        height, width = merged_color_bar.shape[:2]
        for i in range(width):
            for j in range(height):
                # 从上往下从左向右
                pixel = merged_color_bar[j, i, :]
                if pixel[0] > 120 and pixel[0] > pixel[1] * 2 and pixel[0] > pixel[2] * 2 or \
                        pixel[0] > 160 and pixel[0] > pixel[1] * 1.75 and pixel[0] > pixel[2] * 1.75 or \
                        pixel[1] > 160 and pixel[1] > pixel[2] * 1.75 and pixel[0] > pixel[2] * 1.2:
                    # 青色
                    blue_pixels += 1
                elif pixel[2] > 120 and pixel[2] > pixel[1] * 2 and pixel[2] > pixel[0] * 2 or \
                        pixel[2] > 160 and pixel[2] > pixel[1] * 1.8 and pixel[2] > pixel[0] * 1.8:
                    red_pixels += 1

                if red_pixels > 0 and blue_pixels > 0 and red_pixels + blue_pixels > 5:
                    return 3
            # if red_pixels + blue_pixels > 5:
            #     return 3

        if blue_pixels == 0:
            # if red_pixels > 10:
            #     return 3
            if red_pixels >= 3:
                return 1
        elif red_pixels == 0:
            if blue_pixels > 10:
                return 3
            if blue_pixels >= 3:
                return 2
        elif blue_pixels + red_pixels > 5:
            return 3
        return 0

    @classmethod
    def check_with_pseudo_color(cls, color_bar):
        height, width = color_bar.shape[:2]

        red_pixels = 0
        blue_pixels = 0
        for i in range(height):
            for j in range(width):
                pixel = color_bar[i, j, :]
                if pixel[0] > 120 and pixel[0] > pixel[1] * 2 and pixel[0] > pixel[2] * 2:
                    blue_pixels += 1
                elif pixel[2] > 120 and pixel[2] > pixel[1] * 2 and pixel[2] > pixel[0] * 2:
                    red_pixels += 1

        return red_pixels > 4 and blue_pixels > 4 or red_pixels > 10 or blue_pixels > 10

    @classmethod
    def merge_two_bbox(cls, top, bottom):
        half_width = min(top[2], bottom[2]) * 0.5
        if abs(top[2] - bottom[2]) > half_width or abs(top[0] - bottom[0]) > half_width:
            return cls.merge_with_two_sticky_color_bar(top, bottom)

        # sort according to y
        if top[1] > bottom[1]:
            top, bottom = bottom, top

        if top[0] == bottom[0] and top[2] == bottom[2] and top[3] > top[2] * 2.4 and bottom[3] > bottom[2] * 2.4 and \
                top[1] + top[3] < bottom[1] + 4:
            pass
        else:
            if top[3] < top[2] * 3 and bottom[3] < bottom[3] * 3 and top[2] < 5 and bottom[2] < 5:
                return

            thresh = 8
            if top[3] > top[2] * 5 or bottom[3] > bottom[2] * 5:
                thresh = 15
            elif top[3] > top[2] * 4 or bottom[3] > bottom[2] * 4:
                thresh = 12

            if top[1] + top[3] > bottom[1] + 2 or bottom[1] > top[1] + top[3] + thresh:
                return

        x = min(top[0], bottom[0])
        y = min(top[1], bottom[1])
        width = max(top[0] + top[2], bottom[0] + bottom[2]) - x
        height = max(top[1] + top[3], bottom[1] + bottom[3]) - y

        return [x, y, width, height]

    @classmethod
    def merge_with_two_sticky_color_bar(cls, top, bottom):

        half_width = min(top[2], bottom[2]) * 0.5

        left = top
        right = bottom
        if left[0] > right[0]:
            left = bottom
            right = top

        # width: double ratio, align with right
        if abs(left[2] - right[2] * 2) > half_width or abs(left[0] + left[2] - right[0] - right[2]) > half_width:
            return

        if top[1] > bottom[1]:
            temp = top
            top = bottom
            bottom = temp

        if top[1] + top[3] > bottom[1] + 2:
            # intersection
            if abs(bottom[1] - top[1] - bottom[3]) > 5:
                return
        elif bottom[1] - top[1] - top[3] > 8:
            return

        x = min(top[0], bottom[0])
        y = min(top[1], bottom[1])
        width = max(top[0] + top[2], bottom[0] + bottom[2]) - x
        height = max(top[1] + top[3], bottom[1] + bottom[3]) - y

        return [x, y, width, height]

    @classmethod
    def merge_vertical_bbox(cls, bbox1, bbox2, height, width):
        # 垂直合并
        x1, y1, w1, h1, center_x1, center_y1 = bbox1
        x2, y2, w2, h2, center_x2, center_y2 = bbox2
        min_width = min(w1, w2)
        if bbox1[1] > height * 0.4 and width * 0.3 < bbox1[0] < width * 0.7 or \
                bbox2[1] > height * 0.4 and width * 0.3 < bbox2[0] < width * 0.7:
            return None

        if abs(w1 - w2) > 4 or abs(center_x1 - center_x2) > min_width // 4 or \
                abs(center_y1 - center_y2) < (h1 + h2) // 2:
            # 宽度应该基本一致，上下基本垂直，不合并水平的盒子
            return None

        if center_y1 < center_y2:
            top_bbox = bbox1
            bottom_bbox = bbox2
        else:
            top_bbox = bbox2
            bottom_bbox = bbox1

        x1, y1, w1, h1, center_x1, center_y1 = top_bbox
        x2, y2, w2, h2, center_x2, center_y2 = bottom_bbox

        merged_bbox = [x1, y1, max(w1, w2), y2 - y1 + h2, top_bbox, bottom_bbox]  # len==6

        return merged_bbox

    @classmethod
    def check_bottom_spectrum(cls, bin_image, gray_image):
        """
        whether is dopler with bottom spectrum
        """
        height, width = bin_image.shape[:2]

        # find the start position where has few points (gap)
        cv2.imshow('bottom roi', bin_image[int(height * 0.4):, :])
        cont = 0
        for i in range(int(height * 0.4), height):
            row = bin_image[i, 10:-10]
            nzs = np.count_nonzero(row)
            if nzs < width * 0.14:
                cont += 1
                if cont > 3 and nzs < width * 0.07:
                    i += 1
                    break
            elif cont > 3:
                break
            else:
                cont = 0

        if i > height * 0.9:
            return False

        # whether has a long strip accross the width
        roi_image = bin_image[i:, 10:-10]
        gray_image = gray_image[i:, 10:-10]
        contours, _ = cv2.findContours(roi_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        bbox_list = []
        small_list = []
        largest_bbox = None
        for contour in contours:
            if len(contour) < 10:
                continue

            bbox = cv2.boundingRect(contour)
            # remove small pieces
            if bbox[3] < height // 10 or bbox[2] < width // 10:
                continue

            if bbox[2] > width * 0.55:
                bbox_list.append(bbox)
                if largest_bbox is None:
                    largest_bbox = bbox
                elif bbox[2] > largest_bbox[2]:
                    largest_bbox = bbox
            else:
                small_list.append(bbox)

        if largest_bbox is not None:
            if largest_bbox[3] > height * 0.56:
                return False

            return cls.is_spectrum(ImageUtility.get_roi_image(roi_image, largest_bbox),
                                   ImageUtility.get_roi_image(gray_image, largest_bbox))

        # whether can be merged into one long strip
        if len(small_list) < 2:
            return False

        # sort according to x
        small_list.sort(key=take_first)

        bbox = small_list[0]
        for s_bbox in small_list[1:]:
            if s_bbox[0] - 4 < bbox[0] + bbox[2] < s_bbox[0] + 4 and abs(s_bbox[3] - bbox[3]) < 5:
                bbox = ImageUtility.merge_bbox(bbox, s_bbox)

        if bbox[2] > width * 0.2:
            return cls.is_spectrum(ImageUtility.get_roi_image(roi_image, bbox),
                                   ImageUtility.get_roi_image(gray_image, bbox))

        return False

    @classmethod
    def is_spectrum(cls, roi_image: np.ndarray, gray_image: np.ndarray):
        nzs = np.count_nonzero(roi_image)
        cv2.imshow('bin without otsu', roi_image)
        ratio = 0.6 if roi_image.size < 20000 else 0.55
        if nzs > roi_image.size * ratio:
            _, roi_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            cv2.imshow('bin with otsu', roi_image)

        kernel = np.ones((3, 3), dtype=np.uint8)
        roi_image = cv2.morphologyEx(roi_image, cv2.MORPH_CLOSE, kernel)
        # cv2.imshow('bottom spectrum', roi_image)

        height, width = roi_image.shape[:2]
        # top
        contours, _ = cv2.findContours(roi_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bbox_list = []
        contour_list = []
        for i, contour in enumerate(contours):
            if len(contour) < 10:
                continue

            bbox = cv2.boundingRect(contour)
            if bbox[2] > width // 2:
                bbox_list.append(bbox)
                contour_list.append(i)

        # should be only one
        if len(bbox_list) != 1:
            return False

        bbox = bbox_list[0]
        contour = contours[contour_list[0]]

        # ////////////////////
        display_image = cv2.cvtColor(roi_image, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(display_image, contours, contour_list[0], (0, 0, 255))
        cv2.imshow('spectrum contour', display_image)
        # ////////////////////////

        # check peak and
        return cls.has_peaks_and_valleys(contour, display_image)

    @classmethod
    def has_peaks_and_valleys(cls, contour, display_image=None):
        """
        """
        if len(contour) < 5:
            return False

        peak_list = []
        valley_list = []

        # smooth via continuous 9 points
        pt_list = [pt[0] for pt in contour]
        for i in range(0, len(pt_list) - 9, 9):
            sub_region = pt_list[i: i + 9]
            peak_list.append(max(sub_region, key=compare_y))
            valley_list.append(min(sub_region, key=compare_y))

        if i < len(pt_list) - 5:
            sub_region = pt_list[i:]
            peak_list.append(max(sub_region, key=compare_y))
            valley_list.append(min(sub_region, key=compare_y))

        #
        extrem_list = []
        # peak_pt_list = []
        # valley_pt_list = []

        peak_list = [peak_list[-1], *peak_list, peak_list[0]]
        valley_list = [valley_list[-1], *valley_list, valley_list[0]]

        peak_or_valley = True
        for i in range(1, len(peak_list) - 1):
            if peak_list[i][1] > peak_list[i - 1][1] and peak_list[i][1] >= peak_list[i + 1][1] or \
                    peak_list[i][1] >= peak_list[i - 1][1] and peak_list[i][1] > peak_list[i + 1][1]:

                # and abs(peak_list[i][0] - extrem_list[-1][0]) > 4
                if len(extrem_list) == 0 or not peak_or_valley and peak_list[i][1] > extrem_list[-1][1] + 6 and \
                        cls.block_dist(peak_list[i], extrem_list[-1]) > 20:
                    extrem_list.append(peak_list[i])
                    # peak_pt_list.append(peak_list[i])
                    peak_or_valley = True
                elif peak_or_valley and peak_list[i][1] > extrem_list[-1][1]:
                    extrem_list[-1] = peak_list[i]
                    # peak_pt_list[-1] = peak_list[i]

            if valley_list[i][1] < valley_list[i - 1][1] and valley_list[i][1] <= valley_list[i + 1][1] or \
                    valley_list[i][1] <= valley_list[i - 1][1] and valley_list[i][1] < valley_list[i + 1][1]:

                # and abs(valley_list[i][0] - extrem_list[-1][0]) > 4
                if len(extrem_list) == 0 or peak_or_valley and valley_list[i][1] < extrem_list[-1][1] - 6 and \
                        cls.block_dist(valley_list[i], extrem_list[-1]) > 20:
                    extrem_list.append(valley_list[i])
                    # valley_pt_list.append(valley_list[i])
                    peak_or_valley = False
                elif not peak_or_valley and valley_list[i][1] < extrem_list[-1][1]:
                    extrem_list[-1] = valley_list[i]
                    # valley_pt_list[-1] = valley_list[i]

        # remove noise
        peak_noise = 0
        noise_tag = [False] * len(extrem_list)
        for i in range(2, len(extrem_list) - 2, 2):
            diff0 = abs(extrem_list[i - 2][1] - extrem_list[i][1])
            diff1 = abs(extrem_list[i + 2][1] - extrem_list[i][1])
            if diff0 > 5 and diff1 > 5 and diff0 + diff1 > 20:
                peak_noise += 1
                noise_tag[i] = True
            # else:
            #     diff0 = extrem_list[i + 2][0] - extrem_list[i][0]
            #     diff1 = extrem_list[i][0] - extrem_list[i - 2][0]
            #     if diff0 * diff1 < 0 or abs(diff0) > 2 * abs(diff1) or abs(diff1) > 2 * abs(diff0):
            #         peak_noise += 1
            #         noise_tag[i] = True

        valley_noise = 0
        for i in range(3, len(extrem_list) - 2, 2):
            diff0 = abs(extrem_list[i - 2][1] - extrem_list[i][1])
            diff1 = abs(extrem_list[i + 2][1] - extrem_list[i][1])
            if diff0 > 5 and diff1 > 5 and diff0 + diff1 > 20:
                valley_noise += 1
                noise_tag[i] = True
            # else:
            #     diff0 = extrem_list[i + 2][0] - extrem_list[i][0]
            #     diff1 = extrem_list[i][0] - extrem_list[i - 2][0]
            #     if diff0 * diff1 < 0 or diff0 > 2 * abs(diff1) or diff1 > 2 * abs(diff0):
            #         valley_noise += 1
            #         noise_tag[i] = True

        if display_image is not None:

            for i, pt in enumerate(extrem_list):
                if noise_tag[i]:
                    color = (0, 255, 255)
                else:
                    color = (255, 0, 0) if i % 2 == 0 else (0, 255, 0)
                cv2.drawMarker(display_image, (pt[0], pt[1]), color, cv2.MARKER_CROSS)

            # for pt in peak_pt_list:
            #     cv2.drawMarker(display_image, (pt[0], pt[1]), (255, 0, 0), cv2.MARKER_CROSS)

            # for pt in valley_pt_list:
            #     cv2.drawMarker(display_image, (pt[0], pt[1]), (0, 255, 0), cv2.MARKER_CROSS)

            cv2.imshow('marker', display_image)

        thresh = max(2, len(extrem_list) // 5)
        if valley_noise > thresh or peak_noise > thresh:
            return False

        total_noise = peak_noise + valley_noise
        if len(extrem_list) > 20 and total_noise > len(extrem_list) * 0.3:
            return False

        if len(extrem_list) - total_noise > 6:  # len(peak_pt_list) > 3 and len(valley_pt_list) > 3:
            print('== new dopler specturm ==')

        # return len(peak_pt_list) > 3 and len(valley_pt_list) > 3
        return len(extrem_list) - total_noise > 6

    @classmethod
    def block_dist(cls, pt0, pt1):
        return abs(pt1[0] - pt0[0]) + abs(pt1[1] - pt0[1])


if __name__ == '__main__':
    import shutil
    import time

    image_dir = r'H:\video problem\陆柏林_62287589856074133_100850\20240522_100850\多普勒'
    image_dir = r'H:\measure-data\dopler'
    for image_name in os.listdir(image_dir):
        if not image_name.endswith('jpg') and not image_name.endswith('png') and not image_name.endswith('jpeg'):
            continue

        # image_name = 'shuangshen.jpg'
        # image_name = '2e386162538d4812b5e593c99a335b92.png'
        # image_name = '7710149ebceb4c5baba9786a0958b993.jpg'
        # image_name = '02a9e631ee3248c38435bb8612381504.jpg'
        # image_name = '1dce2a7ed2eb457abf6c306ce06fe4c0.jpg'
        # image_name = '2feed466ca604a57bd1963d46eb8481d.jpg'
        # image_name = '0d0179b2a3e14e58bdfed08ed2065a33.jpg'
        # image_name = 'cfb8e9fdfd384b5e9c9c4496a92f499b.jpg'
        # image_name = '101c4cc364914954aaa5328c0de6ea5a.jpg'

        print(image_name)

        image_path = os.path.join(image_dir, image_name)
        origin_image = ImageUtility.cv2_imread(image_path)

        start = time.time()
        is_dopler = DoplerDetector.is_dopler_by_color_bar(origin_image)
        print('is_dopler time: ', time.time() - start, is_dopler)

        # cv2.imshow('image', origin_image)

        # start = time.time()
        # bin_image = DoplerDetector.convert_RB_to_bin_image(origin_image, rb_thresh=80, rg_thresh=60)
        # # cv2.imwrite(r'D:\rb.bmp', bin_image)
        # # cv2.imwrite(r'D:\origin.bmp', origin_image)
        # print('RB time: ', time.time() - start)

        # cv2.imshow('original image', origin_image)
        # cv2.imshow('rb dopler image', bin_image)

        # start = time.time()
        # bin_image = DoplerDetector.convert_color_to_bin_image(origin_image, thresh=200)
        # print('color time: ', time.time() - start)

        # cv2.imshow('bin dopler image', bin_image)

        # mask, nzs = DoplerDetector.get_dopler_mask(origin_image, diff_thresh=200, max_dopler_pixels=60 * 60)
        # h, w = origin_image.shape[:2]
        # bin_image = np.zeros((h, w), dtype=np.uint8)

        # bin_image[mask] = 255
        # cv2.imshow('dopler image mask', bin_image)

        cv2.waitKey()
        continue

        # scale
        gray_image, scale = ImageUtility.scale_and_gray(origin_image)
        # binary image
        _, bin_image, _ = ImageUtility.adaptive_binary(gray_image, extra_thresh=0)

        cv2.imshow('color image', origin_image)

        # rois = DoplerDetector.detect_dopler_sector(image, bin_image)
        # if rois is not None:
        #     rois, is_dopler = rois
        #     if rois is not None and len(rois) > 0:
        #         for roi in rois:
        #             cv2.rectangle(origin_image, (roi[0] * scale, roi[1] * scale),
        #                         ((roi[0] + roi[2]) * scale, (roi[1] + roi[3]) * scale), (0, 0, 255))

        #     if is_dopler:
        #         target_path = os.path.join(image_dir, 'dopler')
        #         if not os.path.exists(target_path):
        #             os.makedirs(target_path)
        #         target_path = os.path.join(target_path, image_name)
        #         shutil.move(image_path, target_path)

        # cv2.imshow('orgin image', origin_image)

        if DoplerDetector.is_dopler_by_color_bar(origin_image, scale):
            target_path = os.path.join(image_dir, 'dopler')
            if not os.path.exists(target_path):
                os.makedirs(target_path)
            target_path = os.path.join(target_path, image_name)
            shutil.move(image_path, target_path)

        # if DoplerDetector.is_dopler_with_spectrum(color_image, )

        cv2.waitKey()
        continue
