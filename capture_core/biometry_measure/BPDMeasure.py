import cv2
import numpy as np
import math


class BPDMeasure:
    """
    measurement for BiParietal Diameter (BPD)
    """

    def __init__(self):
        '''constructor'''
        pass

    @classmethod
    def do_measure(cls, gray_image, mask, end_points, measure_mode='hadlock', is_bin_mask=False):
        """
        return BPD line in (start_point, end_point)
        """
        if len(gray_image.shape) > 2:
            gray_image = cv2.cvtColor(gray_image, cv2.COLOR_BGR2GRAY)

        x0, y0 = end_points[0]
        x1, y1 = end_points[1]

        slope = (x1 - x0) / (y1 - y0) if abs(y1 - y0) > 1 else 0
        # bpd_points = end_points
        # bpd_points = [[0, 0], [0, 0]]

        hadlock_bpd_points = [[0, 0], [0, 0]]
        intergrowth_21st_bpd_points = [[0, 0], [0, 0]]

        hadlock_bpd_points = [[0, 0], [0, 0]]
        intergrowth_21st_bpd_points = [[0, 0], [0, 0]]

        # upper
        height, width = gray_image.shape[:2]
        if 0 < y0 < height and 0 < x0 < width:
            roi = cls.get_roi_image(end_points[0], [width, height], 20, True)
            roi_mask = mask[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2]]
            roi_upper = gray_image[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2]]

            # cv2.imshow('roi_mask', roi_mask)
            # cv2.imshow('roi_upper', roi_upper)

            pt_upper = [round(x0 - roi[0]), round(y0 - roi[1])]
            pt = cls.find_upper_point_by_threshold(roi_upper, roi_mask, pt_upper, slope, is_bin_mask)
            # for hadlock
            hadlock_bpd_points[0] = [end_points[0][0] + pt[0] - pt_upper[0], end_points[0][1] + pt[1] - pt_upper[1]]
            # for intergrowth-21st
            if intergrowth_21st_bpd_points[0] == [0, 0]:
                intergrowth_21st_bpd_points[0] = [x0, y0]
            else:
                intergrowth_21st_bpd_points[0][0] = min(intergrowth_21st_bpd_points[0][0], x0)
                intergrowth_21st_bpd_points[0][1] = y0
        # down
        if 0 < y1 < height and 0 < x1 < width:
            roi = cls.get_roi_image(end_points[1], [width, height], 20, False)
            roi_mask = mask[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2]]
            roi_down = gray_image[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2]]

            pt_down = [round(x1 - roi[0]), round(y1 - roi[1])]
            pt = cls.find_down_point_by_threshold(roi_down, roi_mask, pt_down, slope)

            # for hadlock
            hadlock_bpd_points[1] = [end_points[1][0] + pt[0] - pt_down[0], end_points[1][1] + pt[1] - pt_down[1]]
            # for intergrowth-21st
            if intergrowth_21st_bpd_points[1] == [0, 0]:
                intergrowth_21st_bpd_points[1] = [x1, y1]
            else:
                intergrowth_21st_bpd_points[1][0] = max(intergrowth_21st_bpd_points[1][0], x1)
                intergrowth_21st_bpd_points[1][1] = y1

        measure_mode_to_bpd_points = {"hadlock": hadlock_bpd_points, "intergrowth-21st": intergrowth_21st_bpd_points}
        for _, bpd_points in measure_mode_to_bpd_points.items():
            # 两个点都超过
            if bpd_points[0] == [0, 0] and bpd_points[1] == [0, 0]:
                # for hadlock
                print("error:两个点都超出了roi")
                bpd_points[0] = [x0, y0]
                bpd_points[1] = [x1, y1]

            # 一个点超过的情况upper
            if bpd_points[0] == [0, 0] and bpd_points[1] != [0, 0]:
                distance = (math.sqrt((bpd_points[1][0] - x1) ** 2 + (bpd_points[1][1] - y1) ** 2)) / 2
                angle = math.atan(slope)
                sin_value = math.sin(angle)
                cos_value = math.cos(angle)
                delta_x = sin_value * distance
                delta_y = cos_value * distance
                bpd_points[0][0] = x0 + delta_x
                bpd_points[0][1] = y0 + delta_y

            # 一个点超过的情况down
            if bpd_points[0] != [0, 0] and bpd_points[1] == [0, 0]:
                distance = (math.sqrt((bpd_points[0][0] - x0) ** 2 + (bpd_points[0][1] - y0) ** 2)) / 2
                angle = math.atan(slope)
                sin_value = math.sin(angle)
                cos_value = math.cos(angle)
                delta_x = sin_value * distance
                delta_y = cos_value * distance
                bpd_points[1][0] = x1 + delta_x
                bpd_points[1][1] = y1 + delta_y

        return measure_mode_to_bpd_points  # dict
        # return LineAnnotation(bpd_points[0], bpd_points[1])

    @classmethod
    def get_roi_image(cls, pt_center, size, radius=20, is_upper=True):
        left = max(pt_center[0] - radius, 0)
        right = pt_center[0] + radius + 1

        width, height = size
        if right > width:
            right = width

        # add 10 more rows for upper or bottom
        upper = pt_center[1] - radius
        down = pt_center[1] + radius + 1
        if is_upper:
            down += 10
        else:
            upper -= 10

        upper = max(upper, 0)
        if down > height:
            down = height

        return [round(left), round(upper), round(right - left), round(down - upper)]

    @classmethod
    def find_upper_point_by_threshold(cls, image, mask, pt_upper, slope, is_bin_mask=False):
        """
        pt_upper: initial upper point
        """
        height, width = mask.shape[:2]
        # mask[mask <= 100] = 0
        if not is_bin_mask:
            _, mask = cv2.threshold(mask, 100, 255, cv2.THRESH_BINARY)

        _, bin_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask[bin_image <= 10] = 0
        # mask = cv2.bitwise_and(bin_image, mask)

        # find the intersecion point wisth mask
        cur_tag = mask[pt_upper[1], pt_upper[0]]

        pt_result = None
        x = pt_upper[0]
        if cur_tag > 0:
            # upwards
            for row in range(pt_upper[1] - 1, 0, -1):
                x -= slope
                col = round(x) if x > 0 else 0
                if mask[row, col] == 0:
                    pt_result = [x + slope, row + 1]
                    break
        else:
            # downwards
            for row in range(pt_upper[1] + 1, height - 1):
                x += slope
                col = round(x) if x < width else width - 1

                if mask[row, col] > 0:
                    pt_result = [x, row]
                    break

        if pt_result is not None:
            return pt_result

        return cls.find_point_downwards(bin_image, pt_upper, slope)

    @classmethod
    def find_down_point_by_threshold(cls, image, mask, pt_down, slope):

        _, bin_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        bin_image = cv2.morphologyEx(bin_image, cv2.MORPH_CLOSE, kernel)

        return cls.find_point_upwards(bin_image, pt_down, slope)

    @classmethod
    def find_point_upwards(cls, bin_image, pt_start, slope):

        # height, width = bin_image.shape[:2]
        x, y = pt_start
        pre_row = y + 5
        fg_pnts = 0

        edge_points = []
        num_fg_points = []
        for row in range(y - 1, 0, -1):
            x -= slope
            col = round(x)
            if col < 0 or col >= bin_image.shape[1]:
                return pt_start

            cur_tag = bin_image[row, col]
            if cls.is_edge_point(bin_image, row, col) and row < pre_row - 3:
                num_fg_points.append(fg_pnts)
                fg_pnts = 0

                edge_points.append([col, row])
                pre_row = row
            elif cur_tag > 0:
                fg_pnts += 1

        if not edge_points:
            return [round(x + slope * 2), 2] if fg_pnts > max(y - 5, 0) else pt_start
        elif len(edge_points) == 1:
            if num_fg_points[0] < 2:
                return pt_start

            return cls.get_avg_point(bin_image, edge_points[0], 1)

        max_idx = np.argmax(num_fg_points)
        if num_fg_points[max_idx] < 5:
            max_idx = 0

        return cls.get_avg_point(bin_image, edge_points[max_idx])

    @classmethod
    def get_avg_point(cls, image, pos, radius=1):
        height, width = image.shape[:2]
        start_row = pos[1] - radius if pos[1] > radius else 0
        end_row = pos[1] + radius + 1 if pos[1] < height - radius else height

        start_col = pos[0] - radius if pos[0] > radius else 0
        end_col = pos[0] + radius + 1 if pos[0] < width - radius else width

        roi = image[start_row:end_row, start_col:end_col]
        nzs_pos = np.nonzero(roi)

        if nzs_pos is None or len(nzs_pos) == 0:
            return pos

        # average position
        pos = np.average(nzs_pos, axis=-1)
        return [start_col + pos[1], start_row + pos[0]]

    @classmethod
    def find_point_downwards(cls, bin_image, pt_start, slope):
        x, y = pt_start
        pre_row = 0

        edge_points = []
        height, width = bin_image.shape[:2]
        for row in range(y, height - 1):
            col = round(x)
            if col < 0 or col >= width:
                return pt_start

            if cls.is_edge_point(bin_image, row, col) and row > pre_row + 1:
                edge_points.append([col, row])
                pre_row = row
            x += slope

        if len(edge_points) < 2:
            return cls.find_point_upwards(bin_image, pt_start, slope)
        elif len(edge_points) == 2:
            return cls.get_avg_point(bin_image, edge_points[0])
        else:
            cur_flag = bin_image[pt_start[1], pt_start[0]]
            if cur_flag > 0:
                max_idx = 0
                spaces = [edge_points[i + 1][1] - edge_points[i][1] for i in range(1, len(edge_points) - 1, 2)]
                # spaces.insert(0, edge_points[0][1] - pt_start[1])
                if len(edge_points) % 2 == 0:
                    spaces.append(height - 1 - edge_points[-1][1])
                max_idx = np.argmax(spaces)
                if spaces[max_idx] < edge_points[0][1] - pt_start[1]:
                    return edge_points[0]

                return cls.get_avg_point(bin_image, edge_points[max_idx * 2 + 1])
            else:
                if len(edge_points) % 2 != 0:
                    edge_points.append(())
                    return cls.get_avg_point(bin_image, edge_points[0])

                spaces = [edge_points[i + 1][1] - edge_points[i][1] for i in range(0, len(edge_points), 2)]
                max_idx = np.argmax(spaces)
                return cls.get_avg_point(bin_image, edge_points[max_idx * 2])

        return pt_start

    @classmethod
    def is_edge_point(cls, image, row, col, radius=1):
        height, width = image.shape[:2]
        start_row = row - radius if row > radius else 0
        end_row = row + radius + 1 if row < height - radius else height

        start_col = col - radius if col > radius else 0
        end_col = col + radius + 1 if col < width - radius else width

        roi = image[start_row:end_row, start_col:end_col]
        nzs = np.count_nonzero(roi)

        total = (2 * radius + 1) * (2 * radius + 1)
        if total / 4 < nzs <= total * 0.75:
            return True

        return False
