import numpy as np
import cv2
import os


def length_compare(elem):
    return elem[1] - elem[0]


class ImageUtility:

    def __init__(self):
        '''constructor'''
        pass

    # flags: IMREAD_UNCHANGED, IMREAD_COLOR, IMREAD_GRAY
    # color image is stored in B G R format
    @classmethod
    def cv2_imread(cls, image_path, flags=cv2.IMREAD_UNCHANGED):
        """
        flags: IMREAD_UNCHANGED, IMREAD_COLOR, IMREAD_GRAY
        color image is stored in B G R format
        """
        img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), flags=flags)
        return img

    @classmethod
    def cv2_imwrite(cls, image_path, image):
        path, ext = os.path.splitext(image_path)
        cv2.imencode(ext, image)[1].tofile(image_path)

    @classmethod
    def scale_of_image(cls, image):
        # scale adaptive
        # scale
        height, width = image.shape[:2]
        scale = 1
        if width > 2048 or height > 2048:
            scale = 8
        elif width > 1024 and height > 1024 or width > 1120 or height > 1120:
            # elif width > 1024 or height > 1024:
            scale = 4
        elif width > 360 or height > 360:
            # elif width > 320 and height > 256 or height > 320 and width > 256:
            scale = 2

        return scale

    @classmethod
    def scale_image(cls, image):
        # scale adaptive
        scale = cls.scale_of_image(image)

        scaled_image = image
        if scale > 1:
            height, width = image.shape[:2]

            height = height // scale
            width = width // scale
            scaled_image = cv2.resize(image, (width, height))
            # scaled_image = image[::scale, ::scale]

        return scaled_image, scale

    @classmethod
    def scale_and_gray(cls, image):
        """
        docstring
        """

        scaled_image, scale = cls.scale_image(image)

        # gray
        if len(scaled_image.shape) > 2:
            gray_image = cv2.cvtColor(scaled_image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = scaled_image

        return gray_image, scale

    @classmethod
    def adaptive_binary(cls, gray_image, extra_thresh=0):
        """
        docstring
        """
        # scale
        height, width = gray_image.shape[0:2]

        # from matplotlib import pyplot as plt
        # plt.hist(gray_image.flatten(), 128)
        # plt.show()
        # cv2.imshow('binarization', gray_image)

        # histogram
        hist = cv2.calcHist([gray_image], [0], None, [128], [0, 256])

        hist = np.squeeze(hist)
        thresh, low_contrast = cls.get_threshold(hist, width * height)
        thresh = max(thresh, 5)

        thresh += extra_thresh

        ret, bin_image = cv2.threshold(gray_image, thresh, 255, cv2.THRESH_BINARY)

        return thresh, bin_image, low_contrast

    @classmethod
    def get_threshold(cls, hist, total_pixels):
        '''
        return value: (threshold, is_low_contrast)
        '''
        sub_hist = hist[0:50]
        max_idx = np.argmax(sub_hist)

        # case 1: # of pixels of each gray level are almost the same
        #         i.e. either images with equalized hist or images by camera
        if sub_hist[max_idx] < total_pixels * 0.04:
            return cls.thresh_of_uniform_distrib(sub_hist, max_idx, total_pixels)
            # if thresh > 90 and sum(hist[36:90]) > total_pixels * 0.8:
            #     thresh = 71
        elif sub_hist[max_idx] < total_pixels * 0.08 and max_idx > 20:
            # if max_idx < 25 and sum(sub_hist[:max_idx + 1]) < total_pixels * 0.1:
            #     return 2 * (max_idx + 1) + 1, True
            max_idx = np.argmax(sub_hist[0:20])
            if sub_hist[max_idx] == 0:
                # find the first element not equal to 0
                for i in range(20, 30):
                    if sub_hist[i] > 0:
                        max_idx = i
                        break
                if i >= 30:
                    max_idx = i
                else:
                    max_idx += np.argmax(sub_hist[max_idx: max_idx + 3])
            return 2 * (max_idx + 1) + 1, True

        # case 2: # of bg pixels ( < max_idx) > 40% of total pixels
        #         i.e. has many bg pixels
        black_pts = sum(sub_hist[:max_idx + 1])
        if black_pts > total_pixels * 0.5 and sub_hist[max_idx + 1] < min(total_pixels * 0.07, 4000):
            return 2 * max_idx + 1, False
        if black_pts > total_pixels * 0.4 and max_idx > 4:
            if sub_hist[max_idx + 2] > sub_hist[max_idx + 1] * 1.5:
                idx = max_idx + 2
                if sub_hist[idx + 1] > sub_hist[idx] * 1.5 \
                        or sub_hist[idx] > total_pixels * 0.08 and sub_hist[idx + 1] > sub_hist[idx]:
                    idx += 1
            else:
                idx = max_idx + 1

            return 2 * idx + 1, False

        low_constrast = False
        thresh = 50
        high_thresh = total_pixels * 0.1

        # case 3: max count is larger than 10% of total pixels
        if sub_hist[max_idx] >= high_thresh:
            origin_max_idx = max_idx
            if sub_hist[max_idx] < high_thresh * 4:
                for i in range(max_idx + 1, len(sub_hist) - 1):
                    if sub_hist[i] > high_thresh:
                        black_pts += sum(sub_hist[max_idx + 1: i + 1])
                        if max_idx > 4 and black_pts > total_pixels * 0.45:
                            if origin_max_idx < 5 and max_idx >= 10 and black_pts > total_pixels * 0.65:
                                idx = max_idx - 1
                                return 2 * idx + 1, low_constrast
                            if i < max_idx + 3:
                                idx = max_idx + 1
                                return 2 * idx + 1, low_constrast
                            break
                        elif max_idx > 6 and sub_hist[i] + sub_hist[origin_max_idx] > total_pixels * 0.55:
                            if sub_hist[i] > sub_hist[origin_max_idx] * 0.5:
                                return 2 * (i - 1) + 1, False
                        max_idx = i
                        pass
                    elif sub_hist[i] + sub_hist[i + 1] > total_pixels * 0.18:
                        max_idx = i + 1

            if max_idx == 0:
                diff = [abs(sub_hist[i + 1] - sub_hist[i]) for i in range(1, 4)]
                idx = np.argmax(diff) + 1

                high_thresh = max(5000, total_pixels * 0.05)
                low_thresh = max(1000, total_pixels * 0.01)
                if sub_hist[idx] > high_thresh:
                    old_idx = idx
                    for i in range(idx + 1, 5):
                        if sub_hist[i] < low_thresh:
                            break
                        elif sub_hist[i] > high_thresh:
                            idx = old_idx + i + 1
            else:
                idx = max_idx + 1

        else:
            idx = max_idx + 1
            diff = sub_hist[max_idx] * 0.1
            while idx < len(sub_hist):
                if sub_hist[max_idx] - sub_hist[idx] > diff:
                    if idx + 1 < len(sub_hist) and sub_hist[idx + 1] > sub_hist[idx] * 2:
                        idx += 1
                    # thresh = idx * 2 + 1
                    break
                idx += 1

        # maximum of the last three
        if idx < len(sub_hist):
            temp = np.array(sub_hist[idx: min(idx + 3, len(sub_hist))])
            idx0 = np.argmax(temp)
            idx += idx0

            if idx == 3 and idx0 == 2 and sub_hist[3] < sub_hist[4] * 1.2:
                idx = 4

        thresh = idx * 2 + 1

        return thresh, low_constrast

    @classmethod
    def thresh_of_uniform_distrib(cls, sub_hist, max_idx, total_pixels):
        '''
        the gray level is uniformly distributed, i.e. # of each gray level are almost the same
        case 1: images with equalized histogram
        case 2: images captured by camera, the gray level is disturbed
        '''
        low_constrast = False

        total = sum(sub_hist)
        if total < total_pixels * 0.5:
            # many pixels with gray > 50, usually captured by camera
            end_idx = max(max_idx + 6, 25)
            i = 0
            for i, count in enumerate(sub_hist[max_idx + 1: end_idx]):
                if count < sub_hist[max_idx] * 0.4:
                    break
            max_idx += i + 1
        elif max_idx > len(sub_hist) - 6:
            if total > total_pixels * 0.6:
                second_idx = np.argmax(sub_hist[:max_idx])
                if second_idx < 20:
                    if second_idx < 5:
                        second_idx = 5
                    return 2 * second_idx + 1, low_constrast

            idx = int(len(sub_hist) * 0.6) if sub_hist[max_idx] < total_pixels * 0.025 else len(sub_hist) - 6
            return 2 * idx + 1, low_constrast
        elif total < total_pixels * 0.75:
            # only small amount pixels with gray above 50, the image is dark
            total_to_now = sum(sub_hist[0:max_idx + 1])
            # total_to_now = 0
            if total_to_now > total_pixels * 0.35 and max_idx > 20:
                thresh = max(total_pixels // 100, sub_hist[max_idx] // 2)
                for i in range(max_idx - 1, 20, -1):
                    if sub_hist[i] < thresh:
                        max_idx = i
                        break
            else:
                for i, count in enumerate(sub_hist[max_idx + 1:]):
                    total_to_now += count
                    if total_to_now > total_pixels * 0.38:
                        break
                    if count > sub_hist[max_idx] * 0.4:
                        continue

                    if total_to_now < sub_hist[max_idx] and i < 5:
                        continue
                    break
                max_idx += i + 2
        elif max_idx > 15:
            thresh = total_pixels * 0.05
            total_to_now = 0
            for i in range(15):
                total_to_now += sub_hist[i]
                if total_to_now > thresh:
                    max_idx = i - 1
                    break
            if max_idx > 15:
                max_idx = 15

            low_constrast = True
        else:
            max_idx += 2

        print('uniform thresh', 2 * max_idx + 1)
        return 2 * max_idx + 1, low_constrast

    @classmethod
    def detect_horizontal_black_zone(cls, bin_image, min_zone_width=4,
                                     merge_adjacent=False, sort_by_gap=True, nzs_thresh=5):
        """
        docstring
        """
        # open
        kernel = np.ones((3, 3), np.uint8)
        bin_image = cv2.morphologyEx(bin_image, cv2.MORPH_OPEN, kernel)

        # cv2.imshow('horzontal zone', bin_image)

        height, width = bin_image.shape[: 2]

        # list of black zone such as [[start, end], [start, end]]
        black_list = []
        prev_y = -1
        for i in range(height):
            row = bin_image[i, 5:-5]
            nzs = np.count_nonzero(row)
            # pos = np.where(row > 0)[0]
            if nzs > nzs_thresh:
                if prev_y >= 0 and i - prev_y >= min_zone_width:
                    # merge with previous
                    if merge_adjacent and len(black_list) > 0 and prev_y - black_list[-1][1] < 5:
                        black_list[-1][1] = i
                    else:
                        black_list.append([prev_y, i])
                prev_y = -1
                continue
            if prev_y < 0:
                prev_y = i

        if prev_y > 0:
            # merge with previous
            if merge_adjacent and len(black_list) > 0 and prev_y - black_list[-1][1] < 5:
                black_list[-1][1] = height - 1
            elif height - prev_y >= min_zone_width:
                black_list.append([prev_y, height - 1])

        # find largest
        if len(black_list) == 0:
            return

        # return one with largest gap
        sort_by_gap and black_list.sort(key=length_compare, reverse=True)
        return black_list[0]

    @classmethod
    def detect_vertical_black_zone(cls, bin_image, min_zone_width=4, black_line_thresh=5):
        """
        docstring
        """
        height, width = bin_image.shape[: 2]

        black_list = []
        prev_x = -1
        # count = []
        for i in range(width):
            col = bin_image[:, i]
            # pos = np.where(col > 0)[0]
            nzs = np.count_nonzero(col)
            # count.append(len(pos))
            if nzs > black_line_thresh:
                # only larger than specified width is accounted
                if prev_x >= 0 and i - prev_x >= min_zone_width:
                    black_list.append([prev_x, i])
                prev_x = -1
                continue
            if prev_x < 0:
                prev_x = i

        if prev_x >= 0 and width - prev_x >= min_zone_width:
            black_list.append([prev_x, width])

        black_list.sort(key=length_compare, reverse=True)
        return black_list

    @classmethod
    def preprocess(cls, image, annotations, image_size=380):
        image_height, image_width = image.shape[:2]
        if image_height > image_width:
            scale = image_size / image_height
            resized_height = image_size
            resized_width = int(image_width * scale)
        else:
            scale = image_size / image_width
            resized_height = int(image_height * scale)
            resized_width = image_size

        # perform same action for annotations

        image = cv2.resize(image, (resized_width, resized_height))

        # dtype = np.uint8
        dtype = np.float32
        new_image = np.ones((image_size, image_size, 3), dtype=dtype) * 128
        offset_h = (image_size - resized_height) // 2
        offset_w = (image_size - resized_width) // 2
        new_image[offset_h:offset_h + resized_height, offset_w:offset_w + resized_width] = image.astype(dtype)

        if dtype == np.float32:
            new_image /= 255.

        return image, annotations

    @classmethod
    def max_min_value_filter(cls, image, ksize=3, mode='max'):
        rows, cols = image.shape[:2]
        # convert to gray
        if len(image.shape) > 2 and image.shape[2] > 1:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # else:
        #     image = image.copy()

        padding = (ksize - 1) // 2
        new_image = cv2.copyMakeBorder(image, padding, padding, padding, padding, cv2.BORDER_REPLICATE)

        if mode == 'max':
            for i in range(rows):
                for j in range(cols):
                    roi = new_image[i: i + ksize, j: j + ksize]
                    min_val, max_val, min_idx, max_idx = cv2.minMaxLoc(roi)
                    image[i, j] = max_val
        else:
            for i in range(rows):
                for j in range(cols):
                    roi = new_image[i: i + ksize, j: j + ksize]
                    min_val, max_val, min_idx, max_idx = cv2.minMaxLoc(roi)
                    image[i, j] = min_val

        return image

    @classmethod
    def convert_to_bin_image(cls, image, thresh=90, exclude_red=False):
        image = image.astype(np.int16)
        bin_image = abs(image[:, :, 1] - image[:, :, 0]) + \
            abs(image[:, :, 2] - image[:, :, 0]) + \
            abs(image[:, :, 2] - image[:, :, 1])

        # bin_image = bin_image // 3

        _, bin_image = cv2.threshold(bin_image, thresh, 255, cv2.THRESH_BINARY)
        bin_image = bin_image.astype(np.uint8)
        # cv2.imshow('color bar', bin_image)

        if exclude_red:
            pos = image[:, :, 2] > image[:, :, 0] + 50
            bin_image[pos] = 0
            image[:, :, 2] > image[:, :, 1] + 50
            bin_image[pos] = 0

        return bin_image

    @classmethod
    def convert_green_to_bin(cls, color_image, thresh=50):

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
    def convert_blue_major_to_bin(cls, color_image, thresh=25):
        color_image = color_image.astype(np.int16)
        diff01 = color_image[:, :, 0] - color_image[:, :, 1]
        diff02 = color_image[:, :, 0] - color_image[:, :, 2]

        diff = diff01 + diff02
        _, bin_image = cv2.threshold(diff, thresh, 255, cv2.THRESH_BINARY)
        bin_image = bin_image.astype(np.uint8)

        bin_image[diff01 < 5] = 0
        bin_image[diff02 < 5] = 0

        return bin_image

    @classmethod
    def vert_edge_of_bin_image(cls, bin_image):
        height, width = bin_image.shape[:2]

        bin_image = bin_image.astype(np.int16)
        edge = bin_image[:, 1:] - bin_image[:, 0:width - 1]

        edge = abs(edge)
        edge = edge.astype(np.uint8)

        first_col = np.zeros(height)
        edge = np.insert(edge, 0, first_col, axis=1)

        return edge

    @classmethod
    def get_roi_image(cls, image, roi):
        """
        roi: [x, y, width, height]
        """
        return image[roi[1]: roi[1] + roi[3], roi[0]: roi[0] + roi[2]]

    @classmethod
    def merge_bbox(cls, bbox0, bbox1):
        """
        bbox0, bbox1: [x, y, width, height]
        """
        x0 = min(bbox0[0], bbox1[0])
        y0 = min(bbox0[1], bbox1[1])

        x1 = max(bbox0[0] + bbox0[2], bbox1[0] + bbox1[2])
        y1 = max(bbox0[1] + bbox0[3], bbox1[1] + bbox1[3])

        return [x0, y0, x1 - x0, y1 - y0]


if __name__ == '__main__':
    image_dir = r'C:\Users\guang\Desktop\roi\error'
    image_dir = r'C:\Users\guang\Desktop\roi\one'

    for image_name in os.listdir(image_dir):
        if not image_name.endswith('jpg') and not image_name.endswith('png') and not image_name.endswith('jpeg'):
            continue

        image_name = '1594732906.jpg'
        image_name = '1723532e11914bd9b0cbb96d666126e9.jpg'
        image_name = '24e429dec72b41cbb4421894992e123f.jpg'
        image_name = '424095.jpg'
        image_name = '42b28f0a889b46989988b0103d6f0c4e.jpg'
        image_name = '43b823febb29420da60a684d7b3fea96.jpg'
        image_name = '4c547fc9f02b42c7984161a5f58a461d.jpg'
        image_name = '4f73bfeab51a42d6bdd3435f4bf8af9a.jpg'
        image_name = '5a4456404abc40a1aeda2cc537b9d18a.jpg'
        # image_name = '7425fb9b15df4f4cb5c8795fb50dae97.jpg'
        # image_name = '7fe1d9c3113a442c84c406e603e979e6.jpg'
        # image_name = '80cdfd3be1644ec3be5434f93bbf4b3b.jpg'
        image_name = '3f9cbace16b1411db57a3c84e765af44.jpg'

        image_path = os.path.join(image_dir, image_name)
        image = ImageUtility.cv2_imread(image_path)

        cv2.imshow('original image', image)

        thresh = 49
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, bin_image = cv2.threshold(gray_image, thresh, 255, cv2.THRESH_BINARY)

        edge = ImageUtility.vert_edge_of_bin_image(bin_image)
        cv2.imshow('bin image', bin_image)
        cv2.imshow('edge', edge)

        # gray_image, scale = ImageUtility.scale_and_gray(image)
        # cv2.imshow('gray image', gray_image)

        # ImageUtility.convert_green_to_bin(image)

        # gray_image = cv2.medianBlur(gray_image, 5)
        # cv2.imshow('meidan blur', gray_image)

        # image = ImageUtility.max_min_value_filter(image, ksize=5, mode='min')

        # cv2.imshow('max filter', image)

        cv2.waitKey()
