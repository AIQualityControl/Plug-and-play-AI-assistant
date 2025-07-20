#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@description:
@date       : 2023/09/09 14:29:27
@author     : Guanghua Tan
@email      : guanghuatan@hnu.edu.cn
@version    : 1.0
'''

import cv2
from collections import deque
from common.model.image_info import ImageInfo
import numpy as np
from loguru import logger
from skimage.metrics import structural_similarity as ssim


class HistoryQueue:

    CMP_INTERVAL = 2
    CMP_INTERVAL2 = 15

    def __init__(self, maxlen=60):
        '''constructor'''
        self.queue = deque(maxlen=maxlen)

        # store all image info with std plane
        self.std_info_list = deque(maxlen=maxlen)

        self.last_std_info = None

        self.release_image = True

        self.still_num = 0

    def __len__(self):
        return len(self.queue)

    def __getitem__(self, idx):
        assert (idx < len(self.queue))
        return self.queue[idx]

    def plane_type_list(self, max_frames=-1):
        """
        return all auto_type if max_frames <= 0
        """
        # latest max_frames
        max_frames = len(self.queue) if max_frames <= 0 else min(max_frames, len(self.queue))
        reuslts = [self.queue[-i].auto_type for i in range(max_frames, 0, -1)]

        return reuslts

    def enqueue(self, image_info: ImageInfo, keep_frame_idx=-1):
        # ignore frames captured by dicom
        if image_info.image_dicom or image_info.is_need_handle_replace:
            return

        # image is not needed for last image_info, release the memory
        if len(self.queue) > self.CMP_INTERVAL2:
            for i in range(self.CMP_INTERVAL2, len(self.queue)):
                prev_info = self.queue[-i]
                # in-use
                if keep_frame_idx > 0 and prev_info.frame_idx >= keep_frame_idx:
                    continue
                # already released
                if prev_info.image is None:
                    break
                # release the image memory
                if not prev_info.need_measure and self.release_image:
                    prev_info.image = None

                if hasattr(prev_info, 'gray_image'):
                    delattr(prev_info, 'gray_image')

        self.queue.append(image_info)

    def enqueue_and_reset_image(self, image_info):
        # ignore frames captured by dicom
        if image_info.image_dicom:
            return

        image_info.image = None
        self.queue.append(image_info)

    def dequeue(self):
        return self.queue.popleft()

    def clear(self):
        self.queue.clear()
        self.std_info_list.clear()
        self.last_std_info = None

    def latest_info(self):
        self.queue[-1] if len(self.queue) > 0 else None

    def plane_type_count(self, plane_type=None):
        """
        if plane_type is not specified, plane type of the latest frame is used
        """
        length = len(self.queue)
        if length < 2:
            return 0

        if plane_type is None:
            plane_type = self.queue[-1].auto_type
            length -= 1

        count = 0
        for i in range(length):
            if self.queue[i].auto_type == plane_type:
                count += 1

        return count

    def get_prev_image_info(self, frame_idx):
        # 前面已经抓了一张标准切面
        if self.last_std_info and self.last_std_info.ruler_info and self.last_std_info.frame_idx > frame_idx:
            return self.last_std_info

        if len(self.queue) < 4 or frame_idx < 0:
            return

        start_idx = len(self.queue) - self.CMP_INTERVAL2
        if start_idx < 0:
            start_idx = 0
        if self.queue[start_idx].frame_idx >= frame_idx:
            return self.queue[start_idx]
        # do
        for i in range(start_idx + 1, len(self.queue)):
            if self.queue[i].frame_idx > frame_idx:
                return self.queue[i - 1]

        return self.queue[start_idx]

    def add_std_info(self):
        """
        the last frame is a std plane
        """
        if len(self.queue) == 0:
            return

        info = self.queue[-1]
        if len(self.std_info_list) > 0 and self.std_info_list[-1].auto_type == info.auto_type:
            info.std_count = self.std_info_list[-1].std_count + 1
            self.std_info_list[-1] = info
        else:
            info.std_count = 1
            self.std_info_list.append(info)
        self.last_std_info = info

    def check_still_video(self, cnt_still_frames=2):
        """
        whether is still: 2 continous frames are all still
        """
        if len(self.queue) < cnt_still_frames + 1:
            return False

        # compare with last_image_info
        # if self.is_same_frame(self.history_info[-1], last_image_info):
        #     return True

        for i in range(1, cnt_still_frames + 1):
            if not self.queue[-i].is_still:
                return False

        return True

    @logger.catch
    def check_still_frame(self):
        """
        compare current frame with previous CMP_INTERVAL frame
        """
        if len(self.queue) <= self.CMP_INTERVAL:
            return False

        try:
            prev_info = self.queue[-self.CMP_INTERVAL - 1]
            # whether to check frame idx
            thresh_ratio = 0.005
            is_still = self.is_same_frame(self.queue[-1], prev_info, thresh_ratio)

            # thresh_ratio = 0.98 if self.still_num > self.CMP_INTERVAL else 0.96
            # is_still = self.is_same_frame(self.queue[-1], prev_info, thresh_ratio)

            self.still_num = self.still_num + 1 if is_still else 0

            self.queue[-1].is_still = is_still
        except Exception:
            logger.error('history queue has been cleared before get last one by detection thread')
            return False

        # not need any more
        if hasattr(prev_info, 'gray_image'):
            delattr(prev_info, 'gray_image')

        return is_still

    @classmethod
    def is_same_frame(cls, cur_image_info: ImageInfo, last_image_info: ImageInfo, thresh_ratio=0.005):

        # type should be same
        cur_auto_type = cur_image_info.auto_type
        last_auto_type = last_image_info.auto_type
        if cur_auto_type not in (0, 1, -1) and last_auto_type not in (0, 1, -1):
            cur_auto_score = cur_image_info.auto_score
            last_auto_score = last_image_info.auto_score
            if abs(cur_auto_type) != abs(last_auto_type) or abs(cur_auto_score - last_auto_score) > 20:
                return False

        if hasattr(last_image_info, 'gray_image'):
            last_gray_image = last_image_info.gray_image
        elif last_image_info.image is not None:
            roi_image = last_image_info.roi_image()
            if roi_image is None:
                logger.error(f'last image is None: {last_image_info.frame_idx}')
                return False
            last_gray_image = cls.get_gray_image(roi_image)
        else:
            return False

        cur_gray_image = cls.get_gray_image(cur_image_info.roi_image())
        # store gray image which can be used in next frame
        cur_image_info.gray_image = cur_gray_image

        #
        is_still = cls.compare_images(cur_gray_image, last_gray_image, thresh_ratio)

        # 计算 SSIM: do not to return the similarity image
        # ssim_score = ssim(cur_gray_image, last_gray_image)
        # is_still = ssim_score >= thresh_ratio

        return is_still

    @classmethod
    def is_same_frame_ssim(cls, cur_image_info: ImageInfo, last_image_info: ImageInfo, thresh_ratio=0.96):
        # type should be same
        cur_auto_type = cur_image_info.auto_type
        last_auto_type = last_image_info.auto_type
        if cur_auto_type not in (0, 1, -1) and last_auto_type not in (0, 1, -1):
            cur_auto_score = cur_image_info.auto_score
            last_auto_score = last_image_info.auto_score
            if abs(cur_auto_type) != abs(last_auto_type) or abs(cur_auto_score - last_auto_score) > 20:
                return False

        if hasattr(last_image_info, 'gray_image'):
            last_gray_image = last_image_info.gray_image
        else:
            roi_image = last_image_info.roi_image()
            if roi_image is None:
                logger.error(f'last image is None: {last_image_info.frame_idx}')
                return False
            last_gray_image = cls.get_gray_image(roi_image)

        cur_gray_image = cls.get_gray_image(cur_image_info.roi_image())

        #
        # is_still = cls.compare_images(cur_gray_image, last_gray_image, thresh_ratio)

        # 计算 SSIM: do not to return the similarity image
        ssim_score = ssim(cur_gray_image, last_gray_image)
        is_still = ssim_score >= thresh_ratio

        # store gray image which can be used in next frame
        cur_image_info.gray_image = cur_gray_image

        return is_still

    @classmethod
    def compare_images(cls, cur_image, last_image, thresh_ratio=0.005):
        """
        whether is almost the same image
        """
        # shape should be same
        if cur_image.shape != last_image.shape:
            logger.error('image shape should be same')
            return False

        diff = cv2.absdiff(cur_image, last_image)
        # 25
        _, bin_image = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)

        # cv2.imshow('diff', bin_image)

        h, w = bin_image.shape[:2]
        thresh = h * w * thresh_ratio

        nzs = np.count_nonzero(bin_image)
        # 2.5
        if nzs > thresh * 2:
            return False

        if nzs <= thresh:
            return True

        # erode
        # kernel = np.ones((2, 2), dtype=np.uint8)
        # bin_image = cv2.morphologyEx(bin_image, cv2.MORPH_OPEN, kernel)
        # bin_image = cv2.erode(bin_image, kernel)
        bin_image = cv2.medianBlur(bin_image, 3)

        # cv2.imshow('erode', bin_image)

        nzs = np.count_nonzero(bin_image)

        # print(f'{nzs} < {thresh}')

        return nzs < thresh

    @classmethod
    def get_gray_image(cls, image):
        # roi
        h, w = image.shape[:2]
        roi_image = image[h // 5: h * 4 // 5, w // 5: w * 4 // 5]

        # scale to half
        h, w = roi_image.shape[:2]
        scale_image = cv2.resize(roi_image, (w // 2, h // 2))

        gray_image = cv2.cvtColor(scale_image, cv2.COLOR_BGR2GRAY)

        return gray_image

    def judge_now_plane(self, num_cnt_frames=6, max_frames=30):

        # ignore the last one
        detection_plane = 0
        length = len(self.queue)
        if length < num_cnt_frames:
            return detection_plane, 0.9

        plane_type_list = self.plane_type_list(max_frames)
        # 先通过最后num帧判断当前检测切面,不满足就通过所有帧中最多数的帧判断
        detection_plane = plane_type_list[-2]
        if all(plane_type_list[i] == plane_type_list[-2] for i in range(-num_cnt_frames, -1)):
            return detection_plane, 0.9

        # 最多的切面
        from collections import Counter
        most_plane = Counter(plane_type_list).most_common(1)[0]
        if most_plane[1] >= num_cnt_frames:
            detection_plane = most_plane[0]
            return detection_plane, most_plane[1] / len(plane_type_list)

        detection_plane = 0
        return detection_plane, 0.9

    def add_score(self, plane_id, class_score):
        """
        判断某切面的是否满足从非标->标准
        """
        # todo: 更加合理的加分条件

        detection_plane, _ = self.judge_now_plane()

        score_list = [self.queue[i].auto_score for i in range(len(self.queue) - 1)
                      if self.queue[i].auto_type == plane_id]
        same_frames = len(score_list)
        if detection_plane != plane_id:
            if class_score < 0.6:
                return -200
            else:
                return -2 * (6 - same_frames) if same_frames < 6 else -1

        if len(score_list) < 6:
            return -(6 - same_frames)  # 信息太少

        # is_decreasing = all(x <= y for x, y in zip(score_list, score_list[1:]))
        avg_score_list = [np.mean(score_list[i: i + 5])
                          for i in range(0, same_frames, 5) if i != same_frames - 1]
        is_decreasing = all(x <= y for x, y in zip(avg_score_list, avg_score_list[1:])
                            if not (x < 60 and y < 60 or 60 < x < 80 and 60 < y < 80 or x > 80 and y > 80))
        all_empty = True  # all([])时，is_decreasing=False
        for x, y in zip(avg_score_list, avg_score_list[1:]):
            if not (x < 60 and y < 60 or 60 < x < 80 and 60 < y < 80 or x > 80 and y > 80):
                all_empty = False
                break
        if all_empty:
            is_decreasing = False
        # flag = sum(
        #     score_list[i] < 60
        #     and score_list[i - 1] > 60
        #     or score_list[i] < 80
        #     and score_list[i - 1] > 80
        #     for i in range(1, len(score_list))
        # )  # 统计该切面留存帧中存在降级的次数

        if is_decreasing:
            if score_list[-1] < 80:
                return 3 * same_frames / len(self.queue)  # 满足递增
            else:
                return 0.5 + 3 * same_frames / len(self.queue)  # 满足递增，且递增到标准
        else:
            return 0  # 不满足递增
