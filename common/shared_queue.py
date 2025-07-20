#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@description:
@date       : 2022/10/11 16:17:32
@author     : Guanghua Tan
@email      : guanghuatan@hnu.edu.cn
@version    : 1.0
'''

import pickle
import numpy as np
from multiprocessing import shared_memory
from loguru import logger


class SharedQueue:
    def __init__(self, queue_size=16, create=True, image_shape=(480, 640, 3), name="images_buffer", rwlock=None,
                 manual_image_info=False, prepatient_info=False):
        """constructor"""
        self.detection_ready = False
        self.inited = False
        self.display_ready = False

        # the capture type of queue head is manual_capture or amniotic_measure
        self.top_is_manual = False
        self.top_frame_idx = -1

        self.IMAGE_SIZE = int(np.prod(image_shape))
        # allocate at least 1920 x 1080 x 3 to avoid invalid configuration
        min_size = 1920 * 1080 * 3
        if self.IMAGE_SIZE < min_size:
            self.IMAGE_SIZE = min_size

        self.QUEUE_SIZE = queue_size
        self.name = name

        self.rwlock = rwlock
        if self.rwlock is None:
            logger.error(f"No Lock in {self.name}")

        # add extra space for manually measured images
        if manual_image_info:
            queue_size += 1

        # release
        if create:
            self.release_if_exist(name)

        self.sm_images = shared_memory.SharedMemory(name=name, create=create, size=self.IMAGE_SIZE * queue_size)
        self.images_buffer = self.sm_images.buf

        # [0]: current idx to be saved, i.e. tail, [1]: last idx to be used, i.e. head, idx is detecting
        # [2]: total frames,
        # [3]: idx is classifying, [4]: idx is detecting (not used), [5]: idx is measuring (not used)
        # [6]: idx for pre patient
        self.num_ctrl_idx = 7
        self.sm_ctrl_idx = shared_memory.SharedMemory(name=name + "_ctrl_idx", create=create, size=self.num_ctrl_idx)
        self.ctrl_idx = self.sm_ctrl_idx.buf
        if create:
            for i in range(self.num_ctrl_idx):
                self.ctrl_idx[i] = 0

        # [0]: clear_saved_image_info; [1]: check_cont_frames
        # [2]: finish_or_not           [3]: has_load
        # [5]: model loaded successfully
        # [6]: fetal kind index
        self.ctrl_param_2_idx = {
            'clear_saved_image_info': 0,
            'check_cont_frames': 1,
            'finish_or_not': 2,
            'has_load': 3,
            'measure_finished': 4,
            'model_success': 5,
            'fetal_kind': 6,
            'update_detection_results': 7
        }
        ctrl_params_num = len(self.ctrl_param_2_idx)
        self.sm_ctrl_params = shared_memory.SharedMemory(name=name + '_ctrl_params', create=create,
                                                         size=ctrl_params_num)
        self.ctrl_params = self.sm_ctrl_params.buf
        if create:
            for i in range(ctrl_params_num):
                self.ctrl_params[i] = False
            self.ctrl_params[1] = True
            # model_success: default to be True
            self.ctrl_params[5] = True
            self.ctrl_params[6] = 0
            self.ctrl_params[7] = False

        # each annotation should be less than 6k
        self.IMAGE_INFO_SIZE = 6 * 1024
        # add extra space for pre patient info
        if prepatient_info:
            if create:
                self.ctrl_idx[6] = queue_size
            queue_size += 1

        self.sm_image_info = shared_memory.SharedMemory(name=name + '_image_infos', create=create,
                                                        size=self.IMAGE_INFO_SIZE * queue_size)
        self.image_info_buffer = self.sm_image_info.buf

        # length of image_info excluding image
        self.image_info_size_list = shared_memory.ShareableList([0] * queue_size, name=name + "_image_info_size") \
            if create else shared_memory.ShareableList(name=name + "_image_info_size")

    def is_empty(self):
        with self.rwlock.reader_lock:
            empty = self.is_empty_no_block()
        return empty

    def is_empty_no_block(self):
        return self.ctrl_idx[0] == self.ctrl_idx[1] and self.ctrl_idx[2] <= 0

    def is_full(self):
        with self.rwlock.reader_lock:
            full = self.is_full_no_block()
        return full

    def is_full_no_block(self):
        return self.ctrl_idx[0] == self.ctrl_idx[1] and self.ctrl_idx[2] > 0

    def queue_length(self):
        with self.rwlock.reader_lock:
            length = self.queue_length_no_block()
        return length

    def queue_length_no_block(self):
        if self.is_full_no_block():
            return self.QUEUE_SIZE

        length = self.ctrl_idx[0] - self.ctrl_idx[1]
        if length < 0:
            length += self.QUEUE_SIZE

        return length

    def clear(self):
        with self.rwlock.writer_lock:
            # do not reset idx for pre patient: 6
            for i in range(self.num_ctrl_idx):
                if i != 6:
                    self.ctrl_idx[i] = 0

    def next_idx(self, idx):
        return (idx + 1) % self.QUEUE_SIZE

    def prev_idx(self, idx):
        return idx - 1 if idx > 0 else self.QUEUE_SIZE - 1

    def add_frame(self, image_info):
        """
        replace the last when queue is full
        """
        with self.rwlock.writer_lock:
            logger.debug(f'{self.name}-b: [{self.ctrl_idx[0]}, {self.ctrl_idx[1]}, {self.ctrl_idx[2]}, '
                         f'{self.ctrl_idx[3]}]  frame_idx: {image_info.frame_idx}')

            self._add_frame(image_info)

            logger.debug(f'{self.name}-a: [{self.ctrl_idx[0]}, {self.ctrl_idx[1]}, {self.ctrl_idx[2]}, '
                         f'{self.ctrl_idx[3]}]  frame_idx: {image_info.frame_idx}')

    def _add_frame(self, image_info):
        """
        replace the last when queue is full
        """

        if image_info.image_dicom:
            logger.info(f'{self.name}: dicom影像加入了image_queue队列， image_uid: {image_info.image_uid}, '
                        f'frame_idx:{image_info.frame_idx}')

        if self.ctrl_idx[1] < 2 and self.ctrl_idx[0] > 2 and not self.detection_ready:
            # at least one image should be finished all processes
            logger.info(f'--- {self.name}: detection process is not ready, ignore frame {image_info.frame_idx}---')
            return

        if self.ctrl_idx[1] > 1:
            self.detection_ready = True

        # QUAD_SIZE = self.QUEUE_SIZE // 4
        HALF_SIZE = self.QUEUE_SIZE // 2
        # when manual_captured image is got or full,
        # remove QUEUE_SIZE/4 elements to avoid to delay too long or discard

        prompt_info = ''
        replace_with_previous = False
        # if self.ctrl_idx[2] > HALF_SIZE and image_info.is_manual_or_measure():
        #     # in order to response manual captured image as soon as possible
        #     skip_count = QUAD_SIZE
        #     prompt_info = 'manual_captured image is got'

        is_manual = image_info.is_manual_or_measure()
        is_full = self.is_full_no_block()
        if is_full or self.ctrl_idx[2] > HALF_SIZE and is_manual:
            front_info = self.get_image_info_no_block(self.ctrl_idx[1])
            if front_info and front_info.is_manual_or_measure():
                # replace with last frame
                replace_with_previous = 0 < image_info.frame_idx - front_info.frame_idx <= self.QUEUE_SIZE
                if self.top_is_manual:
                    logger.info(f'{self.name}: both front {front_info.frame_idx} and top {self.top_frame_idx} '
                                f'are manual-captured image, ignore the current frame {image_info.frame_idx}, '
                                f'queue length: {self.ctrl_idx[2]}, is manual: {is_manual}')
                    return
            else:
                skip_frame_list = [front_info.frame_idx]
                # skip 2 frames
                skip_count = 1
                old_idx = self.ctrl_idx[1]

                next_idx = self.next_idx(self.ctrl_idx[1])
                info = self.get_image_info_no_block(next_idx)
                if not info or not info.is_manual_or_measure():
                    skip_count += 1
                    skip_frame_list.append(info.frame_idx)

                self.skip_frames(skip_count)

                # ////////// for logger
                if is_manual:
                    logger.info(f'{self.name}: manual_captured image is got for frame {image_info.frame_idx}, '
                                f'[{self.ctrl_idx[0]}, {self.ctrl_idx[1]}, {self.ctrl_idx[2]}, {self.ctrl_idx[3]}], '
                                f'skip frame: {skip_frame_list} from [{old_idx}, {self.ctrl_idx[1]})')
                else:
                    logger.debug(f'{self.name}: detection queue is full for frame {image_info.frame_idx}, '
                                 f'[{self.ctrl_idx[0]}, {self.ctrl_idx[1]}, {self.ctrl_idx[2]}, {self.ctrl_idx[3]}], '
                                 f'skip frame: {skip_frame_list} from [{old_idx}, {self.ctrl_idx[1]})')

        delta = self.ctrl_idx[0] - self.ctrl_idx[3]
        if delta < 0:
            delta += self.QUEUE_SIZE

        # too many frames to classify: replace with previous frame
        if delta >= 4 or replace_with_previous:
            if self.top_is_manual:
                if not is_full and is_manual:
                    self._add_frame_no_block(self.ctrl_idx[0], image_info)

                    # total frames + 1, next idx +
                    self.ctrl_idx[0] = self.next_idx(self.ctrl_idx[0])
                    self.ctrl_idx[2] += 1
                else:
                    logger.info(f'{self.name}: top {self.top_frame_idx} is manual_captured image, '
                                f'ignore the current frame {image_info.frame_idx}, is mannual: {is_manual}')
                    # ignore
                return

            # replace previous one
            idx = self.prev_idx(self.ctrl_idx[0])

            if not prompt_info:
                prompt_info = f'length of classify queue >= 4 for frame {image_info.frame_idx}: ' + \
                              f'[{self.ctrl_idx[0]}, {self.ctrl_idx[1]}, {self.ctrl_idx[2]}, {self.ctrl_idx[3]}]'

            logger.info(f'{self.name}: {prompt_info}, replace previous frame {self.top_frame_idx} '
                        f'with {image_info.frame_idx}, is manual: {is_manual}')

            self._add_frame_no_block(idx, image_info)

        else:
            self._add_frame_no_block(self.ctrl_idx[0], image_info)

            # total frames + 1, next idx +
            self.ctrl_idx[0] = self.next_idx(self.ctrl_idx[0])
            self.ctrl_idx[2] += 1

        self.top_is_manual = is_manual
        self.top_frame_idx = image_info.frame_idx

        self.inited = True

    def skip_frames(self, skip_count):
        # detection idx
        self.ctrl_idx[1] = (self.ctrl_idx[1] + skip_count) % self.QUEUE_SIZE

        # classification idx
        if self.ctrl_idx[3] < self.ctrl_idx[1] < self.ctrl_idx[0] or \
                self.ctrl_idx[0] < self.ctrl_idx[3] < self.ctrl_idx[1] or \
                self.ctrl_idx[1] < self.ctrl_idx[0] < self.ctrl_idx[3]:
            self.ctrl_idx[3] = self.ctrl_idx[1]

        # total
        size = self.ctrl_idx[0] - self.ctrl_idx[1]
        self.ctrl_idx[2] = size if size >= 0 else size + self.QUEUE_SIZE

    def update_image_info(self, image_info):
        with self.rwlock.writer_lock:
            image = image_info.image

            # no need to update image
            image_info.image = None
            self._add_frame_no_block(image_info.queue_idx, image_info)

            # restore image
            image_info.image = image

    @logger.catch
    def add_display_frame(self, image_info, ignore_image=True):
        """
        pop front when queue is full
        """
        with self.rwlock.writer_lock:
            logger.debug(f'{self.name}-b: [{self.ctrl_idx[0]}, {self.ctrl_idx[1]}, {self.ctrl_idx[2]}, '
                         f'{self.ctrl_idx[3]}]')

            # do not pass image: save image and set to None
            image = image_info.image
            image_shape = image_info.image_shape
            if ignore_image:
                image_info.image = None
                image_info.image_shape = None

            self._add_display_frame(image_info)

            # restore image
            image_info.image = image
            image_info.image_shape = image_shape

            logger.debug(f'{self.name}-a: [{self.ctrl_idx[0]}, {self.ctrl_idx[1]}, {self.ctrl_idx[2]}, '
                         f'{self.ctrl_idx[3]}]  frame_idx: {image_info.frame_idx}')

    def _add_display_frame(self, image_info):
        """
        pop front when queue is full
        """
        # 合并检测结果和测量结果
        if self.ctrl_idx[2] > 1 and image_info.need_measure:
            queue_idx = self.ctrl_idx[1]
            while queue_idx != self.ctrl_idx[0]:
                pre_image_info = self.get_image_info_no_block(queue_idx)
                if pre_image_info.frame_idx == image_info.frame_idx:
                    # replace, no need to add image
                    frame = image_info.image
                    image_info.image = None

                    # 合并后不能当作测量结果对待，而需当作检测结果对待
                    image_info.need_measure = False
                    self._add_frame_no_block(queue_idx, image_info)
                    image_info.image = frame

                    logger.info(f'{self.name}: merge detection with measure reusults for frame {image_info.frame_idx}')
                    return
                if pre_image_info.frame_idx > image_info.frame_idx:
                    break

                queue_idx = self.next_idx(queue_idx)

        if self.is_full_no_block():
            self.add_display_frame_for_full(image_info)
            return

        self._add_frame_no_block(self.ctrl_idx[0], image_info)

        # total frames + 1, next idx +
        self.ctrl_idx[0] = self.next_idx(self.ctrl_idx[0])
        self.ctrl_idx[2] += 1

    def add_display_frame_for_full(self, image_info):
        # discard one frame
        front_info = self.get_image_info_no_block(self.ctrl_idx[1])
        # 如果队首的image_info不是人工抓取帧或测量帧，直接忽略队首的帧，否则替换最后一帧
        if front_info and front_info.is_manual_or_measure_or_std():

            # 如果当前帧不是人工抓取的帧或者测量帧，忽略当前帧
            if not image_info.is_manual_or_measure_or_std():
                logger.info(f'{self.name}: current frame {image_info.frame_idx} is discard since '
                            f'front frame {front_info.frame_idx} is manual or has measure results')
                return

            pre_idx = self.prev_idx(self.ctrl_idx[0])
            pre_info = self.get_image_info_no_block(pre_idx)
            # 如果前一帧不是人工抓取的帧或者测量帧，覆盖前一帧
            if not pre_info or not pre_info.is_manual_or_measure_or_std():
                # replace
                self._add_frame_no_block(pre_idx, image_info)
                logger.info(f'{self.name}: prev frame {pre_info.frame_idx} is discard since '
                            f'front frame {front_info.frame_idx} and current frame {image_info.frame_idx} '
                            'are both manual or has measure results')
                return

            # 如果前一帧不是测量帧，即new_or_update_mode >= 0，且切面类型一致，只保留最好的一帧
            # if not pre_info.is_manual_or_measure and pre_info.auto_type == image_info.auto_type:
            #     # replace
            #     self._add_frame_no_block(pre_idx, image_info)
            #     logger.info(f'prev image with frame idx {pre_info.frame_idx} is discard since prev and current image '
            #                 f'{image_info.frame_idx} are both std with same type {image_info.auto_type}')
            #     return

            # 如果前一帧也是人工抓取的帧或者测量帧，移动
            # move and replace last
            self.move_and_replace(pre_idx, image_info, pre_info)
            return
        else:
            # pop front
            self.ctrl_idx[1] = self.next_idx(self.ctrl_idx[1])

            self._add_frame_no_block(self.ctrl_idx[0], image_info)
            self.ctrl_idx[0] = self.next_idx(self.ctrl_idx[0])

            # -1 and +1, so total frames are the same
            # self.ctrl_idx[2] -= 1

            logger.info(f'{self.name} is full for frame {image_info.frame_idx} with type {image_info.auto_type}: '
                        f'[{self.ctrl_idx[0]}, {self.ctrl_idx[1]}, {self.ctrl_idx[2]}, {self.ctrl_idx[3]}], '
                        f'discard front frame {front_info.frame_idx}')
            return

    def move_and_replace(self, queue_idx, cur_image_info, last_image_info):
        pre_idx = self.prev_idx(queue_idx)

        image_info_list = [last_image_info]
        update_idx = -1
        list_idx = -1
        while pre_idx != self.ctrl_idx[1]:
            pre_info = self.get_image_info_no_block(pre_idx)
            if not pre_info or not pre_info.is_manual_or_measure_or_std():
                # move
                idx = pre_idx
                for info in image_info_list[::-1]:
                    self._add_frame_no_block(idx, info)
                    idx = self.next_idx(idx)
                # replace
                self._add_frame_no_block(queue_idx, cur_image_info)
                logger.info(f'{self.name}: move and replace for frame {cur_image_info.frame_idx}, '
                            f'discard frame idx: {pre_info.frame_idx}')
                return
            elif image_info_list[-1].new_or_update_mode == 1 and image_info_list[-1].auto_type == pre_info.auto_type:
                # 后面一张图像会替换前一张图像
                update_idx = pre_idx
                list_idx = len(image_info_list) - 1

            pre_idx = self.prev_idx(pre_idx)
            image_info_list.append(pre_info)

        if update_idx >= 0:
            # move
            idx = update_idx
            for info in image_info_list[list_idx::-1]:
                self._add_frame_no_block(idx, info)
                idx = self.next_idx(idx)
            # replace
            self._add_frame_no_block(queue_idx, cur_image_info)
            logger.info(f'{self.name}: move and replace for frame {cur_image_info.frame_idx}, '
                        f'discard updated frame idx: {pre_info.frame_idx}')

            return

        frame_list = [info.frame_idx for info in image_info_list[::-1]]
        logger.warning(f'{self.name}: all frames in {self.name} are manual or update_mode >= 0 or has measure results, ' +
                       f'{frame_list}, '
                       f'discard current frame {cur_image_info.frame_idx} with update mode {cur_image_info.new_or_update_mode}')

    def _add_frame_no_block(self, queue_idx, image_info):
        # image
        image = image_info.image

        if image is not None:
            start = queue_idx * self.IMAGE_SIZE
            buffer = self.images_buffer[start:start + self.IMAGE_SIZE]
            temp = np.ndarray(image.shape, dtype=np.uint8, buffer=buffer)
            temp[:] = image

            # frame idx
            image_info.image_shape = image.shape

        # 不能置为None，因为update_frame_info时，不更新图像
        # else:
        #     image_info.image_shape = None

        image_info.queue_idx = queue_idx
        # set image to be None to avoid time-consuming serialization time
        image_info.image = None
        start = queue_idx * self.IMAGE_INFO_SIZE
        image_info_byte = pickle.dumps(image_info)

        # restore image
        image_info.image = image

        if len(image_info_byte) >= self.IMAGE_INFO_SIZE:
            raise Exception(
                f'{self.name}: image_info with idx {image_info.frame_idx} is too long: {len(image_info_byte)}')

        self.image_info_buffer[start: start + len(image_info_byte)] = image_info_byte
        self.image_info_size_list[queue_idx] = len(image_info_byte)

    def get_classify_frame(self):
        # get the latest frame to classify
        with self.rwlock.writer_lock:
            self.inited and logger.debug(f'{self.name}-b: [{self.ctrl_idx[0]}, {self.ctrl_idx[1]}, '
                                         f'{self.ctrl_idx[2]}, {self.ctrl_idx[3]}]')

            image_info = self._get_classify_frame()

            if image_info is not None:
                self.inited = True
                frame_idx = image_info.frame_idx
            else:
                frame_idx = -100
            self.inited and logger.debug(f'{self.name}-a: [{self.ctrl_idx[0]}, {self.ctrl_idx[1]}, '
                                         f'{self.ctrl_idx[2]}, {self.ctrl_idx[3]}], frame_idx: {frame_idx}')

            return image_info

    def _get_classify_frame(self):
        if self.is_empty_no_block():
            return None

        # get next frame
        idx = self.ctrl_idx[3]
        if self.ctrl_idx[0] == idx:
            # classify_queue is empty
            logger.debug('classify queue is empty')
            return None

        image_info = self.get_image_info_no_block(idx)
        if image_info is None:
            return None

        # image
        image = self.get_image_no_block(idx, image_info.image_shape)
        image_info.image = image

        # update classification queue
        self.ctrl_idx[3] = self.next_idx(idx)

        return image_info

    def get_detection_frame(self):
        with self.rwlock.writer_lock:
            self.inited and logger.debug(f'{self.name}-b: [{self.ctrl_idx[0]}, {self.ctrl_idx[1]}, '
                                         f'{self.ctrl_idx[2]}, {self.ctrl_idx[3]}]')

            image_info = self._get_detection_frame()

            if image_info is not None:
                self.inited = True
                frame_idx = image_info.frame_idx
            else:
                frame_idx = -100
            self.inited and logger.debug(f'{self.name}-a: [{self.ctrl_idx[0]}, {self.ctrl_idx[1]}, '
                                         f'{self.ctrl_idx[2]}, {self.ctrl_idx[3]}], frame_idx: {frame_idx}')

            return image_info

    def _get_detection_frame(self):
        # whether detection queue is empty
        if self.is_empty_no_block():
            self.inited and logger.debug('detection queue is empty')
            return None

        delta = self.ctrl_idx[3] - self.ctrl_idx[1]
        if delta == 0 and not self.is_full_no_block():
            logger.debug('No frame is ready for detection')
            return None

        queue_idx = self.ctrl_idx[1]
        if delta < 0:
            delta += self.QUEUE_SIZE

        image_info = self.get_image_info_no_block(queue_idx)
        if image_info is None:
            return None

        # only pop out from classification queue when classification has not finished
        # 分类结果为空的时候，说明分类进程尚未完成,'胎儿心动图', '甲状腺'无需分类
        if delta < 3 and not image_info.class_results and image_info.FetalKind is not None and \
                image_info.FetalKind not in ('甲状腺'):
            logger.debug(f'frame {image_info.frame_idx} is classifying but not finished')
            return None

        image = self.get_image_no_block(queue_idx, image_info.image_shape)
        # no need to do deep copy
        if image is not None:
            image_info.image = image.copy()
        else:
            logger.error(f"{self.name}: failed to get image {image_info.frame_idx} from detection queue")

        # update detection index
        self.ctrl_idx[1] = self.next_idx(queue_idx)

        # queue size
        self.ctrl_idx[2] -= 1
        # size = self.ctrl_idx[0] - self.ctrl_idx[1]
        # self.ctrl_idx[2] = size + self.QUEUE_SIZE if size < 0 else size

        return image_info

    def get_display_frame(self):
        with self.rwlock.writer_lock:
            if self.display_ready:
                logger.debug(f'{self.name}-b: [{self.ctrl_idx[0]}, {self.ctrl_idx[1]}, {self.ctrl_idx[2]}, '
                             f'{self.ctrl_idx[3]}]')

            image_info = self._get_display_frame()

            # do not show log until display queue is ready
            if image_info:
                frame_idx = image_info.frame_idx
                image_info.convert_envelop()
                self.display_ready = True
            else:
                frame_idx = -100

            if self.display_ready:
                logger.debug(f'{self.name}-a: [{self.ctrl_idx[0]}, {self.ctrl_idx[1]}, {self.ctrl_idx[2]}, '
                             f'{self.ctrl_idx[3]}] frame_idx: {frame_idx}')

            return image_info

    def _get_display_frame(self):

        if self.is_empty_no_block():
            return None

        queue_idx = self.ctrl_idx[1]

        # image info
        image_info = self.get_image_info_no_block(queue_idx)

        # deep copy to avoid overwriting
        if image_info is not None and image_info.image_shape is not None:
            image = self.get_image_no_block(queue_idx, image_info.image_shape)
            if image is not None:
                image_info.image = image.copy()

        self.ctrl_idx[1] = self.next_idx(queue_idx)
        self.ctrl_idx[2] -= 1

        return image_info

    def get_image_no_block(self, queue_idx, image_shape):

        if not image_shape:
            return None

        start = queue_idx * self.IMAGE_SIZE
        image_size = int(np.prod(image_shape))
        buffer = self.images_buffer[start:start + image_size]
        image = np.ndarray(image_shape, dtype=np.uint8, buffer=buffer)

        return image

    def get_image_info_no_block(self, queue_idx):
        length = self.image_info_size_list[queue_idx]
        if length == 0:
            return None

        # image info
        start = queue_idx * self.IMAGE_INFO_SIZE
        buffer = self.image_info_buffer[start: start + length]

        try:
            image_info = pickle.loads(buffer)
            return image_info
        except Exception as e:
            logger.error(f'{self.name}:{str(e)}')

        return None

    def set_ctrl_param(self, property, value):
        if property in self.ctrl_param_2_idx:
            with self.rwlock.writer_lock:
                idx = self.ctrl_param_2_idx[property]
                self.ctrl_params[idx] = value

    @logger.catch
    def get_ctrl_param(self, property):
        if property in self.ctrl_param_2_idx:
            with self.rwlock.reader_lock:
                idx = self.ctrl_param_2_idx[property]
                ctrl_param = self.ctrl_params[idx]

                return ctrl_param
        return False

    def update_prepatient_result(self, item):  # 用于存放prepatient_result
        with self.rwlock.writer_lock:
            idx = self.ctrl_idx[6]
            if idx <= 0:
                logger.error(f'{self.name}: No shared memory created for prepatient result')
                return None

            if item is None:
                self.image_info_size_list[idx] = 0
                return False

            try:
                item_str = pickle.dumps(item)
                content_length = len(item_str)

                start = idx * self.IMAGE_INFO_SIZE

                self.image_info_buffer[start: start + content_length] = item_str
                self.image_info_size_list[idx] = content_length

                return True
            except Exception as e:
                logger.error(f'{self.name}:{str(e)}')
                return False

    @logger.catch
    def get_prepatient_result(self):  # 用于获取prepatient_result
        # 改成写锁：检测进程中，主线程会从队列中获取prepatient_reuslt，后处理线程会往队列中写detection_result，读写的位置不同
        # 改成写锁后，两者可以同时读和写（因为该锁为进程锁，只要该进程拿到写锁后，就可以执行任何操作）
        # 需要调用者自己注意 在同一个进程内的两个线程的访问同步
        with self.rwlock.writer_lock:
            queue_idx = self.ctrl_idx[6]
            if queue_idx <= 0:
                logger.error(f'{self.name}: No shared memory created for prepatient result')
                return None

        return self.get_image_info_no_block(queue_idx)

    def close(self):
        """
        only close the shared memory, the shared memory is not destroyed
        """
        logger.info('shared memory is closing ....')

        self.sm_images.close()
        self.sm_image_info.close()
        self.image_info_size_list.shm.close()

        self.sm_ctrl_idx.close()
        self.sm_ctrl_params.close()

    def release(self):
        """
        close and release shared memory, i.e. the shared memory is destroyed
        """

        self.close()

        self.sm_images.unlink()
        self.sm_image_info.unlink()
        self.image_info_size_list.shm.unlink()

        self.sm_ctrl_idx.unlink()
        self.sm_ctrl_params.unlink()

    @staticmethod
    def release_if_exist(name):
        try:
            sm = shared_memory.SharedMemory(name=name, create=False)
            sm.close()
            sm.unlink()
        except Exception:
            pass

        try:
            sm = shared_memory.SharedMemory(name=name + '_image_infos', create=False)
            sm.close()
            sm.unlink()
        except Exception:
            pass

        try:
            sm = shared_memory.SharedMemory(name=name + '_ctrl_params', create=False)
            sm.close()
            sm.unlink()
        except Exception:
            pass

        try:
            sm = shared_memory.SharedMemory(name=name + '_ctrl_idx', create=False)
            sm.close()
            sm.unlink()
        except Exception:
            pass

        # although the following is sharedlist, also use sharedmemory to check whether exist
        try:
            sm = shared_memory.SharedMemory(name=name + '_frame_idx', create=False)
            sm.close()
            sm.unlink()
        except Exception:
            pass

        try:
            sm = shared_memory.SharedMemory(name=name + '_image_info_size', create=False)
            sm.close()
            sm.unlink()
        except Exception:
            pass


if __name__ == '__main__':
    from rwlock import RWLock
    from model.image_info import ImageInfo

    rwlock = RWLock()
    image_queue = SharedQueue(rwlock=rwlock, prepatient_info=True, manual_image_info=True)

    image = np.zeros((1080, 1920, 3), dtype=np.uint8)

    for i in range(5):
        image_info = ImageInfo(image, i)
        image_queue.add_display_frame(image_info)

    for i in range(5):
        image_info = ImageInfo(image, i)
        image_info.need_measure = True
        image_queue.add_display_frame(image_info)

    for i in range(10):
        image_info = image_queue.get_display_frame()
        print(image_info)
