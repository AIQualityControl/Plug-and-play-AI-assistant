import os
import numpy as np
from loguru import logger
import cv2
from ..utility.sub_region_detector.dopler_detector import DoplerDetector
from capture_core.ruler.ruler_recognizer import RulerRecognizer


class QcModel:
    """
    docstring
    """

    # default to use the first gpu card
    def __init__(self, model_file_name, class_mapping_file, config, load_model,
                 gpu_id=0, model_dir=r'/data/QC_python/model/', is_measure=False):
        """
        docstring
        """
        self.model_dir = model_dir
        self.gpu_id = gpu_id
        self.model = None

        self.is_measure = is_measure

        if 'target_width' in config and 'target_height' not in config:
            config['target_height'] = config['target_width']
        elif 'target_height' in config and 'target_width' not in config:
            config['target_width'] = config['target_height']
        self.config = config

        self.class_to_name = {}
        self.name_to_class = {}
        self.parts_weight_mapping = {}

        self.name_to_db_id_map = {}
        self.db_id_to_name_map = {}

        # self.init_weight(parts_weight_mapping)
        self.init_class_mapping(class_mapping_file, model_file_name)
        self.init_model(model_file_name, gpu_id, load_model)

        self.detect_with_roi = True

        self.plane_type = None
        self.plane_id = 0

        self.is_detection_wrong = False
        self.frame_idx = -1

        self.history_queue = None

        self.history_measure_queue = None

    def get_plane_type(self):
        return self.plane_type

    def get_plane_id(self):
        return self.plane_id

    def set_plane_type_and_id(self, plane_type, plane_id):
        self.plane_type, self.plane_id = plane_type, plane_id

    def set_frame_idx(self, frame_idx):
        self.frame_idx = frame_idx

    def set_is_detection_wrong(self, is_detection_wrong=True):
        self.is_detection_wrong = is_detection_wrong

    def set_history_queue(self, history_queue):
        self.history_queue = history_queue

    def set_history_measure_queue(self, history_measure_queue):
        self.history_measure_queue = history_measure_queue

    def get_annotation_name(self, class_id):
        """
        docstring
        """
        if class_id not in self.class_to_name:
            return '其它'
        return self.class_to_name[class_id]

    def get_annotation_id(self, class_name):
        """
        docstring
        """
        if class_name not in self.name_to_class:
            return -1
        return self.name_to_class[class_name]

    def all_annotation_names(self):
        """
        docstring
        """
        return self.name_to_class.keys()

    def is_inited(self):
        """
        docstring
        """
        return self.model is not None

    def set_detect_with_roi(self, detect_with_roi=True):
        """
        docstring
        """
        self.detect_with_roi = detect_with_roi

    def is_detect_with_roi(self):
        """
        docstring
        """
        return self.detect_with_roi

    def init_weight(self, parts_weight_mapping):
        """
        docstring
        """
        totoal_score = sum(parts_weight_mapping.values())

        if totoal_score > 0:
            # normalize to [0, 100]
            coeff = 100 / totoal_score
            for key, score in parts_weight_mapping.items():
                parts_weight_mapping[key] = score * coeff
        self.parts_weight_mapping = parts_weight_mapping

    def init_class_mapping(self, class_mapping_file, model_name=""):
        """
        docstring
        """
        if not self.is_measure and not class_mapping_file:
            # print("class mapping file is not specified")
            logger.warning("class mapping file is not specified: " + model_name)
            return False

        csv_dir = os.path.join(self.model_dir, 'classmapping/')
        class_mapping_path = os.path.join(csv_dir, class_mapping_file)
        if not os.path.exists(class_mapping_path):
            curdir = os.path.abspath('.')
            abs_mapping_path = os.path.join(curdir, 'model_config', 'classmapping', class_mapping_file)
            if not os.path.exists(abs_mapping_path):
                # print("class mapping file does not exist: " + class_mapping_path)
                logger.error(f"class mapping file does not exist: {class_mapping_path} or {abs_mapping_path}")
                return False

        # id -> name and name --> id
        with open(class_mapping_path, 'r', encoding='utf-8') as fs:
            for line in fs:
                items = line.split(',')
                class_id = int(items[1])

                name = items[0].strip()
                self.class_to_name[class_id] = name
                self.name_to_class[name] = class_id

                # todo: add special treatment

        return True

    def init_name_id_map(self, name_to_db_id_map, db_id_to_name_map):
        self.name_to_db_id_map = name_to_db_id_map
        self.db_id_to_name_map = db_id_to_name_map

    def id_of_image_type(self, image_type):
        """
        return the id in db of the image type
        """
        if image_type in self.name_to_db_id_map:
            return self.name_to_db_id_map[image_type]
        return -1

    def image_type_of_id(self, type_id):
        """
        return the name of image type given the type id
        """
        if type_id in self.db_id_to_name_map:
            return self.db_id_to_name_map[type_id]
        return '未知'

    def init_model(self, model_name, gpu_id, load_model):
        """
        docstring
        """
        if load_model:
            model_path = os.path.join(self.model_dir, 'deep_models', model_name)
            if not os.path.exists(model_path):
                # print('model does not exist: ' + model_path)
                curdir = os.path.abspath('.')
                # print(curdir)
                abs_path = os.path.join(curdir, 'model_config', 'deep_models', model_name)
                if not os.path.exists(abs_path):
                    # print('model does not exist: ' + model_path)
                    logger.error(f'model does not exist: {model_path} or {abs_path}')
                    return False

            # specify the gpu
            # import torch
            # if not torch.cuda.is_available():
            #     gpu_id = 'cpu'

            # load model: different model maybe have different loading method, this method should be overided
            # tensorflow and torch use different method to specify gpu id
            backbone = self.config['backbone'] if 'backbone' in self.config else ''

            try:
                self.model = self.load_model(model_path, gpu_id=gpu_id, backbone_name=backbone)
            except Exception as e:
                logger.exception(str(e))
                logger.error('Failed to load ' + model_path)
                return False

            # print(model_path + " are loaded successfully")
            logger.debug(model_path + " are loaded successfully")
        return True

    def clear_model(self):
        del self.model

        import torch
        torch.cuda.empty_cache()

    def load_model(self, model_path, gpu_id, backbone_name):
        """
        docstring
        """
        # return None
        raise NotImplementedError('Please define "a load_model method"')

    def detect(self, image_list, image_info_list=None):
        """
        docstring
        """
        # 1. preprocess -> # 2. detect ->  # 3. postprocess
        raise NotImplementedError('Please define "a detect method"')

    def detect_plane_type(self, parameter_list):
        """
        docstring
        """
        return 0

    # pre_process and postprocess can be load from utils_tgh
    def preprocess(self, raw_image, annotations=None, target_size=(608, 416)):
        raise NotImplementedError('Please define "a preprocess method"')

    def postprocess(self, boxes_batch, scores_batch, labels_batch, scale_batch, offset_batch, score_threshold=0.4,
                    max_detections=10):
        raise NotImplementedError('Please define "a postprocess method"')

    def postprocess_special(self, image_labels, image_scores, image_boxes):
        # get one detection at most for each label
        raise NotImplementedError('Please define "a postprocess_special method"')

    def calculate_qc_score(self, std_info, std_score, parts_found, label_list, score_list):
        return 0

    def confidence_score(self, scores_list, labels_list, boxes_list, std_info, std_score, parts_found,
                         label_name_list):
        return -1

    @classmethod
    def get_std_info(cls, part):
        std_info = 'nstd'
        items = part.split('_')
        if len(items) < 2:
            items = part.split('-')
        if len(items) < 2:
            if part.endswith('非标准'):
                std_info = 'nstd'
                part = part[:-3]
            elif part.endswith('NSTD') or part.endswith('nstd'):
                std_info = 'nstd'
                part = part[:-4]
            elif part.endswith('标准'):
                std_info = 'std'
                part = part[:-2]
            elif part.endswith('STD') or part.endswith('std'):
                std_info = 'std'
                part = part[:-3]
        else:
            part = items[0].strip()
            items[1] = items[1].strip()
            if items[1] in ['非标准', 'nstd', 'NSTD']:
                std_info = 'nstd'
            elif items[1] in ['标准', 'std', 'STD']:
                std_info = 'std'

        return part, std_info

    def description(self, parameter_list):
        """
        docstring
        """
        pass

    def confidence(self, scores):
        """
        detection confidence, which is used to judge which model is better
        """
        return 0

    def suppress_video_score(self, check_frames=15, continue_same_type_thres=5):
        """Suppress video score to better save video-clips for some planes"""
        """
        params:
        check_frames: number of frames to check before current image info
        return:
        suppress_video_score: a value that suppresses video score for current plane type with few detection nums before
        """
        if len(self.history_queue) <= 1:
            return 0

        current_image_info = self.history_queue[-1]  # current image_info is already in self.history_queue

        # Calculate the number of non-same type frames in the check range
        non_same_type_count = 0
        continue_same_type_count = 0
        tmp_cur_image_info = current_image_info

        for i in range(1, min(check_frames, len(self.history_queue) - 1) + 1):
            previous_image_info = self.history_queue[-i - 1]
            if previous_image_info.auto_type != current_image_info.auto_type:
                non_same_type_count += 1
            elif tmp_cur_image_info.auto_type == previous_image_info.auto_type:
                continue_same_type_count += 1
            tmp_cur_image_info = previous_image_info

        # Determine the suppression score based on the proportion of non-same type frames and continuous same type frames
        if continue_same_type_count >= continue_same_type_thres:
            return 0  # 连续超过continue_same_type_thres帧相同的切面不抑制
        elif 0 <= continue_same_type_count < continue_same_type_thres:
            if current_image_info.auto_score >= 75:
                suppress_video_score = min(((60 - current_image_info.auto_score) + non_same_type_count // 2), 70)
                return suppress_video_score
            return 0
        else:
            return 0

    def add_score_by_history(self):
        """
        get video score change val by self.history queue
        """
        if self.history_queue and len(self.history_queue) > 5:
            cur_info = self.history_queue[-1]
            return self.history_queue.add_score(cur_info.auto_type, cur_info.class_score)
        return 0

    def description_and_score(self, boxes_list, scores_list, labels_list, image=None, wrong_type=0):
        plane_type = self.plane_type
        label_name_list = []
        for label in labels_list:
            label_name_list.append(self.get_annotation_name(label))

        description, reason, std_info, std_score, parts_found = \
            self.description(scores_list, labels_list, boxes_list, image, wrong_type)

        auto_score = self.calculate_qc_score(std_info, std_score, parts_found, label_name_list, scores_list)
        confidence = self.confidence_score(scores_list, labels_list, boxes_list, std_info, std_score, parts_found,
                                           label_name_list)
        if confidence == -1:
            confidence = auto_score

        return plane_type, auto_score, description, reason, confidence

    def is_doubleside(self, auto_type=0):
        return False

    def compute_avg_confidence(self, boxes_list, scores_list, labels_list):
        if len(scores_list) > 1:
            return np.mean(scores_list)
        if len(scores_list) == 1:
            return scores_list[0]
        return 0

    @staticmethod
    def normalize_conf(score, pass_score):
        """
        分数标准化,转化为以60分为及格线
        """
        norm_score = 0
        if score < 0:
            norm_score = 0
        elif score < pass_score:
            norm_score = score / pass_score * 60
        else:
            norm_score = 60 + (score - pass_score) / (100 - pass_score) * 40

        norm_score = round(norm_score, 2)

        return norm_score

    @classmethod
    def normalize_score(cls, score, nstd_thresh=60, std_thresh=80):
        """
        < nstd_thresh: [10, 60)
        >= std_thresh and <= 95: [80, 95]
        > 95: [95, 98]
        """
        if score < 10:
            return 10
        if score < nstd_thresh:
            norm_score = score / nstd_thresh * 60
        elif score >= std_thresh:
            # > 95: [95, 98]
            if score > 95:
                norm_score = (score - 95) * 3 / 5 + 95
                if norm_score > 100:
                    norm_score = 100
            else:
                # <= 95: [80, 95]
                norm_score = (score - std_thresh) / (95 - std_thresh) * 15 + 80
        else:
            # [nstd_thresh, std_thresh]: (60, 80)
            norm_score = (score - nstd_thresh) / (std_thresh - nstd_thresh) * 20 + 60

        norm_score = round(norm_score, 2)
        return norm_score

    @classmethod
    def scale_image(cls, masks, image_shape, ratio_pad=None):
        """
        Takes a mask, and resizes it to the original image size

        Args:
        masks (torch.Tensor): resized and padded masks/images, [h, w, num]/[h, w, 3].
        im0_shape (tuple): the original image shape
        ratio_pad (tuple): the ratio of the padding to the original image.

        Returns:
        masks (torch.Tensor): The masks that are being returned.
        """
        # Rescale coordinates (xyxy) from im1_shape to im0_shape
        im1_shape = masks.shape
        if im1_shape[:2] == image_shape:
            return masks
        if ratio_pad is None:  # calculate from im0_shape
            gain = min(im1_shape[0] / image_shape[0], im1_shape[1] / image_shape[1])  # gain  = old / new
            pad = (im1_shape[1] - image_shape[1] * gain) / 2, (im1_shape[0] - image_shape[0] * gain) / 2  # wh padding
        else:
            gain = ratio_pad[0][0]
            pad = ratio_pad[1]
        top, left = int(pad[1]), int(pad[0])  # y, x
        bottom, right = int(im1_shape[0] - pad[1]), int(im1_shape[1] - pad[0])

        if len(masks.shape) < 2:
            raise ValueError(f'"len of masks shape" should be 2 or 3, but got {len(masks.shape)}')
        masks = masks[top:bottom, left:right]

        masks = cv2.resize(masks, (image_shape[1], image_shape[0]))
        if len(masks.shape) == 2:
            masks = np.expand_dims(masks, 2)

        return masks

    @classmethod
    def scale_mask(cls, mask, image_shape):
        if len(mask.shape) > 2:
            mask = mask.transpose((1, 2, 0))
            mask = QcModel.scale_image(mask, image_shape)
            mask = mask.transpose((2, 0, 1))
        else:
            mask = QcModel.scale_image(mask, image_shape)

        # a = np.where(0 < mask < 1)
        # whether has to multiply with 255
        mask *= 255
        mask = mask.astype(np.uint8)
        return mask

    @classmethod
    def get_scale_offset(cls, mask, image_shape):
        im1_shape = mask.shape
        if len(im1_shape) > 2:
            im1_shape = im1_shape[1:]

        gain = min(im1_shape[0] / image_shape[0], im1_shape[1] / image_shape[1])  # gain  = old / new
        pad = (im1_shape[1] - image_shape[1] * gain) / 2, (im1_shape[0] - image_shape[0] * gain) / 2  # wh padding

        return 1.0 / gain, [-pad[0], -pad[1]]

    @classmethod
    def is_dopler(cls, image):
        """
        return: (is_dopler, is_pseudo_color)
        """
        # check whether is dopler
        return RulerRecognizer.is_dopler(image)

    @classmethod
    def is_pseudo_color_image(cls, image):
        return RulerRecognizer.is_pseudo_color_image(image)

    @classmethod
    def dopler_pixels_ratio(cls, image, roi=None, is_pseudo_color=False):
        """
        statistics for ratio of dopler pixels in roi of the image
        roi: [x1, y1, x2, y2]
        """
        roi_image = image
        if roi is not None:
            roi_image = image[int(roi[1]):int(roi[3]), int(roi[0]):int(roi[2])]

        dopler_pixels = RulerRecognizer.dopler_pixels(roi_image, is_pseudo_color)
        # dopler_pixels = cls._dopler_pixels_(roi_image, is_pseudo_color)

        all_pixels = roi_image.shape[0] * roi_image.shape[1]
        return dopler_pixels / all_pixels

    @classmethod
    def dopler_pixels(cls, image, roi=None, is_pseudo_color=False):
        """
        statistics for ratio of dopler pixels in roi of the image
        roi: [x1, y1, x2, y2]
        """
        roi_image = image
        if roi is not None:
            roi_image = image[int(roi[1]):int(roi[3]), int(roi[0]):int(roi[2])]

        # return cls._dopler_pixels_(roi_image, is_pseudo_color)
        return RulerRecognizer.dopler_pixels(roi_image, is_pseudo_color)

    @classmethod
    def _dopler_pixels_(cls, roi_image, is_pseudo_color=False):

        if is_pseudo_color:
            dopler_mask = DoplerDetector.convert_RB_to_bin_image(roi_image, 80, 100)
        else:
            dopler_mask = DoplerDetector.convert_RB_to_bin_image(roi_image, 80, 60)
        cv2.imshow('dopler mask', dopler_mask)

        dopler_pixels = np.count_nonzero(dopler_mask)

        '''
            选用实验得到的伪彩图参数与正常参数做比较:
            若正常参数时比伪彩图参数大很多倍，即是背景被判断为多普勒像素,此时采用伪彩图的参数
            否则采用正常参数
        '''

        return dopler_pixels

    @classmethod
    def adjust_score_by_dopler(cls, auto_score, dopler_ratio, low_thresh=0.05, high_thresh=0.1, weight=50):
        # 多普勒打分规则：
        if dopler_ratio < low_thresh:
            # 彩色面积过小，判定为非标准
            # dopler像素的浮动范围为10，最多可以达到60分，要考虑你们的抓取阈值，避免高于阈值
            auto_score = min(auto_score, 50) + 10 * dopler_ratio / low_thresh
        elif dopler_ratio > high_thresh and auto_score > 75:
            # 彩色面积足够大，忽略小结构打分，判定为标准
            # dopler像素的分数浮动范围为[5-8]，总分为95~98
            auto_score = min(90 + dopler_ratio * weight, 98)
        else:
            # 其他情况根据彩色面积加分，最高分不超过 95分
            auto_score = min(auto_score, 90) + dopler_ratio * weight

        return auto_score

    @classmethod
    def letterbox(cls, img, new_shape, min_rect=False, padding_value=(114, 114, 114), stride=32):
        """
        new_shape: (h, w)
        return: (resized_image, padding_roi)
                 padding_roi is in format with xywh
        """

        # Resize and pad image while meeting stride-multiple constraints
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)
        if isinstance(padding_value, int):
            padding_value = (padding_value, padding_value, padding_value)

        # Scale ratio (new / old)
        if new_shape[0] == 0 and new_shape[1] == 0:
            return img, [0, 0, shape[1], shape[0]]

        MIN_SIZE = stride * 3  # 有的环境下等于stride时概率还会报错（神经网络最深层特征分辨率H或W为0），再调大一些能够解决

        n_h, n_w = new_shape
        if n_w == 0:
            # according to h
            r = n_h / shape[0]
            n_w = int(round(shape[1] * r))
            new_unpad = [n_w, n_h]    # (w, h)
            n_w = ((n_w + stride - 1) // stride) * stride if n_w > MIN_SIZE else MIN_SIZE
            n_h = ((n_h + stride - 1) // stride) * stride if n_h > MIN_SIZE else MIN_SIZE
        elif n_h == 0:
            # according to w
            r = n_w / shape[1]
            n_h = int(round(shape[0] * r))
            new_unpad = [n_w, n_h]    # (w, h)
            n_w = ((n_w + stride - 1) // stride) * stride if n_w > MIN_SIZE else MIN_SIZE
            n_h = ((n_h + stride - 1) // stride) * stride if n_h > MIN_SIZE else MIN_SIZE
        else:
            r = min(n_h / shape[0], n_w / shape[1])
            # if not scaleup:  # only scale down, do not scale up (for better val mAP)
            #     r = min(r, 1.0)
            new_unpad = [int(round(shape[1] * r)), int(round(shape[0] * r))]    # (w, h)

        # Compute padding
        # ratio = r, r  # width, height ratios

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

        dw, dh = n_w - new_unpad[0], n_h - new_unpad[1]  # wh padding
        if min_rect:
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        # padding
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=padding_value)  # add border

        unpad_w, unpad_h = new_unpad
        return img, [left, top, unpad_w, unpad_h]


if __name__ == "__main__":

    # model = QcModel(model_file_name='', class_mapping_file='', config={}, load_model=False)
    # import sys

    # if not model.is_inited():
    #     sys.exit()

    # do detection
    image = cv2.imread(r'C:\Users\guang\Desktop\measure-data\xiaonao_measure\20230711_084736.json.jpg')
    resized_image, roi = QcModel.letterbox(image, (400, 270), min_rect=False)

    print(roi)
    cv2.imshow('result', resized_image)
    cv2.waitKey()
