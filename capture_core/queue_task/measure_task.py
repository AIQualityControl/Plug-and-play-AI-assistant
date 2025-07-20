#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@description:
@date       : 2022/02/18 23:50:58
@author     : Guanghua Tan
@email      : guanghuatan@hnu.edu.cn
@version    : 1.0
'''

import cv2
import os
import sys
import time
import numpy as np
from loguru import logger

from common.model.EllipseAnnotation import EllipseAnnotation
from common.model.LineAnnotation import LineAnnotation
from common.model.AcMeasureInfo import AcMeasureInfo
from common.model.HcMeasureInfo import HcMeasureInfo
from common.model.FLMeasureInfo import FLMeasureInfo
from common.model.HLMeasureInfo import HLMeasureInfo
from common.model.AFIMeasureInfo import AFIMeasureInfo
from common.model.image_info import ImageInfo

from common.shared_queue import SharedQueue
from common.config.config import (IDX_2_FETAL_KIND, AMNIOTIC_MEASURE_MODE, SPECTURM_MEASURE_MODE, PLANES_TO_REJUDGMENT,
                                  normalized_fetal_kind, measure_model_params, Config)

# Allow relative imports when being executed as script.
if __name__ == "__main__" and (__package__ is None or __package__ == ''):
    project_dir = os.path.join(os.path.dirname(__file__), '..')
    sys.path.insert(0, project_dir)
    sys.path.append(os.path.join(project_dir, 'QcDetection'))
    # print(sys.path)
    # import utility.sub_region_detector  # noqa: F401
    __package__ = "stdplane.queue_task"

from ..QcDetection.task.task import Task
from ..ruler.ruler_recognizer import RulerRecognizer
from .history_deque import HistoryQueue

import logging

logging.basicConfig(level=logging.WARNING)


class MeasureTask(Task):

    def __init__(self, config, model_dir, image_dir=None):
        """constructor"""
        super(MeasureTask, self).__init__(model_dir, image_dir)
        self.measure_queue = None
        self.model_dir = model_dir
        # self.model_list = []
        # config['measure_mode'] = 'hadlock'
        self.config: Config = config

        self.display_queue = None
        self.model_params = measure_model_params
        self.saved_measure_info = {}
        self.history_measure_queue = HistoryQueue()
        self.modified_planes_info = {}

        self.fetal_kind = self.config.fetal_kind

    # override function call
    def __call__(self, options, measure_queue, rwlock_display):

        self.measure_queue = measure_queue

        # config = options['config']
        # self.config.common_config = config
        self.config.check_consistence(need_ruler=True)

        config = self.config.common_config
        sm_config = config['shared_memory']
        video_size = sm_config['video_shape']
        self.display_queue = SharedQueue(queue_size=sm_config['display_queue_length'], create=False,
                                         image_shape=video_size, name=sm_config['display_queue_name'],
                                         rwlock=rwlock_display)

        return super().__call__(options=options)

    def init(self, options):
        # load segmentation model
        logger.debug(f'segmentation pid: {os.getpid()}')

        if not super().init(options):
            return False

        self.load_model = options['load_model'] if 'load_model' in options else 'False'
        # gpu
        self.gpu_id = int(options['gpu_id']) if str(options["gpu_id"]) not in ["mps", "cpu"] else options["gpu_id"]
        if self.load_model:
            self.specify_gpu_device(self.gpu_id)

        # 初始化每个模式下的模型
        all_success = True
        for fetal_kind in self.config.fetal_kind_list:
            fetal_kind = normalized_fetal_kind(fetal_kind)
            if fetal_kind in self.model_params:
                model_map, success = self.init_models(self.model_params[fetal_kind], self.gpu_id, self.load_model)
                self.model_map[fetal_kind] = model_map

                if not success:
                    all_success = False

        if self.display_queue:
            self.display_queue.set_ctrl_param('has_load', True)

            if not all_success:
                self.display_queue.set_ctrl_param('model_success', False)
                logger.error('not all measure models are loaded successfully')

        return all_success

    def init_models(self, model_params, gpu_id, load_model):
        model_map = {}
        self.init_image_subtypes(model_params)

        success = True
        for image_type_name, model_param in model_params.items():
            if not self.config['measure_heart_spine'] and image_type_name in ('四腔心测量切面', '脊柱测量切面'):
                continue
            # set measure mode
            model_param['params']['config']['measure_mode'] = self.config.common_config['measure_mode']
            if load_model:
                model = self.create_model(model_param, gpu_id, load_model, 'capture_core.measure_models')
                if model is None or load_model and not model.is_inited():
                    model = None
                    logger.warning(f'Failed to init model for {image_type_name}')
                    success = False
                elif 'detect_with_roi' in model_param:
                    model.set_detect_with_roi(model_param['detect_with_roi'])
                    model.plane_type = image_type_name
            else:
                model = None

            if model is not None:
                self.dummy_segment(model)

            if image_type_name in self.image_sub_types:
                for sub_image_type in self.image_sub_types[image_type_name]:
                    model_map[sub_image_type] = model

            if image_type_name in self.name_to_db_id_map:
                image_type = self.name_to_db_id_map[image_type_name]
                model_map[image_type] = model

        return model_map, success

    def dummy_segment(self, model):
        width, height = model.config['target_width'], model.config['target_height']
        if width == 0 or height == 0:
            return
        roi_image = np.ones((height, width, 3), dtype=np.uint8)
        image_info = ImageInfo(roi_image)

        if self.fetal_kind == '甲状腺':
            model.multi_cls([roi_image])
        else:
            model.do_segment([roi_image], [image_info])
        logger.debug(f'dummy segment for {model.plane_type}')

    def construct_name2id_map(self):
        """
        overwrite to contruct map with configuration file
        base class: construct map with db file
        """
        from common.config.name2id import construct_name2id_map
        self.name_to_db_id_map, self.db_id_to_name_map = construct_name2id_map()
        self.AMNIOTIC_PLANE_ID = self.id_of_image_type('羊水测量切面')

    @logger.catch
    def run(self):
        """
        loop to get images to do segmentation and measure
        """

        logger.info('--- segmentation & measure task is ready ---')
        while not self.is_finished:

            self.update_saved_image_info()

            # second: check automatic measure
            try:
                image_info = self.measure_queue.get_nowait()
            except Exception:
                # no frame in queue, whether need to switch model
                fetal_kind_idx = self.display_queue.get_ctrl_param('fetal_kind')
                if fetal_kind_idx > 0:
                    if fetal_kind_idx in IDX_2_FETAL_KIND:
                        self.switch_model(IDX_2_FETAL_KIND[fetal_kind_idx])
                    self.display_queue.set_ctrl_param('fetal_kind', 0)
                else:
                    time.sleep(0.02)
                continue

            logger.info(f'start to measure frame: {image_info.frame_idx}')

            # wait until fetal kind is not same
            if image_info.FetalKind != self.fetal_kind:
                self.switch_model(image_info.FetalKind)

            self.config.check_consistence(need_ruler=True)

            start_time = time.time()
            # 甲状腺
            if image_info.is_thyroid():
                self.process_thyroid_image(image_info)
                continue

            # 中晚孕期、早孕期、妇科
            self.measure_biometry(image_info)

            if image_info.measure_results is not None:
                # image_info has already added to display queue in detection task
                # update measure results only if measure_results is not none
                if self.image_type_of_id(image_info.auto_type) in PLANES_TO_REJUDGMENT:
                    if image_info.new_or_update_mode >= 3 and image_info.new_or_update_mode != 6:
                        self.add_queue(self.display_queue, image_info)
                        logger.info(f'Succeed to measure frame {image_info.frame_idx} with measure score '
                                    f'{image_info.measure_results.measure_score}')
                    elif image_info.new_or_update_mode == 6:
                        self.add_queue(self.display_queue, image_info)
                        logger.info(
                            f'Succeed to measure frame {image_info.frame_idx} normal measure frame with lower measure score '
                            f'{image_info.measure_results.measure_score}')
                    else:
                        self.add_queue(self.display_queue, image_info)
                        logger.info(f'Succeed to measure frame {image_info.frame_idx}')
                else:
                    self.add_queue(self.display_queue, image_info)
                    logger.info(f'Succeed to measure frame {image_info.frame_idx}')

            else:
                logger.warning(f'Failed to measure frame {image_info.frame_idx} with type '
                               f'{self.image_type_of_id(image_info.auto_type)}')

            during_time = time.time() - start_time
            if during_time > 0.03:
                log_info = f'measure time: {during_time}, image type: ' \
                           + f'{self.image_type_of_id(image_info.auto_type)}, frame_idx: {image_info.frame_idx}'
                if during_time > 0.1:
                    logger.info(log_info)
                else:
                    logger.debug(log_info)

    # ##thyroid cls####
    def process_thyroid_image(self, image_info):
        # 处理甲状腺，做甲状腺分类
        auto_type = self.id_of_image_type('甲状腺切面')
        model = self.get_model(image_info, auto_type)
        if model is None:
            logger.info(f'frame_idx: {image_info.frame_idx}, auto type: {auto_type}')
            raise Exception('Failed to load segmentation model for 甲状腺切面')

        # 结节不存在，则不要加入测量队列，减少开销
        if not image_info.measure_results:
            return

        image = image_info.roi_image()
        if image is None:
            logger.warning(f'frame is None for {image_info.frame_idx}')
            return

        for anno in image_info.measure_results.nodule_anno_list:
            if anno.name != 'nodule':
                continue
            if anno.custom_props is None:
                anno.custom_props = {}
                anno.custom_props['cls_props'] = {}
            img = image[(anno.ptStart[1]-5):(anno.ptEnd[1]+5), (anno.ptStart[0]-5):(anno.ptEnd[0]+5), :]
            if img.shape[0] == 0:
                continue

            (
                (_, cf_out),
                (_, hs_out),
                (_, xt_out),
                (_, by_out),
                (_, cg_out),
                (_, bg_out),
                (_, dz_out),
                (_, bm_out),
                (_, pf_out),
            ) = model.multi_cls([img])

            anno.custom_props['cls_props'] = {
                'cf_out': cf_out[0],
                'hs_out': hs_out[0],
                'xt_out': xt_out[0],
                'by_out': by_out[0],
                'cg_out': cg_out[0],
                'bg_out': bg_out[0],
                'dz_out': dz_out[0],
                'pf_out': pf_out[0],
                'bm_out': bm_out[0]
            }

            # mapping to measure results

        if self.display_queue:
            self.add_queue(self.display_queue, image_info)

    @logger.catch
    def measure_biometry(self, image_info, plane_type=None):

        if image_info.capture_type == AMNIOTIC_MEASURE_MODE:
            logger.info(f'Amniotic measure: frame_idx: {image_info.frame_idx}, '
                        f'({image_info.auto_type}, {self.image_type_of_id(image_info.auto_type)})')

        elif image_info.capture_type == SPECTURM_MEASURE_MODE or plane_type == '频谱测量切面':
            biometry_info = RulerRecognizer.detect_spectrum(image_info, convert_envelop=False)
            image_info.measure_results = biometry_info
            self.add_queue(self.display_queue, image_info)
            return

        if image_info.anno_set is None:
            image_type_id = abs(image_info.auto_type)
            image_type = self.image_type_of_id(image_type_id)

            if not image_info.FetalKind:
                image_info.FetalKind = self.get_fetal_kind(image_type)

        else:
            # used for debug
            if image_info.capture_type == AMNIOTIC_MEASURE_MODE:
                image_type = '羊水测量切面'
            else:
                image_type = plane_type if plane_type else image_info.anno_set.plane_type
            if not image_type:
                raise Exception('image type is not specified for this image')

            image_type_id = abs(self.id_of_image_type(image_type))
            image_info.FetalKind = self.get_fetal_kind(image_type)

        model = self.get_model(image_info, image_type_id)

        if model is None:
            if image_info.anno_set is None:
                msg = f'Failed to load segmentation model for frame {image_info.frame_idx} with type {image_type}'
                logger.error(msg)
                # raise Exception(msg)
                return

            # annotations = image_info.anno_set.annotations
            # if model is not specified while annotation set is specified, use measure results in annotation set

            image_info.measure_results = image_info.anno_set.measure_results
            image_info.measure_score = image_info.anno_set.measure_score
            if image_info.measure_results:
                image_info.measure_results.measure_score = image_info.anno_set.measure_score
            return

        # set history queue
        model.set_history_measure_queue(self.history_measure_queue)
        self.history_measure_queue.enqueue(image_info)

        model.set_plane_type_and_id(image_type, image_type_id)
        model.correct_CRL = self.config.common_config['correct_CRL']

        logger.debug(f'measure biometry with model for frame {image_info.frame_idx}')
        # do measure
        # start = time.time()
        biometry_info = model.measure_biometry(image_info)

        image_type_id = self.image_type_of_id(image_info.auto_type)
        if biometry_info is not None and image_type_id in PLANES_TO_REJUDGMENT:
            if biometry_info.measure_score != -1:
                measure_score = biometry_info.measure_score

                self.check_std_measure(image_info, measure_score)

        if biometry_info is not None:
            if image_info.ruler_info:
                ruler_info = image_info.ruler_info
                logger.debug(f'receive frame {image_info.frame_idx} ruler:{image_info.ruler_info}')
            else:
                ruler_info = self.recognize_ruler(image_info, biometry_info, self.config)
                if ruler_info is None or not isinstance(ruler_info, list) and \
                        (ruler_info['rulerUnit'] == 0 or ruler_info['rulerUnit'] == 1):
                    logger.error(f'Failed to recognize ruler for frame {image_info.frame_idx} with type '
                                 f'{image_type_id}')
                else:
                    if isinstance(ruler_info, list):
                        for i, ruler in enumerate(ruler_info):
                            if ruler is None or ruler['rulerUnit'] == 0 or ruler['rulerUnit'] == 1:
                                logger.error(f'Failed to recognize ruler for frame {image_info.frame_idx} '
                                             f'at quadrant {i} with type {image_type_id}')
                    logger.info(f're-recognize ruler for frame {image_info.frame_idx} with type '
                                f'{image_type_id}')

            biometry_info.update_ruler_info(ruler_info)

            if image_type == '侧脑室水平横切面':
                biometry_info.disease_name_list = model.diagnose_disease(biometry_info, image_info, image_type)

        image_info.measure_results = biometry_info

    def check_std_measure(self, image_info, measure_score):
        if measure_score == -1:
            return

        image_info.measure_score = measure_score

        modified_planes_info = {}

        plane_id = image_info.auto_type
        image_type = self.image_type_of_id(plane_id)
        frame_idx = image_info.frame_idx
        conf = image_info.detection_results['confidence']
        auto_score = image_info.detection_results['auto_score']

        # used for update saved information in detection
        info = (measure_score, frame_idx, conf, auto_score)

        if plane_id in self.saved_measure_info:
            saved_measure_image_info = self.saved_measure_info[plane_id]
            last_measure_score, last_measure_frame_idx, last_conf = saved_measure_image_info[:3]

            if measure_score > last_measure_score or measure_score == last_measure_score and conf > last_conf:
                modified_planes_info[plane_id] = info
                image_info.new_or_update_mode = 5 if image_info.frame_idx - last_measure_frame_idx > 1500 else 4
                image_info.last_measure_frame_idx = last_measure_frame_idx

                logger.info(f'update measure frame: {frame_idx} -> {last_measure_frame_idx} with image type {image_type}, '
                            f'measure_score: {measure_score}')
            elif image_info.video_score > last_measure_score and image_info.new_or_update:
                # for those whose video > best measure, but measure score small than best measure,
                # send display queue incase no measure results in UI
                image_info.optimal_replace = -1
                image_info.measure_score = 0
                image_info.new_or_update_mode = 6

        else:
            modified_planes_info[plane_id] = info
            image_info.new_or_update_mode = 3

            logger.info(f'new measure frame: {frame_idx} with image type {image_type}')

        if modified_planes_info:
            self.saved_measure_info[plane_id] = info

            image_info.detection_results['video_score'] = measure_score

            logger.info(
                f'measure frame {frame_idx} with type {self.image_type_of_id(plane_id)} has been rejuged by measure task, '
                f'video score/measure score:{image_info.video_score}/{measure_score}')

    def update_saved_image_info(self):
        """
        update saved image info when changed to a previous patient case
        """
        if self.display_queue.get_ctrl_param('clear_saved_image_info'):
            saved_measure_info = self.display_queue.get_prepatient_result()

            # None or length = 0
            if saved_measure_info:
                self.saved_measure_info = saved_measure_info
                self.display_queue.update_prepatient_result(None)
                logger.info(f"update_saved_measure_score: {self.saved_measure_info}")
            else:
                self.saved_measure_info = {}

            self.history_measure_queue.clear()

            self.display_queue.set_ctrl_param('clear_saved_image_info', False)

    def get_model(self, image_info, image_type_id):
        model_map = self.model_map.get(normalized_fetal_kind(image_info.FetalKind))
        if model_map is None:
            logger.info(f'No segmentation model for {image_info.FetalKind}')
            raise Exception(f'No segmentation model for {image_info.FetalKind}')

        model = model_map.get(image_type_id)
        return model

    def get_fetal_kind(self, plane_type):
        plane_fetal_kind = self.fetal_kind
        if plane_type:
            for fetal_kind in self.model_params:
                if plane_type in self.model_params[fetal_kind]:
                    plane_fetal_kind = fetal_kind
                    break
        return plane_fetal_kind

    @classmethod
    def recognize_ruler(cls, image_info, biometry_info=None, sm_config: Config = None):
        # ruler unit: should use the whole image, do not depend on roi_bbox
        if biometry_info and isinstance(biometry_info, AFIMeasureInfo):

            # detect ruler for afi
            x_extend = 5
            offset = image_info.offset()

            end_points = biometry_info.get_end_points()
            roi_image = image_info.roi_image(x_extend=x_extend)
            ruler_info_list = RulerRecognizer.detect_afi_ruler(roi_image, end_points)

            if ruler_info_list:
                result_ruler_list = []
                for ruler_info in ruler_info_list:
                    if ruler_info.rulerUnit == -1:
                        continue

                    result_ruler_list.append(cls.ruler_info_to_dict(ruler_info, offset, x_extend))

                return result_ruler_list
        else:
            ruler_info = RulerRecognizer.detect_ruler(image_info.image, image_info.roi, sm_config)

            if ruler_info:
                return cls.ruler_info_to_dict(ruler_info)

    @classmethod
    def ruler_info_to_dict(cls, ruler_info, offset=None, x_extend=0):
        count = ruler_info.count if ruler_info.count < 100 else 100
        if offset is not None:
            return {
                'startX': ruler_info.startX + offset[0] + x_extend,
                'endX': ruler_info.endX + offset[0] + x_extend,
                'startY': ruler_info.startY + offset[1],
                'endY': ruler_info.endY + offset[1],
                'count': count,
                'rulerUnit': ruler_info.rulerUnit
            }
        else:
            return {
                'startX': ruler_info.startX,
                'endX': ruler_info.endX,
                'startY': ruler_info.startY,
                'endY': ruler_info.endY,
                'count': count,
                'rulerUnit': ruler_info.rulerUnit
            }

    @classmethod
    def test_for_ruler(cls, image, sm_config: Config = None):

        anno_list = []
        for i in range(4):
            anno_list.append(LineAnnotation([i * 20, i * 20], [i * 20 + 100, i * 20 + 100]))
        biometry_info = AFIMeasureInfo(anno_list)

        if isinstance(biometry_info, AFIMeasureInfo):
            image_info = ImageInfo(image=image)
            x_extend = 5
            roi_image = image_info.roi_image(x_extend=x_extend)

            end_points = biometry_info.get_end_points()
            ruler_info = RulerRecognizer.detect_afi_ruler(roi_image, end_points)
        else:
            ruler_info = RulerRecognizer.detect_ruler(image, ImageInfo.roi, sm_config)

        return ruler_info

    def test_for_measure(self, image_info, cur_image_info, sm_config: Config = None):

        ruler_info = RulerRecognizer.detect_ruler(image_info.image, image_info.roi, sm_config)
        ruler_unit = ruler_info.rulerUnit
        if image_info.measure_results is not None:
            image_info.measure_results.ruler_unit = ruler_info.rulerUnit
            self.add_queue(self.display_queue, cur_image_info)
            return

        if 'seg_roi' in image_info.detection_results:
            bbox = image_info.detection_results['seg_roi']

            pt_start = [bbox[0], bbox[1]]
            pt_end = [bbox[0] + bbox[2], bbox[1] + bbox[3]]

            image_type = self.image_type_of_id(image_info.auto_type)
            result = None
            if image_type == '股骨长轴切面':
                result = FLMeasureInfo(pt_start, pt_end, ruler_unit)
            elif image_type == '肱骨长轴切面':
                result = HLMeasureInfo(pt_start, pt_end, ruler_unit)
            elif image_type == '上腹部水平横切面':
                result = AcMeasureInfo(pt_start, pt_end, 45, ruler_unit)
            elif image_type == '丘脑水平横切面':
                ellipse = EllipseAnnotation(pt_start, pt_end, 45)

                end_points = ellipse.minor_radius_points()
                line = LineAnnotation(end_points[0], end_points[1])

                result = HcMeasureInfo(ellipse, line, ruler_unit)

            if result is not None:
                result.translate(image_info.offset())

            image_info.measure_results = result

            self.add_queue(self.display_queue, cur_image_info)

    def test_zggj_measure(self, image_info):
        from common.model.ZGGJMeasureInfo import ZGGJMeasureInfo
        from common.model.PolylineAnnotation import PolylineAnnotation

        points = []
        for i in range(15):
            points.append([400 + i * 30, 400 + i * 30])
        gjx_anno = PolylineAnnotation(points)

        gjminor_anno = LineAnnotation([400, 800], [800, 400])
        info = ZGGJMeasureInfo(gjx_anno=gjx_anno, gjmin_anno=gjminor_anno)

        ruler_info = self.recognize_ruler(image_info, info, self.config)
        info.update_ruler_info(ruler_info)

        return info

    @classmethod
    def add_queue(cls, queue, image_info):
        # image_info.image = None
        try:
            if hasattr(image_info, 'gray_image'):
                delattr(image_info, 'gray_image')

            # no need to pass image
            queue.add_display_frame(image_info, ignore_image=True)
        except Exception as e:
            logger.info(f'--- display queue is full, exception:{str(e)}---')

    @classmethod
    def blend_with_mask(cls, image, mask):
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            image = image.copy()

        image[:, :, 0] = cv2.addWeighted(image[:, :, 0], 0.8, mask, 0.2, 0)
        return image

    def switch_model(self, fetal_kind):
        new_fetal_kind = normalized_fetal_kind(fetal_kind)
        origin_fetal_kind = normalized_fetal_kind(self.fetal_kind)
        # the same fetal kind
        if origin_fetal_kind == new_fetal_kind:
            self.fetal_kind = fetal_kind
            return

        # already loaded
        if new_fetal_kind in self.model_map and self.model_map[new_fetal_kind]:
            self.fetal_kind = fetal_kind
            return

        logger.info(f'switch fetal kind from {self.fetal_kind} to {fetal_kind}')

        if self.display_queue:
            self.display_queue.set_ctrl_param('has_load', False)
        # using set to remove duplicate since multiple plane may use the same model to do measure
        model_set = set(self.model_map[origin_fetal_kind].values())
        for model in model_set:
            if model:
                model.clear_model()
        self.model_map[origin_fetal_kind] = {}

        self.fetal_kind = new_fetal_kind
        # load models
        self.fetal_kind = fetal_kind

        model_map, success = self.init_models(self.model_params[new_fetal_kind], self.gpu_id, self.load_model)
        self.model_map[new_fetal_kind] = model_map

        logger.info(f'succeed to switch fetal kind to {fetal_kind} in measure task')

        if self.display_queue:
            self.display_queue.set_ctrl_param('has_load', True)
            self.display_queue.set_ctrl_param('model_success', success)


def show_anno_set(image, anno_set):
    for anno in anno_set.annotations:
        if isinstance(anno, LineAnnotation):
            pt_start = anno.start_point()
            pt_end = anno.end_point()
            cv2.line(image, (int(pt_start[0]), int(pt_start[1])), (int(pt_end[0]), int(pt_end[1])), (0, 0, 255))
        elif isinstance(anno, EllipseAnnotation):
            center = anno.center_point()
            size = anno.size()
            angle = anno.degree()
            cv2.ellipse(image, (round(center[0]), round(center[1])), (round(size[0] / 2), round(size[1] / 2)),
                        angle, 0, 360, (0, 0, 255))
    cv2.imshow('result', image)


def show_measure_info(image, measure_info):
    if isinstance(measure_info, FLMeasureInfo):
        pt_start = measure_info.start_point()
        pt_end = measure_info.end_point()
        cv2.line(image, (int(pt_start[0]), int(pt_start[1])), (int(pt_end[0]), int(pt_end[1])), (0, 0, 255))
    elif isinstance(measure_info, AcMeasureInfo):
        center = measure_info.center_point()
        size = measure_info.size()
        angle = measure_info.degree()
        cv2.ellipse(image, (round(center[0]), round(center[1])), (round(size[0] / 2), round(size[1] / 2)),
                    angle, 0, 360, (0, 0, 255))
    elif isinstance(measure_info, HcMeasureInfo):
        # hc
        anno = measure_info.hc_annotation
        center = anno.center_point()
        size = anno.size()
        angle = anno.degree()
        cv2.ellipse(image, (round(center[0]), round(center[1])), (round(size[0] / 2), round(size[1] / 2)), angle,
                    0, 360, (0, 0, 255))

        # bpd
        anno = measure_info.bpd_annotation
        pt_start = anno.start_point()
        pt_end = anno.end_point()
        cv2.line(image, (int(pt_start[0]), int(pt_start[1])), (int(pt_end[0]), int(pt_end[1])), (0, 0, 255))

    cv2.imshow('result', image)


if __name__ == '__main__':
    # from QcDetection.config import model_params

    # gray_image_list = []
    MODEL_DIR = r"C:\Users\wanghang\Desktop\latest.pth"
    IMAGE_DIR = r'F:\datasets\BUltrasonic\ImageWare\StdPlane\HD-QC\HC\std'
    IMAGE_DIR = r'E:\images_with_annotation (3)'

    task = MeasureTask(MODEL_DIR)

    roi_image = cv2.imread(r'E:\images_with_annotation (3)\1f52935247c94ebab69fe1418839401e.jpg')
    # roi_image = image[0:1024, 300:1580]
    image = roi_image
    gray_image = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
    ruler_unit = task.test_for_ruler(gray_image)
    exit()

    # task.init()
    # if not task.init({'gpu_id': 0}, model_params):
    #     exit(0)

    for file_name in os.listdir(IMAGE_DIR):
        if not file_name.endswith(('.jpg', '.jpeg', '.png', '.tif')):
            continue

        file_name = 'fl.jpg'

        print(file_name)
        image_path = os.path.join(IMAGE_DIR, file_name)
        orig_image = cv2.imdecode(np.fromfile(image_path), cv2.IMREAD_UNCHANGED)

        gray_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2GRAY)

        roi = [356, 220, 521, 200]
        gray_image = gray_image[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2]]
        image = orig_image[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2]]

        mask_list = task.do_segmentation([gray_image])

        if mask_list is None or len(mask_list) == 0:
            continue

        mask = mask_list[0]

        cv2.imwrite(os.path.join('E:/seg', file_name), image)
        file_name = f'{os.path.basename(file_name)}_mask.jpg'
        cv2.imwrite(os.path.join('E:/seg', file_name), mask)

        # ruler
        ruler_info = task.detect_ruler(orig_image)

        # testing measure
        # from model.image_info import ImageInfo
        image_info = ImageInfo(image)
        ImageInfo.roi = roi
        image_info.class_results = [{
            'class_type': task.id_of_image_type('股骨长轴切面'),
            'class_score': 1.0
        }]
        measure_info = task.do_measure(gray_image, mask, image_info, ruler_info.rulerUnit)
        if measure_info is not None:
            measure_info.translate(image_info.offset())
            show_measure_info(orig_image, measure_info)

        cv2.imshow('image', image)
        cv2.imshow('mask', mask)

        blend = MeasureTask.blend_with_mask(image, mask)
        cv2.imshow('blend', blend)

        cv2.waitKey()
