#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import sys
import cv2
import os
import json
from loguru import logger
from pathlib import Path
import os

# Allow relative imports when being executed as script.
if __name__ == "__main__" and (__package__ is None or __package__ == ''):
    from pathlib import Path

    # project_dir = str(Path(__file__).parents[1])
    # sys.path.insert(0, project_dir)
    # import utility.sub_region_detector  # noqa: F401
    __package__ = "pystdplane1"
print(sys.path)

from tests.base_test import BaseTest
from common.config.config import measure_model_params
from common.model.AnnotationSet import AnnotationSet
from capture_core.QcDetection.task.task import Task
from capture_core.queue_task.measure_task import MeasureTask
from capture_core.AnnotationIO.annotation_io import load_annotations

from tests.display_test import draw_image_annotations
from common.model.image_info import ImageInfo
from common.model.converter import measure_info_from_json


class MeasureTest(BaseTest):
    def __init__(self, root_path, test_with_annotations=True, crop_pixels=0):
        '''
        roi: video/image roi to detect, format: (x, y, w, h)
             if roi is not specifed, use roi configuration in config.json
        '''
        super(MeasureTest, self).__init__(root_path, test_with_annotations)

        # load model
        self.model = None

        self.crop_pixels = crop_pixels

    def load_model(self, type_name, load_model=True, gpu_id=0):
        """
        classify_or_detection_or_measure: 0-classification, 1-detection, 2-measure
        """
        # load
        fetal_kind = self.config['fetal_kind'][0]
        model_param_map = measure_model_params[fetal_kind]

        if type_name not in model_param_map:
            logger.error(f'No model param for {type_name} in {fetal_kind}')
            return

        if load_model and gpu_id != 'cpu':
            import torch
            if not torch.cuda.is_available():
                gpu_id = 'cpu'
                logger.warning('GPU is not available, use CPU instead')

        model_param = model_param_map[type_name]
        MODEL_DIR = str(Path.cwd().joinpath('capture_core', 'model_config'))
        task = Task(MODEL_DIR, '')

        package_root = 'capture_core.measure_models'

        # load model or not load model
        # load_model = not self.test_with_annotations
        self.model = task.create_model(model_param, gpu_id, load_model, package_root)

        self.model.set_detect_with_roi(model_param['detect_with_roi'])
        self.model.plane_type = type_name

    def run(self, type_name, load_model=True, start_frame_idx=0, save_result=False, show_result=True, gpu_id=0,
            anno_file="annotations.json"):
        """
        only test image with specified type name for classification or detection or measurement
        classify_or_detection_or_measure: 0-classification, 1-detection, 2-measure
        """
        self.show_result = show_result
        self.load_model(type_name, load_model=load_model, gpu_id=gpu_id)
        
        if not self.model:
            logger.error(f'Failed to init model for {type_name}')
            return
        # goto the run function in best_test.py
        return super().run(start_frame_idx, save_result, anno_file)

    def test_image(self, image, annoset, frame_idx, image_name=None):
        if self.model is None:
            return

        if self.crop_pixels > 0:
            image = self.crop_image(image, self.crop_pixels)

        image_info = self.get_image_info(image, annoset)
        biometry_info = self.model.measure_biometry(image_info)

        if biometry_info:
            ruler_info = MeasureTask.recognize_ruler(image_info, biometry_info, self.config)
            biometry_info.update_ruler_info(ruler_info)
            biometry_info.update_ga()

        if annoset is None:
            annoset = AnnotationSet(self.model.plane_type)
            annoset.measure_results = biometry_info
        else:
            annoset.measure_results = biometry_info

        # draw
        if not image_name:
             image_name = 'frame idx = '+str(frame_idx)
            
        if self.show_result:
            image = draw_image_annotations(image, annoset, image_name)
            cv2.imshow('measure result', image)
            # cv2.waitKey()
            # cv2.destroyAllWindows()

        return annoset

    @classmethod
    def compare_measure_results(cls, gt_anno_path, anno_path):
        gt_image2anno = load_annotations(gt_anno_path)
        if not gt_image2anno:
            logger.error('failed to load gt_annotation: ' + gt_anno_path)
            return
        gt_image2anno, _ = gt_image2anno
        image2anno = load_annotations(anno_path)
        if not image2anno:
            logger.error('failed to load annotation: ' + anno_path)
            return
        image2anno, _ = image2anno

        error_info = {}
        for image_name_or_frame_idx, annoset in image2anno.items():
            annoset = annoset['annosets'][0]
            if image_name_or_frame_idx not in gt_image2anno:
                error_info[image_name_or_frame_idx] = {
                    'gt': {},
                    'cur': annoset,
                    'error_type': 'not exist in gt'
                }
                continue

            gt_annoset = gt_image2anno[image_name_or_frame_idx]['annosets'][0]

            # compare the measure results
            if 'measure_results' not in annoset and 'measure_results' not in gt_annoset:
                continue
            if 'measure_results' not in annoset or 'measure_results' not in gt_annoset:
                error_info[image_name_or_frame_idx] = {
                    'gt': gt_annoset,
                    'cur': annoset,
                    'error_type': 'measurement results existance inconsistent'
                }
                continue
            measure_results = annoset['measure_results']
            gt_measure_results = gt_annoset['measure_results']

            if not measure_results and not gt_measure_results:
                continue
            if not measure_results or not gt_measure_results:
                error_info[image_name_or_frame_idx] = {
                    'gt': gt_measure_results,
                    'cur': measure_results,
                    'error_type': 'measurement results existance inconsistent'
                }
                continue
            if measure_results['type'] != gt_measure_results['type']:
                error_info[image_name_or_frame_idx] = {
                    'gt': gt_measure_results,
                    'cur': measure_results,
                    'error_type': 'the type of measurement results is inconsistent'
                }
                continue

            # convert to measure results
            measure_anno = measure_info_from_json(measure_results)
            gt_measure_anno = measure_info_from_json(gt_measure_results)

            if not measure_anno.is_same_as(gt_measure_anno):
                error_info[image_name_or_frame_idx] = {
                    'gt': gt_measure_results,
                    'cur': measure_results,
                    'error_type': 'measurement results are inconsistent'
                }

        if not error_info:
            logger.info('no difference')
            return

        root_path, anno_name = os.path.split(anno_path)
        anno_name, _ = os.path.splitext(anno_name)
        output_path = os.path.join(root_path, 'error_' + anno_name + '.json')
        with open(output_path, 'w', encoding='utf-8') as fs:
            str_list = ['{\n']
            for image_name, errors in error_info.items():
                error_str = f'"{image_name}": ' + '{\n'
                for key, value in errors.items():
                    error_str += f'  "{key}": {json.dumps(value, ensure_ascii=False)},\n'
                error_str = error_str[:-2] + '},\n'
                str_list.append(error_str)
            str_list[-1] = str_list[-1][:-2] + '\n}'

            fs.writelines(str_list)

        logger.info(f'compare finished, {len(error_info)} differences')
import numpy as np


if __name__ == '__main__':
    from capture_core.ruler.ruler_recognizer import RulerRecognizer
    RulerRecognizer.set_ruler_type('GE_E8')
    from PySide6.QtGui import QGuiApplication

    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    # from PySide6.QtGui import QGuiApplication
    _ = QGuiApplication([''])

    # root_path = r"femur_images/Testing_femur1.jpg"
    # root_path = r"femur_images/20240311_183347_8680.avi"
    root_path = r"femur_images/bone_dm_demo1.avi"
    tester = MeasureTest(root_path, test_with_annotations=True)
    tester.config['fetal_kind'] = ['second and third trimesters']
    tester.run('femur humerus measurement', save_result=True, show_result=True, gpu_id="cpu")

