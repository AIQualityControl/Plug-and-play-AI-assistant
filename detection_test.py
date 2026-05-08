#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import cv2
import os
import json
from loguru import logger
from pathlib import Path
from tests.base_test import BaseTest
from common.config.config import detection_model_params  
from common.model.image_info import ImageInfo
from common.model.AnnotationSet import AnnotationSet
from capture_core.QcDetection.task.task import Task
from capture_core.AnnotationIO.annotation_io import load_annotations
from tests.display_test import draw_image_annotations   
from common.model.BoxAnnotation import BoxAnnotation  

class DetectionTest(BaseTest):
    def __init__(self, root_path, test_with_annotations=True):
        super(DetectionTest, self).__init__(root_path, test_with_annotations)
        self.model = None   

    def load_model(self, type_name, load_model=True, gpu_id=0):
        """
        load model
        """
        fetal_kind = self.config['fetal_kind'][0]
        model_param_map = detection_model_params.get(fetal_kind, {})
        if type_name not in model_param_map:
            logger.error(f'No detection model param for {type_name} in {fetal_kind}')
            return

        if load_model and gpu_id != 'cpu':
            import torch
            if not torch.cuda.is_available():
                gpu_id = 'cpu'
                logger.warning('GPU not available, fallback to CPU')

        model_param = model_param_map[type_name]
        model_dir = str(Path.cwd().joinpath('capture_core', 'model_config'))
        task = Task(model_dir, '')

        package_root = 'capture_core.QcDetection.qc_models'   # Path of the package

        self.model = task.create_model(model_param, gpu_id, load_model, package_root)
        
        self.model.set_detect_with_roi(model_param.get('detect_with_roi', False))
        self.model.plane_type = type_name

    def run(self, type_name, load_model=True, start_frame_idx=0,
            save_result=False, show_result=True, gpu_id=0, anno_file="annotations.json"):
        self.show_result = show_result
        self.load_model(type_name, load_model=load_model, gpu_id=gpu_id)
        if not self.model:
            logger.error(f'Failed to init detection model for {type_name}')
            return
        return super().run(start_frame_idx, save_result, anno_file)

    def test_image(self, image, annoset, frame_idx, image_name=None):
        if self.model is None:
            return annoset

        image_info = self.get_image_info(image, annoset) 
        
        detections = self.model.detect(image_info)

        boxes_batch, confs_batch, clsids_batch = detections

        boxes = boxes_batch[0]
        scores = confs_batch[0]
        labels = clsids_batch[0]
        plane_type, auto_score, description, reason, confidence = \
        self.model.description_and_score(boxes, scores, labels, None, image)

        if annoset is None:
            annoset = AnnotationSet(plane_type)

        annoset.annotations = []
        for d in description:
            name = d['name']
            x1, y1, x2, y2 = map(float, d['vertex'].split(','))
            score = d.get('score', 0.0)

            box = BoxAnnotation()
            box.name = name
            box.ptStart = [x1, y1]
            box.ptEnd = [x2, y2]
            box.score = score
            if auto_score >= 82:
                annoset.annotations.append(box)

        annoset.plane_type = plane_type if auto_score >= 82 else "score"
        annoset.score = auto_score if auto_score >= 82 else 0.0
        annoset.std_type = "standard" if auto_score >= 83 else "nonstandard"
        annoset.confidence = confidence

        if not image_name:
            image_name = 'frame idx = '+str(frame_idx)

        # show
        if self.show_result:
            image_with_boxes = draw_image_annotations(image, annoset, image_name)
            cv2.imshow('result', image_with_boxes)
            # cv2.waitKey(0)

        return annoset

    @staticmethod
    def compare_detection_results(gt_anno_path, pred_anno_path, iou_threshold=0.5):
        """
        Calculate mAP to compare
        """
        gt_data = load_annotations(gt_anno_path)
        pred_data = load_annotations(pred_anno_path)
        if not gt_data or not pred_data:
            logger.error('Failed to load annotation files')
            return

        gt_image2anno, _ = gt_data
        pred_image2anno, _ = pred_data
      
        error_info = {}
        for img_name, pred_annoset_dict in pred_image2anno.items():
            pred_annoset = pred_annoset_dict['annosets'][0]
            if img_name not in gt_image2anno:
                error_info[img_name] = {'error': 'not in ground truth', 'pred': pred_annoset}
                continue

            gt_annoset = gt_image2anno[img_name]['annosets'][0]
            pred_dets = pred_annoset.get('detection_results', [])
            gt_dets = gt_annoset.get('detection_results', [])

            if len(pred_dets) != len(gt_dets):
                error_info[img_name] = {'error': f'detection count mismatch: pred={len(pred_dets)}, gt={len(gt_dets)}',
                                        'pred': pred_dets, 'gt': gt_dets}

        if not error_info:
            logger.info('All detection results match')
        else:
            out_path = os.path.join(os.path.dirname(pred_anno_path), 'detection_diff.json')
            with open(out_path, 'w') as f:
                json.dump(error_info, f, indent=2, ensure_ascii=False)
            logger.info(f'Differences saved to {out_path}')
        return error_info


if __name__ == '__main__':
    # demo
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    from PySide6.QtGui import QGuiApplication
    _ = QGuiApplication([''])

    # root_path = r"femur_images/Testing_femur1.jpg"
    root_path = r"femur_images/bone_dm_demo1.avi"
    tester = DetectionTest(root_path, test_with_annotations=False)
    tester.config['fetal_kind'] = ['second and third trimesters']

    tester.run('Long-axis plane of the femur or humerus', save_result=True, show_result=True, gpu_id="cpu")

