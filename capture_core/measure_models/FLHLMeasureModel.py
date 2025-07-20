#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@description:
@date       : 2022/04/14 22:56:01
@author     : Guanghua Tan
@email      : guanghuatan@hnu.edu.cn
@version    : 1.0
'''
from ..biometry_measure.LineFitting import LineFitting
from common.model.FLMeasureInfo import FLMeasureInfo
from common.model.HLMeasureInfo import HLMeasureInfo
from .DDRNetModel import DDRNetModel


class FLHLMeasureModel(DDRNetModel):
    def __init__(self, model_file_name, class_mapping_file, config, load_model=True,
                 gpu_id=0, model_dir=r'/data/QC_python/model/'):
        '''constructor'''
        super(FLHLMeasureModel, self).__init__(model_file_name, class_mapping_file, config,
                                               load_model, gpu_id, model_dir)

    def do_measure(self, mask, roi_image, image_info):
        line_info = LineFitting.fit_line(gray_image=roi_image, mask=mask) if mask is not None else None
        # line_info = []
        error_type = ''
        is_default_value = False
        if not line_info:
            line_info = self.default_line_info(roi_image, image_info)
            error_type = 'FL/HL error'
            is_default_value = True

        if self.plane_type.startswith('股骨'):
            info = FLMeasureInfo(line_info[0], line_info[1], is_default_value=is_default_value)
        elif self.plane_type.startswith('肱骨'):
            info = HLMeasureInfo(line_info[0], line_info[1], is_default_value=is_default_value)

        info.error_type = error_type

        return info

    def default_line_info(self, roi_image, image_info):
        part_name = '肱骨' if self.plane_type.startswith('肱骨') else '股骨'
        bbox = self.get_part_bbox(image_info, part_name)
        # bbox = None
        if bbox:
            pt_start, pt_end = bbox
            y = (pt_start[1] + pt_end[1]) / 2

            pt_start = [pt_start[0], y]
            pt_end = [pt_end[0], y]
        else:
            h, w = roi_image.shape[:2]
            pt_start = [w * 0.2, h / 2]
            pt_end = [w * 0.8, h / 2]

        return [pt_start, pt_end]
