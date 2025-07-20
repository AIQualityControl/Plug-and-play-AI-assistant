#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@description:
@date       : 2022/04/15 17:50:22
@author     : Guanghua Tan
@email      : guanghuatan@hnu.edu.cn
@version    : 1.0
'''

from .measure_info import MeasureInfo
from .LineAnnotation import LineAnnotation


class NtMeasureInfo(LineAnnotation, MeasureInfo):
    def __init__(self, ptStart=[0, 0], ptEnd=[0, 0], measure_score=0, is_default_value=False):
        # sourcery skip: default-mutable-arg
        """constructor"""
        LineAnnotation.__init__(self, ptStart, ptEnd, is_default_value=is_default_value)
        MeasureInfo.__init__(self)

        self.measure_length = 0

        self.all_biometry = False
        self.measure_score = measure_score

        self.update_measure_annos()

    def update_measure_annos(self):
        self.measure_annos = [self]

    @classmethod
    def from_annotation(cls, line_anno):
        return NtMeasureInfo(line_anno.start_point(), line_anno.end_point())

    @classmethod
    def from_json(cls, json_info):
        fl = NtMeasureInfo()

        fl._from_json(json_info)

        fl.parse_ruler_info(json_info)
        return fl

    def to_json_object(self):
        info = super().to_json_object()
        info['type'] = 'nt'
        info['measure_nt'] = self.measure_length
        # info['ruler_unit'] = self.ruler_unit
        if self.ruler_info is not None:
            info['ruler_info'] = self.ruler_info

        return info

    def update_ga(self):
        self.measure_length = self.length() * self.ruler_unit

    # 拿到所有的测量标注
    def get_all_measure_annotation(self, measure_mode='hadlock'):
        all_measure_annotation = {'line': [self], 'ellipse': []}

        return all_measure_annotation
