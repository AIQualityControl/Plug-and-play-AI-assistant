#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@description:
@date       : 2022/04/16 19:25:28
@author     : Guanghua Tan
@email      : guanghuatan@hnu.edu.cn
@version    : 1.0
'''

from .LineAnnotation import LineAnnotation
from .measure_info import MeasureInfo


class PLMeasureInfo(LineAnnotation, MeasureInfo):
    def __init__(self, ptStart=[0, 0], ptEnd=[0, 0], is_default_value=False):
        # sourcery skip: default-mutable-arg
        """constructor"""
        LineAnnotation.__init__(self, ptStart, ptEnd, is_default_value=is_default_value)
        MeasureInfo.__init__(self)

        self.update_measure_annos()

    def update_measure_annos(self):
        self.measure_annos = [self]

    @classmethod
    def from_annotation(cls, line_anno):
        return PLMeasureInfo(line_anno.start_point(), line_anno.end_point())

    @classmethod
    def from_json(cls, json_info):
        pl = PLMeasureInfo()
        pl._from_json(json_info)

        pl.parse_ruler_info(json_info)
        return pl

    def to_json_object(self):
        info = super().to_json_object()
        info['type'] = 'pl'
        info['measure_pl'] = self.measure_length
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
