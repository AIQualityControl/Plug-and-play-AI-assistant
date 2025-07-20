#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@description:
@date       : 2022/04/30 19:44:37
@author     : Guanghua Tan
@email      : guanghuatan@hnu.edu.cn
@version    : 1.0
'''

from .LineAnnotation import LineAnnotation
from .measure_info import MeasureInfo


class XiaonaoMeasureInfo(MeasureInfo):
    def __init__(self, tcd_anno=None):
        '''constructor'''
        super(XiaonaoMeasureInfo, self).__init__()

        # trans-cerebrellum diameter
        self.tcd_anno = tcd_anno
        self.tcd = 0

        self.update_measure_annos()

    def update_measure_annos(self):
        self.measure_annos = [self.tcd_anno] if self.tcd_anno else []

    @classmethod
    def from_json(cls, json_info):
        tcd_anno = LineAnnotation.from_json(json_info['tcd'])

        measure_info = XiaonaoMeasureInfo(tcd_anno)
        measure_info.parse_ruler_info(json_info)

        return measure_info

    def to_json_object(self):

        info = {
            'type': 'tc',
            'measure_tcd': self.tcd,
            'tcd': self.tcd_anno.to_json_object(),
        }
        if self.ruler_info is not None:
            info['ruler_info'] = self.ruler_info

        return info

    def update_ga(self):
        self.tcd = self.tcd_anno.length() * self.ruler_unit

    # 拿到所有的测量标注
    def get_all_measure_annotation(self, measure_mode='hadlock'):
        all_measure_annotation = {'line': [], 'ellipse': []}
        if self.tcd_anno:
            all_measure_annotation['line'].append(self.tcd_anno)

        return all_measure_annotation

    def is_same_as(self, info, thresh_ratio=0.05):
        if self.tcd == 0 or info.tcd == 0:
            return False

        dist = self.distance_between(self.tcd_anno.start_point(), info.tcd_anno.start_point())
        dist *= self.ruler_unit
        if dist > self.tcd * thresh_ratio:
            return False

        diff = abs(self.tcd - info.tcd)
        return diff < self.tcd * thresh_ratio
