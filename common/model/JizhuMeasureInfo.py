#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@description:
@date       : 2023/06/06 17:39:35
@author     : Guanghua Tan
@email      : guanghuatan@hnu.edu.cn
@version    : 1.0
'''
from .measure_info import MeasureInfo
from ..model.LineAnnotation import LineAnnotation
from ..model.EllipseAnnotation import EllipseAnnotation


class JizhuMeasureInfo(MeasureInfo):
    def __init__(self, vetebral_list, cone2skin=None, cone2end=None):
        '''constructor'''
        super(JizhuMeasureInfo, self).__init__()

        self.vetebral_list = vetebral_list

        self.cone2skin_anno = cone2skin
        self.cone2skin_len = 0

        self.cone2end_anno = cone2end
        self.cone2end_len = 0

        self.update_measure_annos()

    def update_measure_annos(self):
        self.measure_annos = []
        self.measure_annos.extend(self.vetebral_list)

        if self.cone2skin_anno:
            self.measure_annos.append(self.cone2skin_anno)

        if self.cone2end_anno:
            self.measure_annos.append(self.cone2end_anno)

    @classmethod
    def from_json(cls, json_info):

        cone2skin = LineAnnotation.from_json(json_info['cone2skin']) if 'cone2skin' in json_info else None
        cone2end = LineAnnotation.from_json(json_info['cone2end']) if 'cone2end' in json_info else None

        vetebral_list = []
        if 'vetebrals' in json_info:
            for veteb_json in json_info['vetebrals']:
                veteb_anno = EllipseAnnotation.from_json(veteb_json)
                vetebral_list.append(veteb_anno)

        measure_info = JizhuMeasureInfo(vetebral_list, cone2skin=cone2skin, cone2end=cone2end)

        measure_info.parse_ruler_info(json_info)

        return measure_info

    def to_json_object(self):
        info = {
            'type': 'veteb'
        }
        if self.vetebral_list:
            info['vetebrals'] = [veteb.to_json_object() for veteb in self.vetebral_list]
        if self.cone2skin_anno:
            info['cone2skin'] = self.cone2skin_anno.to_json_object()
            info['cone2skin_len'] = self.cone2end_len
        if self.cone2end_anno:
            info['cone2end'] = self.cone2end_anno.to_json_object()
            info['cone2end_len'] = self.cone2end_len

        if self.ruler_info is not None:
            info['ruler_info'] = self.ruler_info

        return info

    def update_ga(self):
        if self.cone2skin_anno:
            self.cone2skin_len = self.cone2skin_anno.length() * self.ruler_unit
        else:
            self.cone2skin_len = 0

        if self.cone2end_anno:
            self.cone2end_len = self.cone2end_anno.length() * self.ruler_unit
        else:
            self.cone2end_len = 0

    def change_end_point(self, idx, pos):
        pass

    # 拿到所有的测量标注
    def get_all_measure_annotation(self, measure_mode='hadlock'):
        all_measure_annotation = {'line': [], 'ellipse': []}
        if self.cone2skin_anno:
            all_measure_annotation['line'].append(self.cone2skin_anno)
        if self.cone2end_anno:
            all_measure_annotation['line'].append(self.cone2end_anno)

        if self.vetebral_list:
            all_measure_annotation['ellipse'] = self.vetebral_list

        return all_measure_annotation

    def is_same_as(self, info, thresh_ratio=0.05):
        if len(self.vetebral_list) != len(info.vetebral_list):
            return False

        diff = abs(self.cone2end_len - info.cone2end_len)
        if diff > self.cone2end_len * thresh_ratio:
            return False

        diff = abs(self.cone2skin_len - info.cone2skin_len)
        return diff <= self.cone2skin_len * thresh_ratio  # 0 <= 0 True
