#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@description:
@date       : 2022/04/15 18:04:58
@author     : Guanghua Tan
@email      : guanghuatan@hnu.edu.cn
@version    : 1.0
'''
import copy
from .LineAnnotation import LineAnnotation
from .measure_info import MeasureInfo


class CRLMeasureInfo(MeasureInfo):
    def __init__(self, crl_info=None, corrected_info=None, center_point=None, measure_score=0, is_default_value=False):
        # sourcery skip: default-mutable-arg
        """constructor"""
        MeasureInfo.__init__(self)

        self.measure_score = measure_score

        self.measure_annos = []
        # crl
        if crl_info:
            self.crl_anno = LineAnnotation(copy.deepcopy(crl_info[0]), copy.deepcopy(crl_info[1]),
                                           is_default_value=is_default_value)
            self.crl_length = 0
        else:
            self.crl_anno = None
            self.crl_length = 0

        # used for corrected crl
        if corrected_info:
            self.corrected_crl_anno = LineAnnotation(copy.deepcopy(corrected_info[0]), copy.deepcopy(corrected_info[1]),
                                                     is_default_value=is_default_value)
            self.corrected_crl_length = 0
        else:
            self.corrected_crl_anno = None
            self.corrected_crl_length = 0

        self.center_point = center_point

        self.update_measure_annos()

        self.all_biometry = False

    @classmethod
    def from_json(cls, json_info):

        crl = CRLMeasureInfo()

        if 'crl' in json_info:
            crl.crl_anno = LineAnnotation.from_json(json_info['crl'])

        if 'corrected_crl' in json_info:
            crl.corrected_crl_anno = LineAnnotation.from_json(json_info['corrected_crl'])

        if 'center_point' in json_info:
            crl.center_point = json_info['center_point']

        crl.update_measure_annos()
        crl.parse_ruler_info(json_info)
        return crl

    def update_measure_annos(self):
        self.measure_annos = []
        if self.crl_anno:
            self.measure_annos.append(self.crl_anno)

        if self.corrected_crl_anno:
            self.measure_annos.append(self.corrected_crl_anno)

    def to_json_object(self):
        info = super().to_json_object()
        info['type'] = 'crl'
        info['measure_crl'] = self.crl_length

        if self.crl_anno:
            info['crl'] = self.crl_anno.to_json_object()

        if self.corrected_crl_anno:
            info['corrected_crl'] = self.corrected_crl_anno.to_json_object()

        if self.center_point:
            info['center_point'] = self.center_point

        return info

    def update_ga(self):
        self.crl_length = self.crl_anno.length() * self.ruler_unit if self.crl_anno else 0
        self.corrected_crl_length = self.corrected_crl_anno.length() * self.ruler_unit if self.corrected_crl_anno else 0

    def is_corrected(self):
        return self.corrected_crl_anno is not None

    def is_same_as(self, info, thresh_ratio=0.05):
        if self.crl_length == 0 or info.crl_length == 0:
            return False

        dist = self.distance_between(self.crl_anno.start_point(), info.crl_anno.start_point())
        dist *= self.ruler_unit
        if dist > self.crl_length * thresh_ratio:
            return False

        diff = abs(self.crl_length - info.crl_length)
        if diff > self.crl_length * thresh_ratio:
            return False

        diff = abs(self.corrected_crl_length - info.corrected_crl_length)
        return diff <= self.corrected_crl_length * thresh_ratio  # 0 <= 0 True

    def translate(self, offset):
        super().translate(offset)

        if self.center_point:
            self.center_point[0] += offset[0]
            self.center_point[1] += offset[1]

    def snap_edit_points(self, pos, tol=8):

        if self.crl_anno:
            idx = self.crl_anno.snap_edit_points(pos, tol)
            if idx >= 0:
                return idx

        if self.corrected_crl_anno:
            idx = self.corrected_crl_anno.snap_edit_points(pos, tol)
            if idx >= 0:
                return idx + 2

        if self.center_point:
            dist0 = LineAnnotation.square_dist(pos, self.center_point)
            if dist0 < tol * tol:
                return 4

        return -1

    def change_end_point(self, idx, pos):
        if idx < 0 or idx > 4:
            return
        if idx < 2 and self.crl_anno:
            self.crl_anno.change_end_point(idx, pos)
        elif idx < 4 and self.corrected_crl_anno:
            self.corrected_crl_anno.change_end_point(idx, pos)
        elif idx == 4 and self.center_point:
            self.center_point[0] = pos[0]
            self.center_point[1] = pos[1]

    # 拿到所有的测量标注
    def get_all_measure_annotation(self, measure_mode='hadlock'):
        all_measure_annotation = {'line': [self.crl_anno], 'ellipse': []}

        if self.corrected_crl_anno:
            all_measure_annotation['line'].append(self.corrected_crl_anno)

        return all_measure_annotation
