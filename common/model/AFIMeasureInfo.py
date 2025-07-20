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


class AFIMeasureInfo(MeasureInfo):
    def __init__(self, amniotic_annos=[]):
        '''constructor'''
        super().__init__()
        # 1 or 4 vertical depth annotation
        self.amniotic_annos = amniotic_annos
        self.amniotic_depths = []
        self.ruler_info = None

        self.afi = 0
        self.afv = 0

        self.update_measure_annos()

    def update_measure_annos(self):
        self.measure_annos = self.amniotic_annos

    @property
    def ruler_unit(self):
        return self.ruler_info[0]['rulerUnit'] if self.ruler_info else 0.0

    @property
    def ruler_unit_list(self):
        return [ruler_info['rulerUnit'] for ruler_info in self.ruler_info] if self.ruler_info else []

    @classmethod
    def from_json(cls, json_info):

        afi = AFIMeasureInfo()

        afi.amniotic_annos = [LineAnnotation.from_json(anno_json) for anno_json in json_info['annos']]
        afi.update_measure_annos()

        afi.parse_ruler_info(json_info)

        return afi

    def parse_ruler_info(self, json_info):
        # 重写AFIMeasureInfo的json读取方法
        if 'ruler_info' in json_info:
            self.ruler_info = json_info['ruler_info']
            if not isinstance(self.ruler_info, list):
                self.ruler_info = [self.ruler_info]
        elif 'ruler_unit' in json_info:
            self.ruler_info = [{
                'rulerUnit': json_info['ruler_unit']
            }]

        self.update_ga()

    def to_json_object(self):
        info = {
            'type': 'afi',
            'afi': self.afi,
            'afv': self.afv,
            'annos': [anno.to_json_object() for anno in self.amniotic_annos]
        }

        if self.ruler_info:
            for i, depth in enumerate(self.amniotic_depths):
                info[f'measure_Q{i + 1}'] = depth

            info['ruler_info'] = self.ruler_info

        return info

    def update_ga(self):
        self.amniotic_depths.clear()

        if self.ruler_info:
            # 兼容原来json读取方式
            for anno, ruler_info in zip(self.amniotic_annos, self.ruler_info):
                depth = anno.length() * ruler_info['rulerUnit']
                self.amniotic_depths.append(depth)

        # compute afi
        if len(self.amniotic_depths) == 0:
            self.afi = 0
            self.afv = 0
        elif len(self.amniotic_depths) == 1:
            self.afi = 0
            self.afv = self.amniotic_depths[0]
        else:
            self.afi = sum(self.amniotic_depths)
            self.afv = max(self.amniotic_depths)

    # /////////////// selection

    def change_end_point(self, idx, pos):
        pass

    # 拿到所有的测量标注
    def get_all_measure_annotation(self, measure_mode='hadlock'):
        all_measure_annotation = {'line': [], 'ellipse': []}

        for each_anno in self.amniotic_annos:
            all_measure_annotation['line'].append(each_anno)

        return all_measure_annotation

    def get_end_points(self):
        end_points = []
        for anno in self.amniotic_annos:
            end_points.extend(anno.start_point())
            end_points.extend(anno.end_point())

        return end_points

    def is_same_as(self, info, thresh_ratio=0.05):
        if len(self.amniotic_depths) != len(info.amniotic_depths):
            return False

        for depth0, depth1 in zip(self.amniotic_depths, info.amniotic_depths):
            diff = abs(depth0 - depth1)
            if diff > depth0 * thresh_ratio:
                return False

        # for anno0, anno1, depth in zip(self.amniotic_annos, info.amniotic_annos, self.amniotic_depths):
        #     dist = self.distance_between(anno0.start_point(), anno1.start_point())
        #     dist *= self.ruler_unit

        #     if dist > depth * thresh_ratio:
        #         return False

        return True
