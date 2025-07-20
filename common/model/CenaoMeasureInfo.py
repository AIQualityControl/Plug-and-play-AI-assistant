#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@description:
@date       : 2022/04/30 18:52:59
@author     : Guanghua Tan
@email      : guanghuatan@hnu.edu.cn
@version    : 1.0
'''

from common.FetalBiometry import FetalBiometry
from .HcMeasureInfo import HcMeasureInfo
from .EllipseAnnotation import EllipseAnnotation
from .LineAnnotation import LineAnnotation


class CenaoMeasureInfo(HcMeasureInfo):
    def __init__(self, hc_annoatation, intergrowth_21st_bpd_annotation=None, hadlock_bpd_annotation=None, lvw_anno=None):
        '''constructor'''
        # lateral ventricle width
        self.lvw_anno = lvw_anno
        self.lvw = 0

        super(CenaoMeasureInfo, self).__init__(hc_annoatation, intergrowth_21st_bpd_annotation, hadlock_bpd_annotation)

    def update_measure_annos(self):
        super().update_measure_annos()
        if self.lvw_anno:
            self.measure_annos.append(self.lvw_anno)

    @classmethod
    def from_json(cls, json_info):
        hc_annotation = EllipseAnnotation.from_json(json_info['hc']) if 'hc' in json_info else None

        intergrowth_21st_bpd_annotation = LineAnnotation.from_json(
            json_info['intergrowth_21st_bpd']) if 'intergrowth_21st_bpd' in json_info else None

        hadlock_bpd_annotation = LineAnnotation.from_json(
            json_info['hadlock_bpd']) if 'hadlock_bpd' in json_info else None

        lvw_annotation = LineAnnotation.from_json(json_info['lvw']) if 'lvw' in json_info else None

        measure_info = CenaoMeasureInfo(hc_annotation, intergrowth_21st_bpd_annotation, hadlock_bpd_annotation,
                                        lvw_annotation)

        measure_info.parse_ruler_info(json_info)

        return measure_info

    def to_json_object(self):
        info = super().to_json_object()
        info['type'] = 'lv'

        if self.lvw_anno is not None:
            info['measure_lvw'] = self.lvw
            lvw_info = self.lvw_anno.to_json_object()
            info['lvw'] = lvw_info

        return info

    def update_ga(self, measure_mode=None):
        is_hc_plane_detected = FetalBiometry.is_hc_plane_detected
        super().update_ga()
        FetalBiometry.is_hc_plane_detected = is_hc_plane_detected

        if self.lvw_anno is not None:
            self.lvw = self.lvw_anno.length() * self.ruler_unit
        else:
            self.lvw = 0

    # 拿到所有的测量标注
    def get_all_measure_annotation(self, measure_mode='hadlock'):
        all_measure_annotation = {'line': [], 'ellipse': []}
        if measure_mode == 'hadlock':
            if self.hadlock_bpd_annotation:
                all_measure_annotation['line'].append(self.hadlock_bpd_annotation)
        if self.hc_annotation:
            all_measure_annotation['ellipse'].append(self.hc_annotation)
        if self.lvw_anno:
            all_measure_annotation['line'].append(self.lvw_anno)

        return all_measure_annotation

    def is_same_as(self, info, thresh_ratio=0.05):
        # if not super().is_same_as(info, thresh_ratio):
        #     return False
        if self.lvw == 0 or info.lvw == 0:
            return False

        dist = self.distance_between(self.lvw_anno.start_point(), info.lvw_anno.start_point())
        dist *= self.ruler_unit
        if dist > self.lvw * thresh_ratio:
            return False

        diff = abs(self.lvw - info.lvw)
        return diff < self.lvw * thresh_ratio
