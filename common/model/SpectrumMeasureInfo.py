#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@description:
@date       : 2023/08/23 15:59:53
@author     : Guanghua Tan
@email      : guanghuatan@hnu.edu.cn
@version    : 1.0
'''

from .measure_info import MeasureInfo
from .EnvelopAnnotation import EnvelopAnnotation
from .LineAnnotation import LineAnnotation


class SpectrumMeasureInfo(MeasureInfo):

    def __init__(self, parameters, envelop=None, num_valley_points=0, above_flag=True, hr_lines=[],
                 speed_unit=1.0, time_unit=1.0, x_axis=0):
        """
        envelop: canbe envelop annotation or (envelop, key_points_idx, offset)
        hr_lines: canbe line annotations or (list of points, offset)
        从测量进程传输到显示进程时，避免内存占用太大，不转换envelop和key_points_idx
        """
        super(SpectrumMeasureInfo, self).__init__()

        self.parameters = parameters

        # canbe envelop annotation or (envelop, key_points_idx, offset)
        self.envelop = envelop

        # above the x axis
        self.above_flag = above_flag

        self.num_valley_points = num_valley_points

        # speed unit
        self.speed_unit = speed_unit
        self.time_unit = time_unit
        self.x_axis = x_axis

        self.hr_lines = []
        if hr_lines:
            for line in hr_lines:
                line_anno = line if isinstance(line, LineAnnotation) else LineAnnotation(line[0], line[1])
                line_anno.parent = self
                self.hr_lines.append(line_anno)

        self.update_measure_annos()

    @property
    def ruler_unit(self):
        return self.speed_unit

    def update_measure_annos(self):
        self.measure_annos = []
        if self.hr_lines:
            self.measure_annos.extend(self.hr_lines)

        if self.envelop:
            self.measure_annos.append(self.envelop)

    def convert_envelop(self):
        if self.envelop and isinstance(self.envelop, (list, tuple)):
            envelop, key_points_idx, offset = self.envelop
            points = [[i + offset[0], envelop[i] + offset[1]]
                      for i in range(len(envelop))]
            self.envelop = EnvelopAnnotation(points, key_points_idx)

            self.update_measure_annos()

    @classmethod
    def from_json(cls, json_info):

        envelop = EnvelopAnnotation.from_json(json_info['envelop']) if 'envelop' in json_info else None

        parameters = json_info['parameters'] if 'parameters' in json_info else {}

        above_flag = json_info['above_flag'] if 'above_flag' in json_info else True

        num_valley_points = json_info['num_valley_points'] if 'num_valley_points' in json_info else 0

        speed_unit = json_info['speed_unit'] if 'speed_unit' in json_info else 1.0
        time_unit = json_info['time_unit'] if 'time_unit' in json_info else 1.0
        x_axis = json_info['x_axis'] if 'x_axis' in json_info else 0

        hr_lines = []
        if 'hr_lines' in json_info:
            for line_json in json_info['hr_lines']:
                line = LineAnnotation.from_json(line_json)
                hr_lines.append(line)

        measure_info = SpectrumMeasureInfo(parameters, envelop, num_valley_points, above_flag, hr_lines,
                                           speed_unit=speed_unit, time_unit=time_unit, x_axis=x_axis)

        measure_info.parse_ruler_info(json_info)

        return measure_info

    def to_json_object(self):
        info = {
            'type': 'spectrum',
            'above_flag': self.above_flag,
            'num_valley_points': self.num_valley_points,
            'speed_unit': self.speed_unit,
            'time_unit': self.time_unit,
            'x_axis': self.x_axis
        }
        if self.parameters:
            info['parameters'] = self.parameters

        if self.envelop is not None:
            info['envelop'] = self.envelop.to_json_object()

        if self.ruler_info is not None:
            info['ruler_info'] = self.ruler_info

        if self.hr_lines:
            info['hr_lines'] = [line.to_json_object() for line in self.hr_lines]

        return info

    def recalc_parameters(self):

        # have to save speed unit and x-axis
        x_axis = self.x_axis

        edv = 0
        for i in range(self.num_valley_points):
            idx = self.envelop.key_points_idx[i]
            edv += (x_axis - self.envelop.points[idx][1])
        if self.num_valley_points > 1:
            edv /= self.num_valley_points

        psv = 0
        for i in range(self.num_valley_points, self.envelop.num_of_keypoints()):
            idx = self.envelop.key_points_idx[i]
            psv += (x_axis - self.envelop.points[idx][1])
        num_psv = self.envelop.num_of_keypoints() - self.num_valley_points
        if num_psv > 1:
            psv /= num_psv

        # sd
        # ri
        if psv == 0 or edv == 0:
            self.parameters['SD'] = 0
            self.parameters['RI'] = 0
        else:
            self.parameters['SD'] = psv / edv
            self.parameters['RI'] = (psv - edv) / psv

        # old vm
        vm = abs(self.parameters['PSV'] - self.parameters['EDV'])

        # pi
        # vm will not be changed
        vm = (self.parameters['PSV'] - self.parameters['EDV']) / self.parameters['PI'] if self.parameters['PI'] > 0 else 0

        edv *= self.speed_unit
        self.parameters['EDV'] = edv

        psv *= self.speed_unit
        self.parameters['PSV'] = psv

        if vm != 0:
            self.parameters['PI'] = abs((psv - edv) / vm)

    # 拿到所有的测量标注

    def get_all_measure_annotation(self, measure_mode='hadlock'):
        all_measure_annotation = {'line': [self], 'ellipse': []}

        return all_measure_annotation
