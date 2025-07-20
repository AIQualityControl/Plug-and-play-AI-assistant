#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@description:
@date       : 2024/01/24 10:38:41
@author     : Guanghua Tan
@email      : guanghuatan@hnu.edu.cn
@version    : 1.0
'''

from .PolylineAnnotation import PolylineAnnotation


class EnvelopAnnotation(PolylineAnnotation):
    def __init__(self, envelop_points=[], key_points_idx=[]):
        '''constructor'''
        super(EnvelopAnnotation, self).__init__(envelop_points)
        self.key_points_idx = key_points_idx

    @classmethod
    def from_json(cls, anno_json, offset=(0, 0)):
        anno = EnvelopAnnotation()
        anno._from_json(anno_json, offset)

        if 'key_points_idx' in anno_json:
            anno.key_points_idx = anno_json['key_points_idx']
        return anno

    def to_json_object(self):
        anno = super().to_json_object()
        anno['type'] = 'envelop'
        anno['key_points_idx'] = self.key_points_idx
        return anno

    def num_of_keypoints(self):
        return len(self.key_points_idx)

    def snap_edit_points(self, pos, tol=8):

        min_idx = -1
        min_dist = tol * tol
        for i, idx in enumerate(self.key_points_idx):
            dist = self.square_dist(self.points[idx], pos)
            if dist < min_dist:
                min_dist = dist
                min_idx = i

        return min_idx

    def change_end_point(self, idx, pos):
        # snap point
        min_idx = -1
        min_dist = 10 * 10
        for i, pt in enumerate(self.points):
            dist = self.square_dist(pt, pos)
            if dist < min_dist:
                min_dist = dist
                min_idx = i

        if min_idx >= 0:
            self.key_points_idx[idx] = min_idx
