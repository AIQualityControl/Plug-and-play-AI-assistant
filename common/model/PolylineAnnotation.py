#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@description:
@date       : 2023/04/19 20:51:19
@author     : Guanghua Tan
@email      : guanghuatan@hnu.edu.cn
@version    : 1.0
'''

from .Annotation import Annotation


class PolylineAnnotation(Annotation):
    def __init__(self, points=[]):
        '''constructor'''
        super(PolylineAnnotation, self).__init__()
        self.points = points

    def num_of_points(self):
        return len(self.points)

    def set_point_at(self, idx, point):
        self.points[idx] = point

    def point_at(self, idx):
        return self.points[idx]

    def all_points(self):
        return self.points

    @classmethod
    def from_json(cls, anno_json, offset=(0, 0)):
        poly_anno = PolylineAnnotation()
        poly_anno._from_json(anno_json, offset)

        return poly_anno

    def _from_json(self, anno_json, offset=(0, 0)):
        super()._from_json(anno_json, offset)

        if 'vertex' in anno_json:
            self.points = anno_json['vertex']

    def to_json_object(self):
        anno = super().to_json_object()
        anno['type'] = 'polyline'
        anno['vertex'] = self.points
        return anno

    def is_point_on(self, pos, tol=8):
        """
        whether point pos is on polyline
        """
        for i in range(1, len(self.points)):
            dist = self.dist_to_lineseg(pos, self.points[i - 1], self.points[i])
            if dist < tol:
                return True

        return False

    def length(self):
        len0 = 0
        for i in range(1, len(self.points)):
            len0 += self.dist(self.points[i], self.points[i - 1])

        return len0

    def translate(self, offset):
        for pt in self.points:
            pt[0] += offset[0]
            pt[1] += offset[1]

    def snap_edit_points(self, pos, tol=8):

        min_idx = -1
        min_dist = tol * tol
        for i, pt in enumerate(self.points):
            dist = self.square_dist(pt, pos)
            if dist < min_dist:
                min_dist = dist
                min_idx = i

        return min_idx

    def change_end_point(self, idx, pos):
        self.points[idx] = [pos[0], pos[1]]

    def translate_endpoint(self, idx, offset):
        self.points[idx][0] += offset[0]
        self.points[idx][1] += offset[1]
