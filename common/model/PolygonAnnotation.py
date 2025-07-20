#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@description:
@date       : 2023/04/19 20:51:19
@author     : Guanghua Tan
@email      : guanghuatan@hnu.edu.cn
@version    : 1.0
'''

from .PolylineAnnotation import PolylineAnnotation


class PolygonAnnotation(PolylineAnnotation):
    def __init__(self, points=[]):
        '''constructor'''
        super(PolygonAnnotation, self).__init__(points)

    @classmethod
    def from_json(cls, anno_json, offset=(0, 0)):
        poly_anno = PolygonAnnotation()

        if 'vertex' in anno_json:
            vertex = anno_json['vertex']
            if isinstance(vertex, str):
                coords = vertex.split(';')
                vertex = []
                for coord in coords:
                    parts = coord.split(',')
                    vertex.append([float(parts[0]), float(parts[1])])

            poly_anno.points = vertex

            # to avoid convertion in base class
            del anno_json['vertex']

        poly_anno._from_json(anno_json, offset)

        return poly_anno

    def to_json_object(self):
        anno = super().to_json_object()
        anno['type'] = 'polygon'
        anno['vertex'] = self.points
        return anno

    def is_point_on(self, pos, tol=8):
        """
        whether point pos is on polyline
        """
        if super().is_point_on(pos, tol):
            return True

        dist = self.dist_to_lineseg(pos, self.points[0], self.points[-1])
        return dist < tol

    def length(self):
        len0 = super().length() + self.dist(self.points[0], self.points[-1])

        return len0
