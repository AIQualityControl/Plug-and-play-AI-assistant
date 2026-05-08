from .Annotation import Annotation
from .LineAnnotation import LineAnnotation
import math


class BiDiameterAnnotation(Annotation):
    def __init__(self, major_axis=None, minor_axis=None):
        """
        ptStart and ptEnd is used for bounding box
        """
        super(BiDiameterAnnotation, self).__init__()

        self.nodule_number = None

        self.major_axis = major_axis
        if major_axis and isinstance(major_axis, (list, tuple)):
            self.major_axis = LineAnnotation(major_axis[0], major_axis[1])

        self.minor_axis = minor_axis
        if minor_axis and isinstance(minor_axis, (list, tuple)):
            self.minor_axis = LineAnnotation(minor_axis[0], minor_axis[1])

        self.major_len = 0
        self.minor_len = 0

    def center_point(self):
        pt_center = self.major_axis.center_point() if self.major_axis else (0, 0)
        if self.minor_axis:
            x, y = self.minor_axis.center_point()
            pt_center = (pt_center[0] + x) / 2, (pt_center[1] + y) / 2

        return pt_center

    def square_area(self):
        if not self.major_axis or not self.minor_axis:
            return 0

        return self.major_axis.square_length() * self.minor_axis.square_length()

    def area(self):
        return math.sqrt(self.square_area())

    def min_y_point(self):
        pt_list = []
        if self.major_axis:
            pt_list.append(self.major_axis.start_point())
            pt_list.append(self.major_axis.end_point())
        if self.minor_axis:
            pt_list.append(self.minor_axis.start_point())
            pt_list.append(self.minor_axis.end_point())

        pt = min(pt_list, key=lambda pt: pt[1])
        return pt

    @classmethod
    def from_json(cls, json_info):

        major_axis = None
        if 'major_axis' in json_info and json_info['major_axis']:
            major_axis = LineAnnotation.from_json(json_info['major_axis'])

        minor_axis = None
        if 'minor_axis' in json_info and json_info['minor_axis']:
            minor_axis = LineAnnotation.from_json(json_info['minor_axis'])

        anno = BiDiameterAnnotation(major_axis, minor_axis)
        anno._from_json(json_info)

        return anno

    def to_json_object(self):
        anno = super().to_json_object()

        anno['type'] = 'bidiameter'
        if self.name == 'nodule':
            anno['nodule_number'] = self.nodule_number
        anno['measure_major_len'] = self.major_len
        anno['measure_minor_len'] = self.minor_len
        anno['major_axis'] = self.major_axis.to_json_object() if self.major_axis else {}
        anno['minor_axis'] = self.minor_axis.to_json_object() if self.minor_axis else {}

        return anno

    def translate(self, offset):
        super().translate(offset)
        if self.major_axis:
            self.major_axis.translate(offset)

        if self.minor_axis:
            self.minor_axis.translate(offset)

    def update_actual_length(self, ruler_unit):
        self.major_len = self.major_axis.length() * ruler_unit if self.major_axis else 0
        self.minor_len = self.minor_axis.length() * ruler_unit if self.minor_axis else 0

    def get_highlight(self, pos, tol=8):
        if self.major_axis and self.major_axis.get_highlight(pos, tol):
            return self.major_axis

        if self.minor_axis and self.minor_axis.get_highlight(pos, tol):
            return self.minor_axis

    def clear_highlight(self):
        if self.major_axis:
            self.major_axis.set_highlight(False)

        if self.minor_axis:
            self.minor_axis.set_highlight(False)

        super().set_highlight(False)

    def set_highlight(self, highlight=True):
        if highlight:
            super().set_highlight(highlight)
        else:
            self.clear_highlight()

    def clear_selection(self):
        if self.major_axis:
            self.major_axis.set_selected(False)

        if self.minor_axis:
            self.minor_axis.set_selected(False)

        super().set_selected(False)

    def set_selected(self, selected=True):
        if selected:
            super().set_selected(selected)
        else:
            self.clear_selection()

    def snap_edit_points(self, pos, tol=8):

        if self.major_axis:
            dist0 = self.square_dist(pos, self.major_axis.start_point())
            if dist0 < tol * tol:
                return 0

            dist1 = self.square_dist(pos, self.major_axis.end_point())
            if dist1 < tol * tol:
                return 1

        if self.minor_axis:
            dist0 = self.square_dist(pos, self.minor_axis.start_point())
            if dist0 < tol * tol:
                return 2

            dist1 = self.square_dist(pos, self.minor_axis.end_point())
            if dist1 < tol * tol:
                return 3

        return -1

    def change_end_point(self, idx, pos):
        if idx == 0:
            self.major_axis.ptStart = [pos[0], pos[1]]
        elif idx == 1:
            self.major_axis.ptEnd = [pos[0], pos[1]]
        elif idx == 2:
            self.minor_axis.ptStart = [pos[0], pos[1]]
        elif idx == 3:
            self.minor_axis.ptEnd = [pos[0], pos[1]]
