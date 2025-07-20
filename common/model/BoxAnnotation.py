
import math
from .Annotation import Annotation


class BoxAnnotation(Annotation):
    def __init__(self, ptStart=[0, 0], ptEnd=[0, 0], degree=0, is_default_value=False):
        # sourcery skip: default-mutable-arg
        super(BoxAnnotation, self).__init__(ptStart, ptEnd, is_default_value=is_default_value)
        # in radian
        self.angle = math.radians(degree)

    def degree(self):
        return math.degrees(self.angle)

    def radian(self):
        return self.angle

    def to_json_object(self):
        anno = super().to_json_object()
        anno['type'] = 'rect'
        anno['angle'] = self.angle
        return anno

    @classmethod
    def from_json(cls, annotation, offset=(0, 0)):
        anno = BoxAnnotation()
        anno._from_json(annotation, offset)
        if 'angle' in annotation:
            anno.angle = annotation['angle']
        elif 'rotation' in annotation:
            anno.angle = annotation['rotation']
        return anno

    def convert_to_local(self, pos):
        center = (self.ptStart[0] + self.ptEnd[0]) / 2, (self.ptStart[1] + self.ptEnd[1]) / 2
        x, y = pos[0] - center[0], pos[1] - center[1]

        if abs(self.angle) < 0.0001:
            return (x, y)

        # unrotate
        c = math.cos(-self.angle)
        s = math.sin(-self.angle)

        pos = (x * c - y * s, x * s + y * c)

        return pos

    def convert_to_global(self, pos):
        if abs(self.angle) < 0.0001:
            return pos

        center = (self.ptStart[0] + self.ptEnd[0]) / 2, (self.ptStart[1] + self.ptEnd[1]) / 2
        x, y = pos[0] - center[0], pos[1] - center[1]

        # rotate
        c = math.cos(self.angle)
        s = math.sin(self.angle)

        pos = (x * c - y * s + center[0], x * s + y * c + center[1])

        return pos

    def contain_point(self, pos):
        local_pos = self.convert_to_local(pos)

        if local_pos[0] < self.ptStart[0] or local_pos[0] > self.ptEnd[0]:
            return False

        if local_pos[1] < self.ptStart[1] or local_pos[1] > self.ptEnd[1]:
            return False

        return True

    def change_end_point(self, idx, pos):
        if abs(self.angle) > 0.0001:
            local_pos = self.convert_to_local(pos)
            center = self.center_point()

            pos = (local_pos[0] + center[0], local_pos[1] + center[1])

            self._change_end_point(idx, pos)

            # update the rotation angle
            new_center = self.center_point()
            offset = (new_center[0] - center[0], new_center[1] - center[1])

            # rotate
            c = math.cos(self.angle)
            s = math.sin(self.angle)

            x = c * offset[0] - s * offset[1]
            y = s * offset[0] + c * offset[1]

            x -= offset[0]
            y -= offset[1]

            self.ptStart[0] += x
            self.ptStart[1] += y

            self.ptEnd[0] += x
            self.ptEnd[1] += y
        else:
            self._change_end_point(idx, pos)

    def _change_end_point(self, idx, pos):

        if idx == 0:
            self.ptStart[0] = pos[0]
        elif idx == 1:
            self.ptStart[1] = pos[1]
        elif idx == 2:
            self.ptEnd[0] = pos[0]
        elif idx == 3:
            self.ptEnd[1] = pos[1]
