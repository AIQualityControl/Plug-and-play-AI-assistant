
from math import sqrt
from .Annotation import Annotation


class LineAnnotation(Annotation):

    def __init__(self, ptStart=[0, 0], ptEnd=[0, 0], is_default_value=False):
        # sourcery skip: default-mutable-arg
        super(LineAnnotation, self).__init__(ptStart, ptEnd, is_default_value)

        # 记录当前激活端点的索引(因为是线段标注，取值只要两个，0: 开始的端点, 1: 结束的端点)
        self.active_endpoint_idx = -1

    @classmethod
    def from_json(cls, anno_json, offset=(0, 0)):
        anno = LineAnnotation()
        anno._from_json(anno_json, offset)
        return anno

    def to_json_object(self):
        anno = super().to_json_object()
        anno['type'] = 'line'
        return anno

    def dir(self):
        return (self.ptEnd[0] - self.ptStart[0], self.ptEnd[1] - self.ptStart[1])

    def normalized_dir(self):
        vec = self.dir()

        # normalize
        len = vec[0] * vec[0] + vec[1] * vec[1]
        if len < 1.0e-8:
            return (0, 0)

        len = sqrt(len)
        return (vec[0] / len, vec[1] / len)

    def is_point_on(self, pos, tol=8):
        """
        whether point pos is on line
        """
        dir = self.normalized_dir()
        vec = (pos[0] - self.ptStart[0], pos[1] - self.ptStart[1])

        # cross product
        dist = abs(vec[0] * dir[1] - vec[1] * dir[0])
        if dist > tol:
            return False

        # whether is between line segment
        proj = vec[0] * dir[0] + vec[1] * dir[1]
        if proj < 0:
            return self.square_dist(pos, self.ptStart) < tol * tol

        if proj > self.length():
            return self.square_dist(pos, self.ptEnd) < tol * tol

        return True

    def snap_edit_points(self, pos, tol=8):

        dist0 = self.square_dist(pos, self.ptStart)
        if dist0 < tol * tol:
            return 0

        dist1 = self.square_dist(pos, self.ptEnd)
        if dist1 < tol * tol:
            return 1

        return -1

    def change_end_point(self, idx, pos):
        if idx == 0:
            self.ptStart[0] = pos[0]
            self.ptStart[1] = pos[1]
        elif idx == 1:
            self.ptEnd[0] = pos[0]
            self.ptEnd[1] = pos[1]
