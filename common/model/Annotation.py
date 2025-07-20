import copy
import json
import math


class Annotation:
    def __init__(self, ptStart=[0, 0], ptEnd=[0, 0], is_default_value=False):
        # sourcery skip: default-mutable-arg
        self.ptStart = ptStart
        self.ptEnd = ptEnd
        self.name = ''
        self.score = 0
        # gestation age
        # self.ga = 0

        # whether is selected or highlight
        self.flag = 0

        # 记录当前激活端点的索引(因为是线段标注，取值只要两个，0: 开始的端点, 1: 结束的端点)
        self.active_endpoint_idx = -1

        # 标记当前测量切面的测量值是否为默认值
        self.is_default_value = is_default_value

        # custom properties
        self.custom_props = None

        # 一些标注是由线段等基本标注构成的，parent指向整体标注
        self.parent = None

    def get_name(self):
        return f'{self.name}-{self.track_id}' if hasattr(self, 'track_id') and self.track_id >= 0 else self.name

    def _from_json(self, annotation, offset=(0, 0)):
        if isinstance(annotation, str):
            annotation = json.loads(annotation)

        self.name = annotation['name'] if 'name' in annotation else ''
        self.score = float(annotation['score']) if 'score' in annotation else 0
        self.is_default_value = annotation['is_default_value'] if 'is_default_value' in annotation else False

        if 'vertex' in annotation:
            str_vertex = annotation['vertex']
            if isinstance(str_vertex, str):
                parts = str_vertex.split(',')
                self.ptStart = [int(float(parts[0]) + 0.5), int(float(parts[1]) + 0.5)]
                self.ptEnd = [int(float(parts[2]) + 0.5), int(float(parts[3]) + 0.5)]
            elif isinstance(str_vertex, list):
                if len(str_vertex) == 1 and len(str_vertex[0]) == 4:
                    # roi
                    roi = str_vertex[0]
                    self.ptStart = [int(roi[0] + 0.5), int(roi[1] + 0.5)]
                    self.ptEnd = [int(roi[2] + 0.5), int(roi[3] + 0.5)]
                elif len(str_vertex) == 2:
                    # self.ptStart = str_vertex[0]
                    # self.ptEnd = str_vertex[1]
                    self.ptStart = copy.copy(str_vertex[0])
                    self.ptEnd = copy.copy(str_vertex[1])
                elif len(str_vertex) == 4 and not isinstance(str_vertex[0], (list, tuple)):
                    self.ptStart = str_vertex[0:2]
                    self.ptEnd = str_vertex[2:]
        else:
            if 'start' in annotation:
                parts = annotation['start'].split(',')
                self.ptStart = [float(parts[0]), float(parts[1])]

            if 'end' in annotation:
                parts = annotation['end'].split(',')
                self.ptEnd = [float(parts[0]), float(parts[1])]

        self.ptStart[0] += offset[0]
        self.ptStart[1] += offset[1]

        self.ptEnd[0] += offset[0]
        self.ptEnd[1] += offset[1]

        if 'custom_props' in annotation:
            self.custom_props = annotation['custom_props']
        elif 'cls_props' in annotation:
            self.custom_props['cls_props'] = annotation['cls_props']

        if 'track_id' in annotation:
            self.track_id = annotation['track_id']

    @classmethod
    def from_json(cls, annotation):
        anno = Annotation()
        anno._from_json(annotation)
        return anno

    def to_json_object(self):
        anno = {
            'type': 'rect',
            'name': self.name,
            'score': float(self.score),
            'vertex': [[float(self.ptStart[0]), float(self.ptStart[1])],
                       [float(self.ptEnd[0]), float(self.ptEnd[1])]],
        }

        if hasattr(self, "is_default_value") and self.is_default_value:
            # 仅为true的时候保存
            anno["is_default_value"] = self.is_default_value

        if self.custom_props:
            anno['custom_props'] = self.custom_props

        if hasattr(self, 'track_id'):
            anno['track_id'] = self.track_id

        return anno

    def contain_point(self, pos):
        if pos[0] < self.ptStart[0] or pos[0] > self.ptEnd[0]:
            return False

        if pos[1] < self.ptStart[1] or pos[1] > self.ptEnd[1]:
            return False

        return True

    def start_point(self):
        return self.ptStart

    def end_point(self):
        return self.ptEnd

    def center_point(self):
        return (self.ptStart[0] + self.ptEnd[0]) / 2, (self.ptStart[1] + self.ptEnd[1]) / 2

    def square_length(self):
        x = self.ptEnd[0] - self.ptStart[0]
        y = self.ptEnd[1] - self.ptStart[1]
        return x * x + y * y

    def size(self):
        return int(abs(self.ptEnd[0] - self.ptStart[0])), int(abs(self.ptEnd[1] - self.ptStart[1]))

    def half_size(self):
        return int(abs(self.ptEnd[0] - self.ptStart[0]) / 2), int(abs(self.ptEnd[1] - self.ptStart[1]) / 2)

    def length(self):
        return math.sqrt(self.square_length())

    def is_selected(self):
        return (self.flag & 0x0001) != 0

    def set_selected(self, selected=True):
        if selected:
            self.flag |= 0x0001
        else:
            self.flag &= (~0x0001)

    def is_highlight(self):
        return (self.flag & 0x0002) != 0

    def set_highlight(self, highlight=True):
        if highlight:
            self.flag |= 0x0002
        else:
            self.flag &= (~0x0002)

    @classmethod
    def square_dist(cls, pt0, pt1):
        offset = (pt1[0] - pt0[0], pt1[1] - pt0[1])
        return offset[0] * offset[0] + offset[1] * offset[1]

    @classmethod
    def dist(cls, pt0, pt1):
        return math.sqrt(cls.square_dist(pt0, pt1))

    def translate(self, offset):
        self.ptStart[0] += offset[0]
        self.ptStart[1] += offset[1]

        self.ptEnd[0] += offset[0]
        self.ptEnd[1] += offset[1]

    @staticmethod
    def DEGREE_TO_RAD(x):
        return 0.01745329252 * x

    @staticmethod
    def RAD_TO_DEGREE(x):
        return 57.295779513 * x

    @classmethod
    def dist_to_lineseg(cls, point, start, end):
        """
        linseg is represented as [start, end]
        """
        # normalize
        dir = [end[0] - start[0], end[1] - start[1]]
        length = math.sqrt(dir[0] * dir[0] + dir[1] * dir[1])
        if length < 1.0e-6:
            # if line segment is too short
            return cls.dist(point, [(start[0] + end[0]) * 0.5, (start[1] + end[1]) * 0.5])
        dir = [dir[0] / length, dir[1] / length]

        vec = [point[0] - start[0], point[1] - start[1]]

        # dot product to get projection length
        proj = vec[0] * dir[0] + vec[1] * dir[1]
        if proj <= 0:
            return cls.dist(point, start)
        elif proj >= length:
            return cls.dist(point, end)

        # cross product to get distance
        dist = abs(vec[0] * dir[1] - vec[1] * dir[0])
        return dist

    def is_point_on(self, pos, tol=8):
        return False

    def get_highlight(self, pos, tol=8):
        if self.is_point_on(pos, tol):
            self.set_highlight(True)
            return self

        return None

    # 拿到激活端点的索引
    def get_active_endpoint_idx(self):
        return self.active_endpoint_idx

    # 设置激活端点的索引
    def set_active_endpoint_idx(self, idx):
        self.active_endpoint_idx = idx

    # 切换激活的端点
    def switch_active_endpoint(self):
        active_endpoint_idx = 0
        if not self.active_endpoint_idx:
            active_endpoint_idx = 1

        self.active_endpoint_idx = active_endpoint_idx

    # 平移端点
    def translate_endpoint(self, idx, offset):
        if idx == 0:
            self.ptStart[0] += offset[0]
            self.ptStart[1] += offset[1]
        elif idx == 1:
            self.ptEnd[0] += offset[0]
            self.ptEnd[1] += offset[1]
