import math


class MeasureInfo:
    def __init__(self):
        """constructor"""
        self.ga = 0
        self.measure_length = 0

        self.all_biometry = False
        self.disease_name_list = []

        # used to display ruler recognization result
        self.ruler_info = None

        # default init for measurement
        self.measure_score = -1

        # used to iterate all annotations
        self.measure_annos = []

        self.error_type = ''

    @property
    def ruler_unit(self):
        if isinstance(self.ruler_info, list):
            # 针对读取json的情况，
            return [ruler_info["rulerUnit"] for ruler_info in self.ruler_info]
        else:
            return self.ruler_info['rulerUnit'] if self.ruler_info else 0.0

    @property
    def ruler_unit_list(self):
        if isinstance(self.ruler_info, list):
            return [ruler_info["rulerUnit"] for ruler_info in self.ruler_info]
        elif self.ruler_info:
            return [self.ruler_info["rulerUnit"]]
        else:
            return []

    def update_measure_annos(self):
        self.measure_annos = []

    def update_ruler_info(self, ruler_info):
        self.ruler_info = ruler_info

        if (isinstance(self.ruler_unit, list) and 0.0 in self.ruler_unit) or self.ruler_unit == 0.0:
            # 读取json时不会调用AFI的ruler_unit函数
            self.error_type = 'ruler error'

    def parse_ruler_info(self, json_info):
        self.ruler_info = None
        if 'ruler_info' in json_info:
            self.ruler_info = json_info['ruler_info']
        elif 'ruler_unit' in json_info:
            self.ruler_info = {
                'rulerUnit': json_info['ruler_unit']
            }

        self.update_ga()

    @classmethod
    def distance_between(cls, start_point, end_point):
        dx = start_point[0] - end_point[0]
        dy = start_point[1] - end_point[1]
        return math.sqrt(dx * dx + dy * dy)

    def update_ga(self):
        pass

    def __iter__(self):
        """
        used to iterate all annotations
        """
        return iter(self.measure_annos)

    def clear_highlight(self):
        for anno in self.measure_annos:
            if anno:
                anno.set_highlight(False)

    def set_highlight(self, highlight=True):
        if highlight:
            pass
        else:
            self.clear_highlight()

    def clear_selection(self):
        for anno in self.measure_annos:
            if anno:
                anno.set_selected(False)

    def set_selected(self, selected=True):
        if selected:
            pass
        else:
            self.clear_selection()

    def get_highlight(self, pos, tol=8):
        for anno in self.measure_annos:
            if anno and anno.get_highlight(pos, tol):
                return anno

        return None

    def translate(self, offset):
        for anno in self.measure_annos:
            if anno:
                anno.translate(offset)

    def to_json_object(self):
        info = {
            'type': 'base'
        }

        if self.ruler_info is not None:
            info['ruler_info'] = self.ruler_info

        if self.error_type:
            info['error_type'] = self.error_type

        return info

    @classmethod
    def from_json(cls, json_info):
        info = MeasureInfo()

        if 'error_type' in json_info:
            info.error_type = json_info['error_type']

        info.parse_ruler_info(json_info)
        return info

    @staticmethod
    def get_all_measure_annotation():
        """
        拿到所有的测量标注
        """
        return {'line': [], 'ellipse': []}

    def is_same_as(self, info, thresh_ratio=0.05):
        if self.measure_length == 0 or info.measure_length == 0:
            return False

        dist = self.distance_between(self.start_point(), info.start_point())
        dist *= self.ruler_unit
        if dist > self.measure_length * thresh_ratio:
            return False

        diff = abs(self.measure_length - info.measure_length)
        return diff < self.measure_length * thresh_ratio


if __name__ == '__main__':
    measure_info = MeasureInfo()
    for anno in measure_info:
        print(anno)
