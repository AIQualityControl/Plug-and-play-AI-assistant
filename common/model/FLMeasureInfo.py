from common.FetalBiometry import FetalBiometry
from common.model.measure_info import MeasureInfo
from common.model.LineAnnotation import LineAnnotation


class FLMeasureInfo(LineAnnotation, MeasureInfo):
    def __init__(self, ptStart=[0, 0], ptEnd=[0, 0], measure_score=0, is_default_value=False):
        # sourcery skip: default-mutable-arg
        """constructor"""
        LineAnnotation.__init__(self, ptStart, ptEnd, is_default_value=is_default_value)
        MeasureInfo.__init__(self)

        self.fl = 0
        self.all_biometry = False
        self.measure_score = measure_score

        self.update_measure_annos()

    def update_measure_annos(self):
        self.measure_annos = [self]

    @classmethod
    def from_annotation(cls, line_anno):
        return FLMeasureInfo(line_anno.start_point(), line_anno.end_point())

    @classmethod
    def from_json(cls, json_info):
        fl = FLMeasureInfo()

        fl._from_json(json_info)

        fl.parse_ruler_info(json_info)
        return fl

    def to_json_object(self):
        info = super().to_json_object()
        info['type'] = 'fl'
        info['measure_fl'] = self.fl
        # info['ruler_unit'] = self.ruler_unit
        if self.ruler_info is not None:
            info['ruler_info'] = self.ruler_info

        return info

    def update_ga(self):
        self.fl = self.length() * self.ruler_unit
        if self.fl == 0:
            self.ga = 0
            self.all_biometry = False
        else:
            self.ga = FetalBiometry.ga_from_fl(self.fl)

            self.all_biometry = FetalBiometry.has_all_biometry()
            if self.all_biometry:
                FetalBiometry.estimate_ga_efw()

    # 拿到所有的测量标注
    def get_all_measure_annotation(self, measure_mode='hadlock'):
        all_measure_annotation = {'line': [self], 'ellipse': []}

        return all_measure_annotation

    def is_same_as(self, info, thresh_ratio=0.05):
        if self.fl == 0 or info.fl == 0:
            return False

        dist = self.distance_between(self.start_point(), info.start_point())
        dist *= self.ruler_unit
        if dist > self.fl * thresh_ratio:
            return False

        diff = abs(self.fl - info.fl)
        return diff < self.fl * thresh_ratio
