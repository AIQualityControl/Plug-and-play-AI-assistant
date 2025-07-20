from .measure_info import MeasureInfo
from .LineAnnotation import LineAnnotation
from common.FetalBiometry import FetalBiometry


class HLMeasureInfo(LineAnnotation, MeasureInfo):

    def __init__(self, ptStart=[0, 0], ptEnd=[0, 0], measure_score=0, is_default_value=False):
        # sourcery skip: default-mutable-arg
        """constructor"""
        LineAnnotation.__init__(self, ptStart, ptEnd, is_default_value=is_default_value)
        MeasureInfo.__init__(self)

        self.hl = 0
        self.all_biometry = False
        self.measure_score = measure_score
        self.update_measure_annos()

    def update_measure_annos(self):
        self.measure_annos = [self]

    @classmethod
    def from_annotation(cls, line_anno):
        return HLMeasureInfo(line_anno.start_point(), line_anno.end_point())

    @classmethod
    def from_json(cls, json_info):
        hl = HLMeasureInfo()
        hl._from_json(json_info)

        hl.parse_ruler_info(json_info)
        return hl

    def to_json_object(self):
        info = super().to_json_object()
        info['type'] = 'hl'
        info['measure_hl'] = self.hl
        # info['ruler_unit'] = self.ruler_unit
        if self.ruler_info is not None:
            info['ruler_info'] = self.ruler_info

        return info

    def update_ga(self):
        self.hl = self.length() * self.ruler_unit
        if self.hl == 0:
            self.ga = 0
            self.all_biometry = False
        else:
            self.ga = FetalBiometry.ga_from_hl(self.hl)

            # self.all_biometry = FetalBiometry.has_all_biometry()
            # if self.all_biometry:
            #     FetalBiometry.estimate_ga_efw()

    # 拿到所有的测量标注
    def get_all_measure_annotation(self, measure_mode='hadlock'):
        all_measure_annotation = {'line': [self], 'ellipse': []}

        return all_measure_annotation

    def is_same_as(self, info, thresh_ratio=0.05):
        if self.hl == 0 or info.hl == 0:
            return False

        dist = self.distance_between(self.start_point(), info.start_point())
        dist *= self.ruler_unit
        if dist > self.hl * thresh_ratio:
            return False

        diff = abs(self.hl - info.hl)
        return diff < self.hl * thresh_ratio
