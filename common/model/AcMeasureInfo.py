from .measure_info import MeasureInfo
from .EllipseAnnotation import EllipseAnnotation
from common.FetalBiometry import FetalBiometry


class AcMeasureInfo(EllipseAnnotation, MeasureInfo):
    def __init__(self, pt_start=[0, 0], pt_end=[0, 0], degree=0, measure_score=100, is_default_value=False):
        # sourcery skip: default-mutable-arg
        """constructor"""
        EllipseAnnotation.__init__(self, pt_start, pt_end, degree, is_default_value=is_default_value)
        MeasureInfo.__init__(self)
        self.ac = 0
        self.all_biometry = False

        self.measure_score = measure_score

        # self.update_ga()

    @classmethod
    def from_annotation(cls, ellipse_anno):
        return AcMeasureInfo(ellipse_anno.start_point(), ellipse_anno.end_point(), ellipse_anno.degree())

    @classmethod
    def from_json(cls, json_info):
        degree = EllipseAnnotation.RAD_TO_DEGREE(json_info['angle'])
        ac = AcMeasureInfo(degree=degree)

        ac._from_json(json_info)
        ac.parse_ruler_info(json_info)

        return ac

    def to_json_object(self):
        info = super().to_json_object()
        info['type'] = 'ac'
        info['measure_ac'] = self.circumference() * self.ruler_unit

        if self.ruler_info is not None:
            info['ruler_info'] = self.ruler_info

        return info

    def update_ga(self):
        self.ac = self.circumference() * self.ruler_unit
        if self.ac == 0:
            self.ga = 0
            self.all_biometry = False
        else:
            self.ga = FetalBiometry.ga_from_ac(self.ac)

            self.all_biometry = FetalBiometry.has_all_biometry()
            if self.all_biometry:
                FetalBiometry.estimate_ga_efw()

    # 拿到所有的测量标注
    def get_all_measure_annotation(self, measure_mode='hadlock'):
        all_measure_annotation = {'line': [], 'ellipse': [self]}

        return all_measure_annotation

    def is_same_as(self, info, thresh_ratio=0.05):
        if self.ac == 0 or info.ac == 0:
            return False

        dist = self.distance_between(self.center_point(), info.center_point())
        dist *= self.ruler_unit
        if dist > self.ac * thresh_ratio:
            return False

        diff = abs(self.ac - info.ac)
        return diff < self.ac * thresh_ratio
