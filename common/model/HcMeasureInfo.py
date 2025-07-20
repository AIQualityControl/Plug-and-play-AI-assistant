from common.FetalBiometry import FetalBiometry
from .EllipseAnnotation import EllipseAnnotation
from .LineAnnotation import LineAnnotation
from .measure_info import MeasureInfo


class HcMeasureInfo(MeasureInfo):
    def __init__(self, hc_annoatation=None, intergrowth_21st_bpd_annotation=None, hadlock_bpd_annotation=None):
        '''constructor'''
        super(HcMeasureInfo, self).__init__()

        self.hc_annotation = hc_annoatation
        self.intergrowth_21st_bpd_annotation = intergrowth_21st_bpd_annotation
        self.hadlock_bpd_annotation = hadlock_bpd_annotation

        self.hc = 0
        self.bpd = 0

        self.hc_ga = 0
        self.bpd_ga = 0

        self.update_measure_annos()

        self.all_biometry = False

    def update_measure_annos(self):
        self.measure_annos = []
        if self.hc_annotation:
            self.measure_annos.append(self.hc_annotation)
        if self.intergrowth_21st_bpd_annotation:
            self.measure_annos.append(self.intergrowth_21st_bpd_annotation)
        if self.hadlock_bpd_annotation:
            self.measure_annos.append(self.hadlock_bpd_annotation)

    @classmethod
    def from_json(cls, json_info):
        if 'hc' in json_info:
            hc_annotation = EllipseAnnotation.from_json(json_info['hc'])
        else:
            hc_annotation = None

        # judge for intergrowth-21st
        hc_minor_radius_ptStart = None
        hc_minor_radius_ptEnd = None

        if hc_annotation:
            hc_minor_radius_points = hc_annotation.minor_radius_points()
            hc_minor_radius_ptStart = hc_minor_radius_points[0]
            hc_minor_radius_ptEnd = hc_minor_radius_points[1]

        intergrowth_21st_bpd_annotation = None
        hadlock_bpd_annotation = None

        if "bpd" in json_info:
            old_bpd_annotation = LineAnnotation.from_json(json_info["bpd"])
            # judge if intergrowth-21st or hadlock through hc minor radius points
            old_bpd_ptStart = old_bpd_annotation.ptStart
            old_bpd_ptEnd = old_bpd_annotation.ptEnd
            if hc_minor_radius_ptStart and hc_minor_radius_ptEnd:
                if abs(hc_minor_radius_ptStart[0] - old_bpd_ptStart[0]) < 3 and \
                        abs(hc_minor_radius_ptStart[1] - old_bpd_ptStart[1]) < 3 and \
                        abs(hc_minor_radius_ptEnd[0] - old_bpd_ptEnd[0]) < 3 and \
                        abs(hc_minor_radius_ptEnd[1] - old_bpd_ptEnd[1]) < 3:
                    # old bpd is intergrowth-21st
                    intergrowth_21st_bpd_annotation = LineAnnotation(hc_minor_radius_ptStart, hc_minor_radius_ptEnd)
                else:
                    # is hadlock
                    intergrowth_21st_bpd_annotation = LineAnnotation(hc_minor_radius_ptStart, hc_minor_radius_ptEnd)
                    hadlock_bpd_annotation = LineAnnotation(old_bpd_ptStart, old_bpd_ptEnd)

        if 'intergrowth_21st_bpd' in json_info:
            intergrowth_21st_bpd_annotation = LineAnnotation.from_json(json_info['intergrowth_21st_bpd'])

        if 'hadlock_bpd' in json_info:
            hadlock_bpd_annotation = LineAnnotation.from_json(json_info['hadlock_bpd'])

        measure_info = HcMeasureInfo(hc_annotation, intergrowth_21st_bpd_annotation, hadlock_bpd_annotation)

        measure_info.parse_ruler_info(json_info)

        return measure_info

    def to_json_object(self):
        info = {
            'type': 'hc'
        }
        if self.hc_annotation is not None:
            hc_info = self.hc_annotation.to_json_object()
            info['hc'] = hc_info
            info['measure_hc'] = self.hc
        if self.intergrowth_21st_bpd_annotation is not None:
            info['intergrowth_21st_bpd'] = self.intergrowth_21st_bpd_annotation.to_json_object()
        if self.hadlock_bpd_annotation is not None:
            info['hadlock_bpd'] = self.hadlock_bpd_annotation.to_json_object()

        info['measure_bpd'] = self.bpd
        if self.ruler_info is not None:
            info['ruler_info'] = self.ruler_info

        return info

    def update_ga(self, measure_mode=None):
        if self.hc_annotation is not None:
            self.hc = self.hc_annotation.circumference() * self.ruler_unit
            self.hc_ga = FetalBiometry.ga_from_hc(self.hc)
        else:
            self.hc = self.hc_ga = 0

        self.bpd = 0
        if measure_mode == "hadlock":
            if self.hadlock_bpd_annotation is not None:
                self.bpd = self.hadlock_bpd_annotation.length() * self.ruler_unit
        else:

            if self.intergrowth_21st_bpd_annotation is not None:
                self.bpd = self.intergrowth_21st_bpd_annotation.length() * self.ruler_unit

        if self.bpd == 0:
            self.bpd_ga = 0
        else:
            self.bpd_ga = FetalBiometry.ga_from_bpd(self.bpd)

        # self.ga = FetalBiometry.ga_from_hc_bpd(self.hc, self.bpd)
        FetalBiometry.is_hc_plane_detected = True

        self.all_biometry = FetalBiometry.has_all_biometry()
        if self.all_biometry:
            FetalBiometry.estimate_ga_efw()

    def get_ofd(self):
        if self.hc_annotation is None:
            return 0

        ofd = self.hc_annotation.major_axis_length()
        ofd *= self.ruler_unit
        return ofd

    def get_bpd(self, measure_mode='hadlock'):
        self.update_ga(measure_mode)
        return self.bpd

    def get_minor_axis_length(self):
        if self.hc_annotation is None:
            return 0

        length = self.hc_annotation.minor_axis_length()
        return length * self.ruler_unit

    def change_end_point(self, idx, pos):
        pass

    # 拿到所有的测量标注
    def get_all_measure_annotation(self, measure_mode='hadlock'):
        all_measure_annotation = {'line': [], 'ellipse': []}
        if measure_mode == 'hadlock':
            if self.hadlock_bpd_annotation:
                all_measure_annotation['line'].append(self.hadlock_bpd_annotation)
        if self.hc_annotation:
            all_measure_annotation['ellipse'].append(self.hc_annotation)

        return all_measure_annotation

    def is_same_as(self, info, thresh_ratio=0.05):
        if self.hc == 0 or info.hc == 0:
            return False

        dist = self.distance_between(self.hc_annotation.center_point(), info.hc_annotation.center_point())
        dist *= self.ruler_unit
        if dist > self.hc * thresh_ratio:
            return False

        diff = abs(self.hc - info.hc)
        if diff > self.hc * thresh_ratio:
            return False

        diff = abs(self.bpd - info.bpd)
        return diff <= self.bpd * thresh_ratio
