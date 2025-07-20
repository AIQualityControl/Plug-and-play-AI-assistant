from .LineAnnotation import LineAnnotation
from .EllipseAnnotation import EllipseAnnotation
from .BiDiameterAnnotation import BiDiameterAnnotation
from .measure_info import MeasureInfo
from capture_core.QcDetection.utility import math_util


class HeartMeasureInfo(MeasureInfo):
    def __init__(self, thorax_anno: BiDiameterAnnotation = None, heart_anno: BiDiameterAnnotation = None,
                 LV_anno: BiDiameterAnnotation = None, RV_anno: BiDiameterAnnotation = None,
                 LA_anno: BiDiameterAnnotation = None, RA_anno: BiDiameterAnnotation = None,
                 DA_r_anno=None, DA_LA_anno=None,
                 IVS_line_anno=None, LVW_line_anno=None, RVW_line_anno=None,
                 thorax_ellipse: EllipseAnnotation = None, heart_ellipse: EllipseAnnotation = None,
                 LL_anno=None, RL_anno=None, ruler_unit=1):
        """constructor"""
        super(HeartMeasureInfo, self).__init__()

        # # 胸腔轴线、心脏轴线、胸腔横径、心脏横径
        self.thorax_bidiameter_anno = thorax_anno
        self.heart_bidiameter_anno = heart_anno

        self.LA_bidiameter_anno = LA_anno
        self.LV_bidiameter_anno = LV_anno

        self.RA_bidiameter_anno = RA_anno
        self.RV_bidiameter_anno = RV_anno

        self.DA_r_anno = DA_r_anno
        self.DA_LA_anno = DA_LA_anno

        self.LL_anno = LL_anno
        self.RL_anno = RL_anno

        self.IVS_line_anno = IVS_line_anno
        self.LVW_line_anno = LVW_line_anno
        self.RVW_line_anno = RVW_line_anno

        self.thorax_ellipse = thorax_ellipse
        self.heart_ellipse = heart_ellipse

        # ca,ttd,tcd 心脏轴、胸腔横径、心脏横径
        self.ca = 0

        self.thorax_circumference = 0
        self.heart_circumference = 0

        self.thorax_area = 0
        self.heart_area = 0

        self.DA_r = 0
        self.DA_LA = 0

        self.IVS_line = 0
        self.LVW_line = 0
        self.RVW_line = 0

        self.LL_area = 0
        self.RL_area = 0

        # self.ruler_unit = ruler_unit

        self.update_measure_annos()

    @property
    def tcd(self):
        return self.heart_bidiameter_anno.minor_len if self.heart_bidiameter_anno else 0.0

    @property
    def ttd(self):
        return self.thorax_bidiameter_anno.minor_len if self.thorax_bidiameter_anno else 0.0

    def update_measure_annos(self):
        self.measure_annos = []
        if self.thorax_bidiameter_anno:
            self.measure_annos.append(self.thorax_bidiameter_anno)

        if self.heart_bidiameter_anno:
            self.measure_annos.append(self.heart_bidiameter_anno)

        if self.RA_bidiameter_anno:
            self.measure_annos.append(self.RA_bidiameter_anno)

        if self.RV_bidiameter_anno:
            self.measure_annos.append(self.RV_bidiameter_anno)

        if self.LA_bidiameter_anno:
            self.measure_annos.append(self.LA_bidiameter_anno)

        if self.LV_bidiameter_anno:
            self.measure_annos.append(self.LV_bidiameter_anno)

        if self.DA_r_anno:
            self.measure_annos.append(self.DA_r_anno)

        if self.DA_LA_anno:
            self.measure_annos.append(self.DA_LA_anno)

        # if self.LL_anno:
        #     self.measure_annos.append(self.LL_anno)
        #
        # if self.RL_anno:
        #     self.measure_annos.append(self.RL_anno)

        if self.IVS_line_anno:
            self.measure_annos.append(self.IVS_line_anno)

        if self.LVW_line_anno:
            self.measure_annos.append(self.LVW_line_anno)

        if self.RVW_line_anno:
            self.measure_annos.append(self.RVW_line_anno)

        if self.heart_ellipse:
            self.measure_annos.append(self.heart_ellipse)

        if self.thorax_ellipse:
            self.measure_annos.append(self.thorax_ellipse)

    @classmethod
    def from_json(cls, json_info):

        tho_anno = BiDiameterAnnotation.from_json(json_info['thorax_bidiameter']) if 'thorax_bidiameter' in json_info else None
        heart_anno = BiDiameterAnnotation.from_json(json_info['heart_bidiameter']) if 'heart_bidiameter' in json_info else None

        RA_anno = BiDiameterAnnotation.from_json(json_info['RA_bidiameter']) if 'RA_bidiameter' in json_info else None
        LA_anno = BiDiameterAnnotation.from_json(json_info['LA_bidiameter']) if 'LA_bidiameter' in json_info else None
        RV_anno = BiDiameterAnnotation.from_json(json_info['RV_bidiameter']) if 'RV_bidiameter' in json_info else None
        LV_anno = BiDiameterAnnotation.from_json(json_info['LV_bidiameter']) if 'LV_bidiameter' in json_info else None

        DA_r_anno = LineAnnotation.from_json(json_info['DA_r_anno']) if 'DA_r_anno' in json_info else None
        DA_LA_anno = LineAnnotation.from_json(json_info['DA_LA_anno']) if 'DA_LA_anno' in json_info else None

        IVS_line_anno = LineAnnotation.from_json(json_info['IVS_line_anno']) if 'IVS_line_anno' in json_info else None
        LVW_line_anno = LineAnnotation.from_json(json_info['LVW_line_anno']) if 'LVW_line_anno' in json_info else None
        RVW_line_anno = LineAnnotation.from_json(json_info['RVW_line_anno']) if 'RVW_line_anno' in json_info else None

        heart_ellipse_anno = EllipseAnnotation.from_json(json_info['heart_ellipse']) if 'heart_ellipse' in json_info else None
        tho_ellipse_anno = EllipseAnnotation.from_json(json_info['tho_ellipse']) if 'tho_ellipse' in json_info else None

        measure_info = HeartMeasureInfo(tho_anno, heart_anno, LV_anno, RV_anno, LA_anno, RA_anno,
                                        DA_r_anno, DA_LA_anno, IVS_line_anno, LVW_line_anno, RVW_line_anno,
                                        tho_ellipse_anno, heart_ellipse_anno)

        measure_info.parse_ruler_info(json_info)

        return measure_info

    def to_json_object(self):
        info = {
            'type': 'heart'
        }
        if self.thorax_bidiameter_anno:
            info['thorax_bidiameter'] = self.thorax_bidiameter_anno.to_json_object()
        if self.heart_bidiameter_anno:
            info['heart_bidiameter'] = self.heart_bidiameter_anno.to_json_object()

        if self.LA_bidiameter_anno:
            info['LA_bidiameter'] = self.LA_bidiameter_anno.to_json_object()
        if self.RA_bidiameter_anno is not None:
            info['RA_bidiameter'] = self.RA_bidiameter_anno.to_json_object()
        if self.LV_bidiameter_anno is not None:
            info['LV_bidiameter'] = self.LV_bidiameter_anno.to_json_object()
        if self.RV_bidiameter_anno is not None:
            info['RV_bidiameter'] = self.RV_bidiameter_anno.to_json_object()

        if self.DA_r_anno is not None:
            info['DA_r_anno'] = self.DA_r_anno.to_json_object()
        if self.DA_LA_anno is not None:
            info['DA_LA_anno'] = self.DA_LA_anno.to_json_object()

        if self.IVS_line_anno is not None:
            info['IVS_line_anno'] = self.IVS_line_anno.to_json_object()
        if self.LVW_line_anno is not None:
            info['LVW_line_anno'] = self.LVW_line_anno.to_json_object()
        if self.RVW_line_anno is not None:
            info['RVW_line_anno'] = self.RVW_line_anno.to_json_object()

        if self.heart_ellipse:
            info['heart_ellipse'] = self.heart_ellipse.to_json_object()

        if self.thorax_ellipse:
            info['tho_ellipse'] = self.thorax_ellipse.to_json_object()

        if self.ruler_info is not None:
            info['ruler_info'] = self.ruler_info
        return info

    def update_ga(self):
        if self.thorax_bidiameter_anno and self.heart_bidiameter_anno:
            self.ca = math_util.angle_between(self.thorax_bidiameter_anno.major_axis.dir(),
                                              self.heart_bidiameter_anno.major_axis.dir(),
                                              ignore_direction=True, in_degree=True)
            if self.ca >= 90:
                self.ca = 180 - self.ca
        else:
            self.ca = 0

        if self.thorax_bidiameter_anno:
            self.thorax_bidiameter_anno.update_actual_length(self.ruler_unit)

        if self.heart_bidiameter_anno:
            self.heart_bidiameter_anno.update_actual_length(self.ruler_unit)

        if self.LV_bidiameter_anno:
            self.LV_bidiameter_anno.update_actual_length(self.ruler_unit)

        if self.RV_bidiameter_anno is not None:
            self.RV_bidiameter_anno.update_actual_length(self.ruler_unit)

        if self.LA_bidiameter_anno:
            self.LA_bidiameter_anno.update_actual_length(self.ruler_unit)

        if self.RA_bidiameter_anno is not None:
            self.RA_bidiameter_anno.update_actual_length(self.ruler_unit)

        if self.IVS_line_anno is not None:
            self.IVS_line = self.IVS_line_anno.length() * self.ruler_unit
        else:
            self.IVS_line = 0

        if self.LVW_line_anno is not None:
            self.LVW_line = self.LVW_line_anno.length() * self.ruler_unit
        else:
            self.LVW_line = 0

        if self.RVW_line_anno is not None:
            self.RVW_line = self.RVW_line_anno.length() * self.ruler_unit
        else:
            self.RVW_line = 0

        if self.DA_LA_anno is not None:
            self.DA_LA = self.DA_LA_anno.length() * self.ruler_unit
        else:
            self.DA_LA = 0

        if self.DA_r_anno is not None:
            self.DA_r = self.DA_r_anno.length() * self.ruler_unit
        else:
            self.DA_r = 0

        if self.LL_anno is not None:
            self.LL_area = self.LL_anno * self.ruler_unit * self.ruler_unit
        else:
            self.LL_area = 0

        if self.RL_anno is not None:
            self.RL_area = self.RL_anno * self.ruler_unit * self.ruler_unit
        else:
            self.RL_area = 0

        if self.thorax_ellipse is not None:
            self.thorax_area = self.thorax_ellipse.area() * self.ruler_unit * self.ruler_unit
            self.thorax_circumference = self.thorax_ellipse.circumference() * self.ruler_unit  # 胸腔圆周
        else:
            self.thorax_area = 0
            self.thorax_circumference = 0

        if self.heart_ellipse is not None:
            self.heart_area = self.heart_ellipse.area() * self.ruler_unit * self.ruler_unit
            self.heart_circumference = self.heart_ellipse.circumference() * self.ruler_unit
        else:
            self.heart_area = 0
            self.heart_circumference = 0

    def change_end_point(self, idx, pos):
        pass

    # 拿到所有的测量标注
    def get_all_measure_annotation(self, measure_mode='hadlock'):
        lines = []
        all_measure_annotation = {'line': lines, 'ellipse': []}
        if self.thorax_bidiameter_anno:
            if self.thorax_bidiameter_anno.major_axis:
                lines.append(self.thorax_bidiameter_anno.major_axis)
            if self.thorax_bidiameter_anno.minor_axis:
                lines.append(self.thorax_bidiameter_anno.minor_axis)
        if self.heart_bidiameter_anno:
            if self.heart_bidiameter_anno.major_axis:
                lines.append(self.heart_bidiameter_anno.major_axis)
            if self.heart_bidiameter_anno.minor_axis:
                lines.append(self.heart_bidiameter_anno.minor_axis)

        if self.LA_bidiameter_anno:
            lines.append(self.LA_bidiameter_anno.major_axis)
            lines.append(self.LA_bidiameter_anno.minor_axis)
        if self.RA_bidiameter_anno:
            lines.append(self.RA_bidiameter_anno.major_axis)
            lines.append(self.RA_bidiameter_anno.minor_axis)

        if self.LV_bidiameter_anno:
            lines.append(self.LV_bidiameter_anno.major_axis)
            lines.append(self.LV_bidiameter_anno.minor_axis)
        if self.RV_bidiameter_anno:
            lines.append(self.RV_bidiameter_anno.major_axis)
            lines.append(self.RV_bidiameter_anno.minor_axis)

        if self.IVS_line_anno:
            all_measure_annotation['line'].append(self.IVS_line_anno)
        if self.LVW_line_anno:
            all_measure_annotation['line'].append(self.LVW_line_anno)
        if self.RVW_line_anno:
            all_measure_annotation['line'].append(self.RVW_line_anno)

        if self.DA_r_anno:
            all_measure_annotation['line'].append(self.DA_r_anno)
        if self.DA_LA_anno:
            all_measure_annotation['line'].append(self.DA_LA_anno)

        return all_measure_annotation
