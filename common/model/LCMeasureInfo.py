from .measure_info import MeasureInfo
from .BiDiameterAnnotation import BiDiameterAnnotation


class LCMeasureInfo(MeasureInfo):
    def __init__(self, lc_anno=None, lp_anno_list=None):
        """constructor"""
        MeasureInfo.__init__(self)

        # 卵巢长轴长
        self.lc_bidiameter = lc_anno

        # 卵泡
        self.lp_anno_list = lp_anno_list

        self.update_measure_annos()

    def update_measure_annos(self):
        self.measure_annos = []
        self.measure_annos.extend(self.lp_anno_list)

        if self.lc_bidiameter:
            self.measure_annos.append(self.lc_bidiameter)

    @classmethod
    def from_json(cls, json_info):

        lc_anno = BiDiameterAnnotation.from_json(json_info['lc']) if 'lc' in json_info else None

        lp_anno_list = []
        if 'lp_list' in json_info:
            for lp_info in json_info['lp_list']:
                lp_anno = BiDiameterAnnotation.from_json(lp_info)
                lp_anno_list.append(lp_anno)

        measure_info = LCMeasureInfo(lc_anno, lp_anno_list)

        measure_info.parse_ruler_info(json_info)

        return measure_info

    def to_json_object(self):
        info = {
            'type': 'lc'
        }
        if self.lc_bidiameter is not None:
            info['lc'] = self.lc_bidiameter.to_json_object()

        if self.lp_anno_list:
            lp_info_list = []
            for lp in self.lp_anno_list:
                lp_info_list.append(lp.to_json_object())
            info['lp_list'] = lp_info_list

        if self.ruler_info is not None:
            info['ruler_info'] = self.ruler_info

        return info

    def update_ga(self):
        if self.lc_bidiameter:
            self.lc_bidiameter.update_actual_length(self.ruler_unit)

        if self.lp_anno_list:
            for lp in self.lp_anno_list:
                lp.update_actual_length(self.ruler_unit)

    def change_end_point(self, idx, pos):
        pass

    # 拿到所有的测量标注
    def get_all_measure_annotation(self, measure_mode='hadlock'):
        all_measure_annotation = {'line': [], 'ellipse': []}

        return all_measure_annotation
