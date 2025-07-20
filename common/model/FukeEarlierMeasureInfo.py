from .LineAnnotation import LineAnnotation
from .measure_info import MeasureInfo


class FukeEarlierMeasureInfo(MeasureInfo):
    def __init__(self, earlier_maj_anno=None, earlier_min_anno=None, plane_type=None, seg_score=1):
        """constructor"""
        MeasureInfo.__init__(self)

        # 长轴长
        self.earlier_maj_anno = earlier_maj_anno
        self.earlier_maj = 0
        # 短轴长
        self.earlier_min_anno = earlier_min_anno
        self.earlier_min = 0

        self.plane_type = plane_type

        self.seg_score = seg_score

        self.update_measure_annos()

    def update_measure_annos(self):
        self.measure_annos = []
        if self.plane_type == '妊娠囊':
            if self.earlier_maj_anno:
                self.measure_annos.append(self.earlier_maj_anno)
            if self.earlier_min_anno:
                self.measure_annos.append(self.earlier_min_anno)
        elif self.plane_type == '卵黄囊' or self.plane_type == '胚芽':
            if self.earlier_maj_anno:
                self.measure_annos.append(self.earlier_maj_anno)

    @classmethod
    def from_json(cls, json_info):

        earlier_maj_anno = LineAnnotation.from_json(json_info['earlier_maj']) if 'earlier_maj' in json_info else None
        earlier_min_anno = LineAnnotation.from_json(json_info['earlier_min']) if 'earlier_min' in json_info else None
        plane_type = LineAnnotation.from_json(json_info['plane_type']) if 'plane_type' in json_info else None

        measure_info = FukeEarlierMeasureInfo(earlier_maj_anno, earlier_min_anno, plane_type)

        measure_info.parse_ruler_info(json_info)

        return measure_info

    def to_json_object(self):
        info = super().to_json_object()
        info['type'] = 'earlier'
        if self.earlier_maj_anno:
            earlier_maj_info = self.earlier_maj_anno.to_json_object()
            info['earlier_maj'] = earlier_maj_info
            info['measure_earlier_maj'] = self.earlier_maj
        if self.earlier_min_anno:
            earlier_min_info = self.earlier_min_anno.to_json_object()
            info['earlier_min'] = earlier_min_info
            info['measure_earlier_min'] = self.earlier_min

        return info

    def update_ga(self):
        if self.earlier_maj_anno is not None:
            self.earlier_maj = self.earlier_maj_anno.length() * self.ruler_unit
            # self.earlier_maj = self.earlier_maj_anno.length() * 1.0
        else:
            self.earlier_maj = 0

        if self.earlier_min_anno is not None:
            self.earlier_min = self.earlier_min_anno.length() * self.ruler_unit
            # self.earlier_min = self.earlier_min_anno.length() * 1.0
        else:
            self.earlier_min = 0

    def get_all_measure_annotation(self, measure_mode='hadlock'):
        all_measure_annotation = {'line': [], 'ellipse': []}
        if self.earlier_maj_anno:
            all_measure_annotation['line'].append(self.earlier_maj_anno)
        if self.earlier_min_anno:
            all_measure_annotation['line'].append(self.earlier_min_anno)

        return all_measure_annotation
