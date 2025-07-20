from .LineAnnotation import LineAnnotation
from .measure_info import MeasureInfo
from .PolylineAnnotation import PolylineAnnotation


class ZGGJMeasureInfo(MeasureInfo):
    def __init__(self, gjmaj_anno=None, gjmin_anno=None, gjx_anno=None):
        """constructor"""
        super(ZGGJMeasureInfo, self).__init__()

        # 宫颈长轴长
        self.gjmaj_anno = gjmaj_anno
        self.gjmaj = 0
        # 宫颈前后径长
        self.gjmin_anno = gjmin_anno
        self.gjmin = 0

        # 宫颈线长度
        self.gjx_anno = gjx_anno
        self.gjx_len = 0

        self.update_measure_annos()

    def update_measure_annos(self):
        self.measure_annos = []
        if self.gjmaj_anno:
            self.measure_annos.append(self.gjmaj_anno)
        if self.gjmin_anno:
            self.measure_annos.append(self.gjmin_anno)
        if self.gjx_anno:
            self.measure_annos.append(self.gjx_anno)

    @classmethod
    def from_json(cls, json_info):

        gjmaj_anno = LineAnnotation.from_json(json_info['gjmaj']) if 'gjmaj' in json_info else None
        gjmin_anno = LineAnnotation.from_json(json_info['gjmin']) if 'gjmin' in json_info else None
        gjx_anno = PolylineAnnotation.from_json(json_info['gjx']) if 'gjx' in json_info else None

        measure_info = ZGGJMeasureInfo(gjmaj_anno, gjmin_anno, gjx_anno)

        measure_info.parse_ruler_info(json_info)

        return measure_info

    def to_json_object(self):
        info = {
            'type': 'zggj'
        }
        if self.gjmaj_anno is not None:
            info['gjmaj'] = self.gjmaj_anno.to_json_object()
            info['measure_gjmaj'] = self.gjmaj
        if self.gjmin_anno is not None:
            info['gjmin'] = self.gjmin_anno.to_json_object()
            info['measure_gjmin'] = self.gjmin
        if self.gjx_anno is not None:
            info['gjx'] = self.gjx_anno.to_json_object()
            info['gjx_len'] = self.gjx_len
        if self.ruler_info is not None:
            info['ruler_info'] = self.ruler_info

        return info

    def update_ga(self):
        if self.gjmaj_anno is not None:
            self.gjmaj = self.gjmaj_anno.length() * self.ruler_unit
        else:
            self.gjmaj = 0

        if self.gjmin_anno is not None:
            self.gjmin = self.gjmin_anno.length() * self.ruler_unit
        else:
            self.gjmin = 0

        if self.gjx_anno is not None:
            self.gjx_len = self.gjx_anno.length() * self.ruler_unit
        else:
            self.gjx_len = 0

    def change_end_point(self, idx, pos):
        pass

    # 拿到所有的测量标注
    def get_all_measure_annotation(self, measure_mode='hadlock'):
        all_measure_annotation = {'line': [], 'ellipse': [], 'polyline': []}
        if self.gjmaj_anno:
            all_measure_annotation['line'].append(self.gjmaj_anno)
        if self.gjmin_anno:
            all_measure_annotation['line'].append(self.gjmin_anno)
        if self.gjx_anno:
            all_measure_annotation['polyline'].append(self.gjx_anno)

        return all_measure_annotation
