from .LineAnnotation import LineAnnotation
from .BiDiameterAnnotation import BiDiameterAnnotation
from .measure_info import MeasureInfo


class ZGZMeasureInfo(MeasureInfo):
    def __init__(self, zg_bidiameter=None, nm_anno=None, jl_list=None):
        """constructor"""
        MeasureInfo.__init__(self)

        # 子宫双径
        self.zg_bidiameter = zg_bidiameter

        # 内膜厚度
        self.nm_anno = nm_anno
        self.nm = 0

        # 存储肌瘤数据
        self.jl_anno_list = []
        if jl_list:
            for jl_nodule in jl_list:
                if isinstance(jl_nodule, BiDiameterAnnotation):
                    self.jl_anno_list.append(jl_nodule)
                else:
                    if isinstance(jl_nodule[0], str):
                        anno = BiDiameterAnnotation(list(jl_nodule[1][0]), list(jl_nodule[1][1]))
                        anno.name = jl_nodule[0]
                    else:
                        anno = BiDiameterAnnotation(list(jl_nodule[0]), list(jl_nodule[1]))
                    self.jl_anno_list.append(anno)

        self.update_measure_annos()

    def update_measure_annos(self):
        self.measure_annos = []
        if self.zg_bidiameter:
            self.measure_annos.append(self.zg_bidiameter)

        if self.nm_anno:
            self.measure_annos.append(self.nm_anno)

        if self.jl_anno_list:
            self.measure_annos.extend(self.jl_anno_list)

    @classmethod
    def from_json(cls, json_info):

        zg_bidiameter = BiDiameterAnnotation.from_json(json_info['zg']) if 'zg' in json_info else None
        nm_anno = LineAnnotation.from_json(json_info['nm']) if 'nm' in json_info else None

        jl_anno_list = []
        if 'jl_nodule' in json_info:
            for nodule_info in json_info['jl_nodule']:
                nodule_anno = BiDiameterAnnotation.from_json(nodule_info)
                jl_anno_list.append(nodule_anno)

        measure_info = ZGZMeasureInfo(zg_bidiameter, nm_anno, jl_anno_list)

        measure_info.parse_ruler_info(json_info)

        return measure_info

    def to_json_object(self):
        info = {
            'type': 'zgz'
        }

        if self.zg_bidiameter:
            info['zg'] = self.zg_bidiameter.to_json_object()

        if self.nm_anno:
            info['nm'] = self.nm_anno.to_json_object()

        if self.jl_anno_list:
            info['jl_nodule'] = [jl_anno.to_json_object() for jl_anno in self.jl_anno_list]

        if self.ruler_info is not None:
            info['ruler_info'] = self.ruler_info

        return info

    def update_ga(self):
        if self.zg_bidiameter:
            self.zg_bidiameter.update_actual_length(self.ruler_unit)

        if self.nm_anno:
            self.nm = self.nm_anno.length() * self.ruler_unit
        else:
            self.nm = 0

        if self.jl_anno_list:
            for jl_anno in self.jl_anno_list:
                jl_anno.update_actual_length(self.ruler_unit)

    def change_end_point(self, idx, pos):
        pass

    # 拿到所有的测量标注
    def get_all_measure_annotation(self, measure_mode='hadlock'):

        all_measure_annotation = {'line': [], 'ellipse': []}

        if self.zg_bidiameter:
            all_measure_annotation['line'].append(self.zg_bidiameter.major_axis)
            all_measure_annotation['line'].append(self.zg_bidiameter.minor_axis)

        if self.nm_anno:
            all_measure_annotation['line'].append(self.nm_anno)

        if self.jl_anno_list:
            for jl_anno in self.jl_anno_list:
                all_measure_annotation['line'].append(jl_anno.major_axis)
                all_measure_annotation['line'].append(jl_anno.minor_axis)

        return all_measure_annotation
