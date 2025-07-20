from .LineAnnotation import LineAnnotation
from .measure_info import MeasureInfo
from .BiDiameterAnnotation import BiDiameterAnnotation


class ZGHMeasureInfo(MeasureInfo):
    def __init__(self, zghor_anno=None, jl_list=None):
        """constructor"""
        MeasureInfo.__init__(self)

        # 子宫长轴长c
        self.zghor_anno = zghor_anno
        self.zghor = 0

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
        if self.zghor_anno:
            self.measure_annos = [self.zghor_anno]
        else:
            self.measure_annos = []

        if self.jl_anno_list:
            self.measure_annos.extend(self.jl_anno_list)

    @classmethod
    def from_json(cls, json_info):

        if 'zghor' in json_info:
            zghor_anno = LineAnnotation.from_json(json_info['zghor'])
        else:
            zghor_anno = None

        jl_anno_list = []
        if 'jl_nodule' in json_info:
            for nodule_info in json_info['jl_nodule']:
                nodule_anno = BiDiameterAnnotation.from_json(nodule_info)
                jl_anno_list.append(nodule_anno)

        measure_info = ZGHMeasureInfo(zghor_anno, jl_anno_list)

        measure_info.parse_ruler_info(json_info)

        return measure_info

    def to_json_object(self):
        info = {
            'type': 'zgh'
        }
        if self.zghor_anno is not None:
            zghor_info = self.zghor_anno.to_json_object()
            info['zghor'] = zghor_info
            info['measure_zghor'] = self.zghor
        if self.ruler_info is not None:
            info['ruler_info'] = self.ruler_info

        if self.jl_anno_list:
            info['jl_nodule'] = [jl_anno.to_json_object() for jl_anno in self.jl_anno_list]

        return info

    def update_ga(self):
        if self.zghor_anno is not None:
            self.zghor = self.zghor_anno.length() * self.ruler_unit
        else:
            self.zghor = 0

        if self.jl_anno_list:
            for jl_anno in self.jl_anno_list:
                jl_anno.update_actual_length(self.ruler_unit)

    def change_end_point(self, idx, pos):
        pass

    # 拿到所有的测量标注
    def get_all_measure_annotation(self, measure_mode='hadlock'):
        all_measure_annotation = {'line': [], 'ellipse': []}
        if self.zghor_anno:
            all_measure_annotation['line'].append(self.zghor_anno)

        if self.jl_anno_list:
            for jl_anno in self.jl_anno_list:
                all_measure_annotation['line'].append(jl_anno.major_axis)
                all_measure_annotation['line'].append(jl_anno.minor_axis)

        return all_measure_annotation
