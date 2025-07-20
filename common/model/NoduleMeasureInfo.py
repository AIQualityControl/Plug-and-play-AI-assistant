from .measure_info import MeasureInfo
from .BiDiameterAnnotation import BiDiameterAnnotation


class NoduleMeasureInfo(MeasureInfo):
    def __init__(self, nodule_list):
        """constructor"""
        MeasureInfo.__init__(self)

        self.nodule_anno_list = []
        if not nodule_list:
            return

        for nodule in nodule_list:
            if isinstance(nodule, BiDiameterAnnotation):
                self.nodule_anno_list.append(nodule)
            else:
                if isinstance(nodule[0], str):
                    anno = BiDiameterAnnotation(list(nodule[1][0]), list(nodule[1][1]))
                    anno.name = nodule[0]
                else:
                    anno = BiDiameterAnnotation(list(nodule[0]), list(nodule[1]))
                self.nodule_anno_list.append(anno)

        self.update_measure_annos()

    def update_measure_annos(self):
        self.measure_annos = self.nodule_anno_list

    @classmethod
    def from_json(cls, json_info):

        nodule_list = []
        if 'nodule' in json_info:
            for nodule_info in json_info['nodule']:
                nodule_anno = BiDiameterAnnotation.from_json(nodule_info)
                nodule_list.append(nodule_anno)

        measure_info = NoduleMeasureInfo(nodule_list)

        measure_info.parse_ruler_info(json_info)

        return measure_info

    def to_json_object(self):
        info = {
            'type': 'nodule',
            'nodule': [nodule_anno.to_json_object() for nodule_anno in self.nodule_anno_list]
        }

        if self.ruler_info is not None:
            info['ruler_info'] = self.ruler_info

        return info

    def update_ga(self):
        for nodule_anno in self.nodule_anno_list:
            nodule_anno.update_actual_length(self.ruler_unit)

    def change_end_point(self, idx, pos):
        pass

    # 拿到所有的测量标注
    def get_all_measure_annotation(self, measure_mode='hadlock'):

        all_measure_annotation = {'line': [], 'ellipse': []}

        for each_nodule_anno in self.nodule_anno_list:
            all_measure_annotation['line'].append(each_nodule_anno.major_axis)
            all_measure_annotation['line'].append(each_nodule_anno.minor_axis)

        return all_measure_annotation
