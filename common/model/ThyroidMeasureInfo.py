from .NoduleMeasureInfo import NoduleMeasureInfo
from .BiDiameterAnnotation import BiDiameterAnnotation


class ThyroidMeasureInfo(NoduleMeasureInfo):
    def __init__(self, nodule_anno_list=[], nodule_list=[], thyroid_list=[]):
        """
        can be any
        """
        anno_list = []
        if nodule_anno_list:
            anno_list.extend(nodule_anno_list)

        if nodule_list:
            for nodule, track_id, bbox, left_info in nodule_list:
                anno = BiDiameterAnnotation(list(nodule[0]), list(nodule[1]))
                anno.ptStart = [int(bbox[0] + 0.5), int(bbox[1] + 0.5)]
                anno.ptEnd = [int(bbox[2] + 0.5), int(bbox[3] + 0.5)]
                anno.name = "nodule"
                anno.track_id = int(track_id)
                anno_list.append(anno)
                anno.leftinfo = left_info

        if thyroid_list:
            for thyroid, track_id, bbox, left_info in thyroid_list:
                if track_id ==-3:
                    anno = BiDiameterAnnotation(list(thyroid[0]), list(thyroid[1]))
                    anno.ptStart = [int(bbox[0] + 0.5), int(bbox[1] + 0.5)]
                    anno.ptEnd = [int(bbox[2] + 0.5), int(bbox[3] + 0.5)]
                    anno.name = "isthum"
                    anno.track_id = int(track_id)
                    anno_list.append(anno)
                    anno.leftinfo = left_info
                else:
                    anno = BiDiameterAnnotation(list(thyroid[0]), list(thyroid[1]))
                    anno.ptStart = [int(bbox[0] + 0.5), int(bbox[1] + 0.5)]
                    anno.ptEnd = [int(bbox[2] + 0.5), int(bbox[3] + 0.5)]
                    anno.name = "thyroid"
                    anno.track_id = int(track_id)
                    anno_list.append(anno)
                    anno.leftinfo = left_info


        NoduleMeasureInfo.__init__(self, anno_list)

    def to_json_object(self):
        json = super().to_json_object()
        json["type"] = "thyroid"
        return json

    @classmethod
    def from_json(cls, json_info):

        nodule_list = []
        if "nodule" in json_info:
            for nodule_info in json_info["nodule"]:
                nodule_anno = BiDiameterAnnotation.from_json(nodule_info)
                nodule_list.append(nodule_anno)

        measure_info = ThyroidMeasureInfo(nodule_anno_list=nodule_list)

        measure_info.parse_ruler_info(json_info)

        return measure_info
