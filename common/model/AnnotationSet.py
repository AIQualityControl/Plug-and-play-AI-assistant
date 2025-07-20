import json
import numpy as np

from .converter import annotation_from_json, measure_info_from_json
from .Annotation import Annotation
from .BoxAnnotation import BoxAnnotation
from .PolygonAnnotation import PolygonAnnotation


def convert_numpy_type(obj):
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return list(obj)
    raise Exception(f'{obj} with {type(obj)} is not serializable')


class AnnotationSet:
    STD_LIST = ['标准', '基本标准', '非标准']

    def __init__(self, plane_type=None, class_score=0, doctor_cpature=False):
        self.annotations = []
        self.measure_results = None

        self.std_type = '非标准'

        self.plane_type = plane_type

        # second classification result
        self.second_class_result = None

        self.is_thyroid = False
        self.is_dopler = False

        # score of classification, used for display
        self.class_score = class_score
        # score of detection
        self.score = 0
        self.video_score = 0

        # confidence of detection
        self.confidence = 0
        self.avg_confidence = 0

        self.doctor_capture = doctor_cpature

        # bbox for segmentation, used for display
        self.seg_roi = None

        # same as image info: 0 --- new, 1 --- update, 2 --- keep both, -1 --- not std image
        self.new_or_update_mode = -1

        # zoom in and still video
        self.zoom_in = False
        self.is_still = False
        self.has_sampling_line = False

        # video clip
        self.video_clip_path = None
        self.video_clip_range = None

        # thyroid_clip_range
        self.thyroid_clip_range = None

        self.optimal_replace = -1

        self.measure_score = 0

        # used to ingore saving annotation
        self.image_dicom = False

    def minus_offset_idx(self,  start_idx):
        if self.optimal_replace > 0:
            self.optimal_replace -= start_idx
        if self.video_clip_range:
            self.video_clip_range[0] -= start_idx
            self.video_clip_range[1] -= start_idx

    def get_annotations(self):
        return self.annotations

    def get_measure_results(self):
        return self.measure_results

    def add_annotation(self, anno):
        if isinstance(anno, (dict, str)):
            anno = Annotation.from_json(anno)
        self.annotations.append(anno)

    def add_annotations(self, annos):
        for anno in annos:
            self.add_annotation(anno)

    def clear(self):
        self.annotations.clear()

    def set_std_type(self, std_type):
        if isinstance(std_type, int):
            if 0 < std_type < 4:
                self.std_type = AnnotationSet.STD_LIST[std_type - 1]
        else:
            self.std_type = std_type

    def get_polygon_annotations(self, anno_name=None):
        poly_annotations = []
        for anno in self.annotations:
            if isinstance(anno, PolygonAnnotation) and (not anno_name or anno_name == anno.name):
                poly_annotations.append(anno)

        return poly_annotations

    @classmethod
    def from_json_str(cls, annotaions, offset=(0, 0)):
        if isinstance(annotaions, str):
            annotaions = json.loads(annotaions)

        # parse to
        anno_set = AnnotationSet()
        for anno in annotaions:
            if isinstance(anno, str):
                anno = json.loads(anno)
            anno = BoxAnnotation.from_json(anno, offset)
            anno_set.annotations.append(anno)

        return anno_set

    @classmethod
    def from_json(cls, json_info):
        plane_type = ''
        if 'plane_type' in json_info:
            plane_type = json_info['plane_type']
        elif 'image_type' in json_info:
            plane_type = json_info['image_type']
        elif 'bodyPart' in json_info:
            plane_type = json_info['bodyPart']

        class_score = json_info['class_score'] if 'class_score' in json_info else 100

        anno_set = AnnotationSet(plane_type, class_score)

        if 'second_result' in json_info:
            if isinstance(json_info['second_result'], str):
                anno_set.second_class_result = json.loads(json_info['second_result'])
            else:
                anno_set.second_class_result = json_info['second_result']

        if 'score' in json_info:
            anno_set.score = json_info['score']
        if 'confidence' in json_info:
            anno_set.confidence = json_info['confidence']
        if 'avg_confidence' in json_info:
            anno_set.avg_confidence = json_info['avg_confidence']
        if 'std_type' in json_info:
            anno_set.std_type = json_info['std_type']
        elif 'standard' in json_info:
            anno_set.std_type = json_info['standard']

        if 'doctor_capture' in json_info:
            anno_set.doctor_capture = json_info['doctor_capture']
        if 'is_thyroid' in json_info:
            anno_set.is_thyroid = json_info['is_thyroid']
        if 'is_dopler' in json_info:
            anno_set.is_dopler = json_info['is_dopler']

        if 'new_or_update' in json_info:
            if isinstance(json_info['new_or_update'], bool):
                anno_set.new_or_update_mode = 0 if json_info['new_or_update'] else -1
            else:
                anno_set.new_or_update_mode = json_info['new_or_update']
        if 'new_or_update_mode' in json_info:
            anno_set.new_or_update_mode = json_info['new_or_update_mode']

        if 'video_score' in json_info:
            anno_set.video_score = json_info['video_score']
        else:
            anno_set.video_score = anno_set.score

        if 'is_still' in json_info:
            anno_set.is_still = json_info['is_still']
        if 'zoom_in' in json_info:
            anno_set.zoom_in = json_info['zoom_in']
        if 'sampling_line' in json_info:
            anno_set.has_sampling_line = json_info['sampling_line']

        if 'image_dicom' in json_info:
            anno_set.image_dicom = json_info['image_dicom']
        elif 'optimal_replace' in json_info:
            anno_set.optimal_replace = json_info['optimal_replace']
        elif 'measure_score' in json_info:
            anno_set.measure_score = json_info['measure_score']

        if 'annotations' in json_info:
            for anno_info in json_info['annotations']:
                anno = annotation_from_json(anno_info)
                if anno is not None:
                    anno_set.annotations.append(anno)

        if 'measure_results' in json_info:
            measure_info = json_info['measure_results']
            anno_set.measure_results = measure_info_from_json(measure_info)

        if 'seg_roi' in json_info:
            anno_set.seg_roi = json_info['seg_roi']

        if 'video_clip_range' in json_info:
            anno_set.video_clip_range = json_info['video_clip_range']

        if 'thyroid_clip_range' in json_info:
            anno_set.thyroid_clip_range = json_info['thyroid_clip_range']

        return anno_set

    def to_json_object(self):
        anno_json = {
            'plane_type': self.plane_type,
            'class_score': round(float(self.class_score), 2),
            'std_type': self.std_type,
        }

        if self.second_class_result:
            self.second_class_result['class_score'] = float(self.second_class_result['class_score'])
            anno_json['second_result'] = self.second_class_result

        if self.is_still:
            anno_json['is_still'] = True
        if self.zoom_in:
            anno_json['zoom_in'] = True
        if self.has_sampling_line:
            anno_json['sampling_line'] = True

        if self.image_dicom:
            anno_json['image_dicom'] = True

        if self.doctor_capture:
            anno_json['doctor_capture'] = True
        if self.is_thyroid:
            anno_json['is_thyroid'] = True

        if self.video_clip_range:
            anno_json['video_clip_range'] = self.video_clip_range

        if self.thyroid_clip_range:
            anno_json['thyroid_clip_range'] = self.thyroid_clip_range

        if self.optimal_replace >= 0:
            anno_json['optimal_replace'] = self.optimal_replace

        if self.measure_score > 0:
            anno_json['measure_score'] = self.measure_score

        if self.seg_roi is not None:
            anno_json['seg_roi'] = self.seg_roi

        if self.annotations:
            anno_json['new_or_update_mode'] = self.new_or_update_mode

            anno_json['score'] = round(float(self.score), 2)
            anno_json['video_score'] = round(float(self.video_score), 2)
            anno_json['confidence'] = round(float(self.confidence), 2)
            anno_json['avg_confidence'] = round(float(self.avg_confidence), 2)

            # annotations
            annotations = [anno.to_json_object() for anno in self.annotations]
            anno_json['annotations'] = annotations

        # measure result
        if self.measure_results:
            anno_json['measure_results'] = self.measure_results.to_json_object()

        if self.is_dopler:
            anno_json['is_dopler'] = self.is_dopler

        return anno_json

    def to_json_str(self):
        anno_json = self.to_json_object()
        return json.dumps(anno_json, ensure_ascii=False, default=convert_numpy_type)

    def translate(self, offset):
        for anno in self.annotations:
            anno.translate(offset)

    def is_FL(self):
        return self.plane_type == '股骨长轴切面'

    def is_HC(self):
        return self.plane_type == '丘脑水平横切面'

    def is_AC(self):
        return self.plane_type == '上腹部水平横切面'

    def is_std(self):
        return self.std_type == '标准'

    def is_astd(self):
        return self.std_type == '基本标准'

    def is_nstd(self):
        return self.std_type == '非标准'

    def is_astd_std(self):
        return self.std_type != '非标准'
