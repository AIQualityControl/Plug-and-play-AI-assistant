from .AnnotationSet import AnnotationSet
from .SpectrumMeasureInfo import SpectrumMeasureInfo
from ..config.config import AMNIOTIC_MEASURE_MODE, DICOM_TRANSFER_MODE, AUTO_CAPTURE_MODE


class ImageInfo:
    # xywh
    roi = [0, 0, 0, 0]  # xywh

    def __init__(self, image, frame_idx=0, queue_idx=0):
        """constructor"""

        self.image = image

        # frame idx in video
        self.frame_idx = frame_idx

        # used in shared memeory queue to indicate image size for each frame
        self.image_shape = None

        # idx in image queue
        self.queue_idx = queue_idx

        # number of frames skipped since video saving is slower than capturing
        self.skipped_frames = 0

        self.dicom_avi_time = 0
        self.dicom_avi = False
        self.dicom_avi_path = None
        self.image_dicom_jpg = False
        self.is_need_handle_replace = False
        self.is_manual_capture_image = False
        self.replace_frame_num_in_video = 0  # 帧号从1开始计数

        self.use_model = None
        # 判断是否保存传来测量值的字段
        self.is_save_measure_for_ui = True

        # whether is dopler image
        self.is_dopler = False
        self.is_still = False  # whether is still frame
        self.zoom_in = False  # whether zoom in
        self.has_sampling_line = False

        # start frame idx for dynamic two foot video
        self.start_frame_idx = -1

        # whether is a better image and need to update
        # 3~5 is measure frame, need to be updated
        # new_or_update_mode: 0 -- new std plane, 1 and 2 -- update std plane
        #                     1 --- replace the previous std plane, 2 --- keep both previous and current std plane
        #                     3 --- new measure plane, 4 --- replace the previous measure plane
        #                     5 --- keep both previous and current measure plane
        # new_or_update and last_std_frame_idx > 0:  replace the last_std_frame_idx with current frame
        # new_or_update and last_std_frame_idx <= 0: add new frame
        self.new_or_update_mode = -1
        self.last_std_frame_idx = -1
        self.last_measure_frame_idx = -1

        # whether is captured by doctor in US machine
        self.doctor_capture = False

        # auto_capture: 0, manual capture: 1, amonic measure: 2, lc（1） capture: 3, image_dicom: 4
        self.capture_type = 0

        # 早孕期，中晚孕期，妇科, ...
        self.FetalKind = None

        # classification results: [{'class_type': 0, 'class_score': 0}, ...]
        self.class_results = []

        # result {'annotations':, }
        self.detection_results = None

        # instance of MeasureInfo
        self.measure_results = None

        # previous image info with max detection score
        self.prev_max_image_info = None

        # 是否需要测量，用于区分测量失败与不需测量的情况
        self.need_measure = False

        # 判定当前帧里面是否有结节
        self.nodule_exists = False
        # ///////////// for test ////////////
        self.anno_set = None
        # ////////////////////////////////

        self.image_uid = None

        # used to differentiate each patient
        self.patient_id = 0

        # parameters for double plane in one image
        self.second_plane = False

        # start and end frame idx for video clip: [startIdx, endIdx]
        self.video_clip_range = None

        self.thyroid_clip_range = None

        # the frame idx to be replaced
        self.optimal_replace = -1

        # score given by segement
        self.measure_score = 0

        # 由UI提供，判断甲状腺纵切面的横纵
        # self.left_info_from_ui = "右叶"
        self.ruler_info = None

        self.is_unnormal_type = False

    @property
    def new_or_update(self):
        return self.new_or_update_mode >= 0

    def is_optimal_replace(self):
        return self.optimal_replace >= 0

    def is_manual_capture(self):
        return self.capture_type > 0

    def is_auto_capture(self):
        return self.capture_type == AUTO_CAPTURE_MODE

    @property
    def image_dicom(self):
        return self.capture_type == DICOM_TRANSFER_MODE

    @image_dicom.setter
    def image_dicom(self, dicom):
        self.capture_type = DICOM_TRANSFER_MODE if dicom else AUTO_CAPTURE_MODE

    @property
    def ignore(self):
        return self.auto_type in (0, 1, -1)

    @property
    def class_score(self):
        return self.class_results[0]['class_score'] if self.class_results else 0

    @property
    def class_type(self):
        # 其它
        return self.class_results[0]['class_type'] if self.class_results else 1

    @class_type.setter
    def class_type(self, type):
        if self.class_results:
            self.class_results[0]['class_type'] = type
        else:
            self.class_results = [{
                'class_type': type,
                'class_score': 0
            }]

    @property
    def auto_type(self):
        return self.detection_results['auto_type'] if self.detection_results else self.class_type

    @auto_type.setter
    def auto_type(self, plane_id):
        if self.detection_results:
            self.detection_results['auto_type'] = plane_id
        else:
            self.detection_results = {
                'auto_type': plane_id
            }

    @property
    def auto_score(self):
        if self.detection_results and 'auto_score' in self.detection_results:
            return self.detection_results['auto_score']
        else:
            return 0.0

    @property
    def video_score(self):
        if self.detection_results and 'video_score' in self.detection_results:
            return self.detection_results['video_score']
        else:
            return 0.0

    @video_score.setter
    def video_score(self, video_score):
        if self.detection_results:
            self.detection_results['video_score'] = video_score
        else:
            self.detection_results = {
                'video_score': video_score
            }

    @property
    def annotations(self):
        if self.detection_results and 'annotations' in self.detection_results:
            return self.detection_results['annotations']
        else:
            return None

    @property
    def std_type(self):
        if self.detection_results and 'auto_std_type' in self.detection_results:
            return self.detection_results['auto_std_type']
        else:
            return '非标准'

    def roi_image(self, roi=None, x_extend=0, y_extend=0):

        if self.image is None:
            return None

        if self.image_dicom and roi is None:
            return self.image

        if roi is None:
            # roi-左上角x y 和宽高
            roi = ImageInfo.roi
            if roi[2] <= 0:
                x_end = self.image.shape[1]
            else:
                x_end = roi[0] + roi[2] + x_extend

            # 提取从roi[1]到图像的最后一行，从roi[0]到图像的最后一列
            if roi[3] <= 0:
                y_end = self.image.shape[0]
            else:
                y_end = roi[1] + roi[3] + y_extend

            # when used for ruler detection, it need to extend the roi image to make sure the ruler is inside image
            x_start = roi[0] - x_extend if roi[0] > x_extend else 0
            y_start = roi[1] - y_extend if roi[1] > y_extend else 0

            return self.image[y_start:y_end, x_start:x_end]

        elif self.image_dicom:
            left = int(roi[0])
            right = int(roi[0] + roi[2])
            top = int(roi[1])
            bottom = int(roi[1] + roi[3])
            return self.image[top:bottom, left:right]
        else:
            left = int(roi[0] + ImageInfo.roi[0])
            right = int(left + roi[2])
            if ImageInfo.roi[2] > 0:
                right = min(right, ImageInfo.roi[0] + ImageInfo.roi[2])
            left = max(left, ImageInfo.roi[0])

            top = int(roi[1] + ImageInfo.roi[1])
            bottom = int(top + roi[3])
            if ImageInfo.roi[3] > 0:
                bottom = min(bottom, ImageInfo.roi[1] + ImageInfo.roi[3])
            top = max(top, ImageInfo.roi[1])

            roi[0:2] = left, top
            roi[2] = right - left
            roi[3] = bottom - top
            return self.image[top:bottom, left:right]

    def offset(self):
        # 返回左上角坐标
        if self.image_dicom:
            # dicom返回 0,0
            return 0, 0
        else:
            return self.roi[:2]

    def get_thyroid_type(self):
        results = self.detection_results
        if not results:
            return ''

        return results['cross_long'] + results['auto_std_type']

    def to_annotation_set(self, plane_name):

        results = self.detection_results
        # plane_name = db_id_to_name_map[self.class_type] if self.class_type in db_id_to_name_map else '未知'

        if hasattr(self, 'change_to_unknown'):
            plane_name = '未知'
            anno_set = AnnotationSet(plane_name, self.class_score, self.doctor_capture)
        # anno_set = None
        elif results and results['auto_type'] not in (0, 1, -1) and 'annotations' in results and results['annotations']:
            annotations = results['annotations']

            # annotations
            offset = self.offset()
            anno_set = AnnotationSet.from_json_str(annotations, offset)

            if self.is_thyroid():
                plane_name = '甲状腺'
                if results['left_info'] and results['left_info'] != '未知':
                    plane_name += results['left_info']
                plane_name += results['cross_long']

            anno_set.plane_type = plane_name
            anno_set.doctor_capture = self.doctor_capture
            anno_set.new_or_update_mode = self.new_or_update_mode

            anno_set.set_std_type(results['auto_std_type'])
            anno_set.score = float(results['auto_score']) if 'auto_score' in results else 0
            anno_set.video_score = float(results['video_score']) if 'video_score' in results else 0
            anno_set.confidence = float(results['confidence']) if 'confidence' in results else 0
            anno_set.avg_confidence = float(results['avg_confidence']) if 'avg_confidence' in results else 0

            # classification score
            anno_set.class_score = float(self.class_score)
            if self.class_results and len(self.class_results) > 1:
                anno_set.second_class_result = self.class_results[1]

            # roi_bbox is used for segmentation
            if 'seg_roi' in results and results['seg_roi'] is not None:
                anno_set.seg_roi = [round(x) for x in results['seg_roi']]

        else:
            anno_set = AnnotationSet(plane_name, self.class_score, self.doctor_capture)
            if results and 'auto_std_type' in results:
                anno_set.set_std_type(results['auto_std_type'])

        anno_set.is_still = self.is_still
        anno_set.zoom_in = self.zoom_in
        anno_set.has_sampling_line = self.has_sampling_line
        anno_set.optimal_replace = self.optimal_replace
        # used to ignore saving annotation
        anno_set.image_dicom = self.image_dicom
        anno_set.is_thyroid = self.is_thyroid()

        anno_set.video_clip_range = self.video_clip_range
        anno_set.thyroid_clip_range = self.thyroid_clip_range

        anno_set.measure_score = self.measure_score

        # measure results
        measure_info = self.measure_results
        if measure_info is not None:  #
            measure_info.update_ga()
        anno_set.measure_results = measure_info

        anno_set.patient_id = self.patient_id

        return anno_set

    def from_annoset(self, anno_set, auto_type, offset=(0, 0)):
        self.anno_set = anno_set
        if anno_set is None:
            return

        # measure results
        self.measure_results = anno_set.measure_results
        self.video_clip_range = anno_set.video_clip_range

        self.doctor_capture = anno_set.doctor_capture
        self.is_still = anno_set.is_still
        self.zoom_in = anno_set.zoom_in
        self.has_sampling_line = anno_set.has_sampling_line
        self.optimal_replace = anno_set.optimal_replace

        self.measure_score = anno_set.measure_score

        # class result
        self.class_results = [{
            'class_type': auto_type,
            'class_score': anno_set.class_score
        }]
        if anno_set.second_class_result:
            self.class_results.append(anno_set.second_class_result)

        offset = self.offset()
        if offset == (0, 0):
            annotations = [anno.to_json_object() for anno in anno_set.annotations]
        else:
            annotations = []
            for anno in anno_set.annotations:
                # translate
                anno.translate((-offset[0], -offset[1]))

                annotations.append(anno.to_json_object())

                # restore
                anno.translate(offset)

        # detection results
        self.detection_results = {
            'auto_type': auto_type,
            'auto_score': anno_set.score,
            'video_score': anno_set.video_score,
            'confidencce': anno_set.confidence,
            'avg_confidence': anno_set.avg_confidence,
            'auto_std_type': AnnotationSet.STD_LIST.index(anno_set.std_type) + 1,
            'seg_roi': anno_set.seg_roi,
            'annotations': annotations
        }

        self.new_or_update_mode = anno_set.new_or_update_mode

    def get_detection_info(self):
        detect_info = None
        if self.detection_results and 'annotations' in self.detection_results:
            detect_info = self.detection_results['annotations']

            # split by ,
            annotations = []
            for info in detect_info:
                if isinstance(info, str):
                    info = eval(info)
                if isinstance(info['vertex'], str):
                    vertices = list(map(float, info['vertex'].split(',')))
                    info['vertex'] = [vertices[:2], vertices[2:]]
                annotations.append(info)
            self.detection_results['annotations'] = annotations
            detect_info = annotations

        elif self.anno_set:
            detect_info = []
            for anno in self.anno_set.annotations:
                detect_info.append({
                    'name': anno.name,
                    'vertex': [[anno.ptStart[0] - self.roi[0], anno.ptStart[1] - self.roi[1]],
                               [anno.ptEnd[0] - self.roi[0], anno.ptEnd[1] - self.roi[1]]],
                    'score': anno.score
                })
        return detect_info

    def get_part_bbox(self, part_name, only_one=True):
        """
        get bbox for specified part name
        only_one: if true, return one bbox with part name only, else return all bboxes with part name
        bbox: [left_upper, right_bottom], coordinates are relative to roi if only_one = true, else
              [[left_upper, right_bottom], ...]
        """
        bbox_list = []
        if self.detection_results and 'annotations' in self.detection_results:
            detect_info = self.detection_results['annotations']
            for info in detect_info:
                if isinstance(info, str):
                    info = eval(info)
                if info['name'] == part_name:
                    vertices = info['vertex']
                    if isinstance(info['vertex'], str):
                        vertices = list(map(float, info['vertex'].split(',')))
                    elif isinstance(info['vertex'], list):
                        vertices = [element for sublist in vertices for element in sublist]
                    bbox = [vertices[:2], vertices[2:]]

                    if only_one:
                        return bbox
                    bbox_list.append(bbox)
        elif self.anno_set:
            for anno in self.anno_set.annotations:
                if anno.name == part_name:
                    bbox = [[anno.ptStart[0] - self.roi[0], anno.ptStart[1] - self.roi[1]],
                            [anno.ptEnd[0] - self.roi[0], anno.ptEnd[1] - self.roi[1]]]

                    if only_one:
                        return bbox
                    bbox_list.append(bbox)

        if not only_one:
            return bbox_list

    def get_seg_roi(self):
        if self.detection_results and 'seg_roi' in self.detection_results:
            return self.detection_results['seg_roi']
        if self.anno_set is not None:
            return self.anno_set.seg_roi

    def get_plane_type(self):
        if self.anno_set:
            return self.anno_set.plane_type

    @classmethod
    def annotations_to_json(cls, annotations, roi):
        detection_results = []
        for anno in annotations:
            vertex = "%.2f, %.2f, %.2f, %.2f" % (anno.ptStart[0] - roi[0], anno.ptStart[1] - roi[1],
                                                 anno.ptEnd[0] - roi[0], anno.ptEnd[1] - roi[1])
            detection = '{' + f'"name": "{anno.name}", "vertex": "{vertex}", "score": {anno.score}' + '}'
            detection_results.append(detection)

        return detection_results

    def is_thyroid(self):
        return self.FetalKind == '甲状腺'

    def is_earlier_measure(self):
        return self.FetalKind == '妇科' and self.class_type == 309  # 309: "早孕期切面"

    def is_fetal(self):
        return not self.is_thyroid()

    def is_amniotic(self):
        return self.capture_type == AMNIOTIC_MEASURE_MODE

    def is_thyroid_or_amniotic(self):
        return self.is_thyroid() or self.is_amniotic()

    def is_manual_or_measure(self):
        return self.capture_type > 0 or self.measure_results

    def is_manual_or_measure_or_std(self):
        return self.capture_type > 0 or self.new_or_update_mode >= 0 or self.measure_results

    def convert_envelop(self):
        """
        """
        if self.measure_results and isinstance(self.measure_results, SpectrumMeasureInfo):
            self.measure_results.convert_envelop()
