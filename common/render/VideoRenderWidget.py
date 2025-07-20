from common.model.image_info import ImageInfo
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPainter, QMouseEvent

from common.render.BaseRenderWidget import BaseRenderWidget


class VideoRenderWidget(BaseRenderWidget):
    ruler_calibrated_event = Signal(list)

    def __init__(self, parent=None):
        super(VideoRenderWidget, self).__init__(parent)
        # self.debug_info = None
        self.ruler_calibrating = False
        self.ruler_ticks = []
        # 在左上角显示fps
        self.fps = None

    def render_scene(self, painter):

        if self.frame is None:
            return

        # start = time.time()
        self.draw_frame(painter, self.frame)
        # print(time.time() - start)

        painter.translate(self.x_offset, self.y_offset)
        painter.scale(self.scale, self.scale)

        # annotation info
        text_to_show = None
        if self.annotation_set is not None:
            if isinstance(self.annotation_set, list):
                annotations = self.annotation_set
            else:
                annotations = self.annotation_set.get_annotations()

                text_to_show = '{} : c {:.2f}'.format(
                    self.annotation_set.plane_type,
                    self.annotation_set.class_score
                )
                # detection score is same for thyroid
                if not self.annotation_set.is_thyroid and (len(annotations) > 0 or self.annotation_set.score > 0):
                    text_to_show += ': d {:.2f}: vs {:.2f}: conf {:.2f}'.format(
                        self.annotation_set.score,
                        self.annotation_set.video_score,
                        self.annotation_set.confidence
                    )
                text_to_show += ' : ' + self.annotation_set.std_type

            # only to show annotations when score >= 60
            if self.config is not None and self.config['show_annotations']:  # and self.annotation_set.score >= 60:
                plane_type = '甲状腺切面' if self.annotation_set.is_thyroid else self.annotation_set.plane_type
                self.draw_annotations(painter, annotations, plane_type, scale=self.scale)

                # measure results
                measure_info = self.annotation_set.get_measure_results()
                if measure_info is not None:
                    self.draw_ruler_info(painter, measure_info.ruler_info)
                    ruler_unit_list = ' '.join([str(round(ruler_unit, 3)) for ruler_unit in measure_info.ruler_unit_list])
                    text_to_show += f': r {ruler_unit_list} cm'

        if len(self.ruler_ticks) > 1:
            # ruler calibrating
            self.draw_line(painter, self.ruler_ticks[0], self.ruler_ticks[1], 1, self.scale)

        # detection roi bbox
        if self.config['show_detection_roi']:

            fps_txt = self.fps_and_still_text()
            text_list = [text_to_show, fps_txt] if text_to_show else [fps_txt]

            roi = ImageInfo.roi
            h, w = self.frame.shape[:2]
            if not roi:
                roi = [0, 0, w, h]
            else:
                roi_w, roi_h = roi[2], roi[3]
                if roi[2] == 0:
                    roi_w = w - roi[0]
                if roi[3] == 0:
                    roi_h = h - roi[1]

                roi = [roi[0], roi[1], roi_w, roi_h]

            # if image_size:
            #     w, h = image_size
            #     if not roi:
            #         roi = [0, 0, w, h]
            #     elif roi[2] == 0 or roi[3] == 0:
            #         roi = [roi[0], roi[1], w - roi[0], h - roi[1]]

            self.draw_roi(painter, roi, text_list)

            if self.fps:
                image_height, image_width = self.frame.shape[:2]
                painter.setBrush(Qt.NoBrush)
                painter.drawText(20 * image_width // 1920, 50 * image_height // 1080, self.fps)

    def fps_and_still_text(self):
        text_to_show = ''
        if self.capture_fps > 0:
            text_to_show += f'capture fps: {self.capture_fps}, '
        if self.detection_fps > 0:
            text_to_show += f'detection fps: {self.detection_fps}, '

        if self.annotation_set:
            if self.annotation_set.second_class_result:
                result = self.annotation_set.second_class_result
                text_to_show += f'{result["class_type"]} : {result["class_score"]:.2f}, '
            if self.annotation_set.is_still:
                text_to_show += 'still, '
            if self.annotation_set.zoom_in:
                text_to_show += 'zoom in, '
            if self.annotation_set.has_sampling_line:
                text_to_show += 'sampling line, '

        return text_to_show

    @classmethod
    def draw_offline_annotations(cls, image, annotation_set, measure_mode,
                                 measure_reuslt_only=True, show_ruler_info=False):
        if annotation_set is None:
            return image

        measure_info = annotation_set.measure_results
        if measure_reuslt_only and measure_info is None:
            return image

        # annotations
        # image to dicom
        painter = QPainter(image)

        if not measure_reuslt_only:
            # painter.translate()
            score = annotation_set.score if annotation_set.score > 0 else annotation_set.class_score
            text_to_show = f'{annotation_set.plane_type} : {score:.3f} : {annotation_set.std_type}'
            if show_ruler_info and measure_info:
                text_to_show += f' : {measure_info.ruler_unit:.3f} cm'

            roi = ImageInfo.roi
            if not roi or roi[2] == 0 or roi[3] == 0:
                roi = [0, 0, image.width(), image.height()]
            cls.draw_roi(painter, roi, [text_to_show])

            annotations = annotation_set.get_annotations()
            plane_type = '甲状腺切面' if annotation_set.is_thyroid else annotation_set.plane_type
            # cls.draw_annotations(painter, annotations, plane_type)

        cls.draw_measure_info(painter, measure_info, measure_mode=measure_mode, show_ruler_info=show_ruler_info)

        painter.end()
        return image

    def mousePressEvent(self, event):
        if self.ruler_calibrating:
            pos = self.window_to_image_coord(event.x(), event.y())
            self.ruler_ticks = [pos]

    def mouseReleaseEvent(self, event) -> None:
        if self.ruler_calibrating:
            pos = self.window_to_image_coord(event.x(), event.y())
            self.ruler_ticks.append(pos)

            self.ruler_calibrated_event.emit(self.ruler_ticks)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        pos = self.window_to_image_coord(event.x(), event.y())
        # upper frame cannot recevive mouseMoveEvent, so here dispatch mouse move event again
        self.mouse_move_event.emit(int(pos[0]), int(pos[1]), -1)

        if self.ruler_calibrating and event.buttons() == Qt.LeftButton:
            if len(self.ruler_ticks) > 1:
                self.ruler_ticks[-1] = pos
            else:
                self.ruler_ticks.append(pos)

            self.update()
