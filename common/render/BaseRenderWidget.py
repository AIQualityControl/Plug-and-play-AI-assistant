import traceback

from common.FetalBiometry import FetalBiometry
from common.model.CRLMeasureInfo import CRLMeasureInfo
from common.model.CenaoMeasureInfo import CenaoMeasureInfo
from common.model.HLMeasureInfo import HLMeasureInfo
from common.model.NtMeasureInfo import NtMeasureInfo
from common.model.AFIMeasureInfo import AFIMeasureInfo
from common.model.PLMeasureInfo import PLMeasureInfo
from common.model.XiaonaoMeasureInfo import XiaonaoMeasureInfo
from common.model.image_info import ImageInfo
from common.model.HcMeasureInfo import HcMeasureInfo
from common.model.AcMeasureInfo import AcMeasureInfo
from common.model.FLMeasureInfo import FLMeasureInfo
from common.model.ZGZMeasureInfo import ZGZMeasureInfo
from common.model.ZGHMeasureInfo import ZGHMeasureInfo
from common.model.ZGGJMeasureInfo import ZGGJMeasureInfo
from common.model.LCMeasureInfo import LCMeasureInfo
from common.model.NoduleMeasureInfo import NoduleMeasureInfo
# from common.model.EarlyGynMeasureInfo import EarlyGynMeasureInfo
from common.model.FukeEarlierMeasureInfo import FukeEarlierMeasureInfo
from common.model.JizhuMeasureInfo import JizhuMeasureInfo
from common.model.HeartMeasureInfo import HeartMeasureInfo
from common.model.ThyroidMeasureInfo import ThyroidMeasureInfo
from common.model.SpectrumMeasureInfo import SpectrumMeasureInfo

from common.model.LineAnnotation import LineAnnotation
from common.model.BoxAnnotation import BoxAnnotation
from common.model.EllipseAnnotation import EllipseAnnotation
from common.model.PolygonAnnotation import PolygonAnnotation

from common.render.color_config import ColorManager
from PySide6.QtCore import QPoint, QRect, Qt, Signal
from PySide6.QtGui import QImage, QPixmap, QPen, QColor, QPainter, QBrush, QFont, QPolygon
from PySide6.QtWidgets import QWidget

import math
from loguru import logger
import numpy as np
import cv2 as cv

try:
    # from custom_ui.Collect_list_images.gestation_calculator import GestationCalculator
    from custom_ui.view_the_image.four_tool_and_config.painter_shadow import DrawShadow
except Exception:
    from ui.DrawShadow import DrawShadow


class BaseRenderWidget(QWidget):
    image_width = 1920
    image_height = 1080
    mouse_move_event = Signal(int, int, int)

    ga_sd_pctl_saver = None

    image_info = None

    color_manager = ColorManager()

    def __init__(self, parent=None):
        super(BaseRenderWidget, self).__init__(parent)

        self.frame = None
        self.annotation_set = None

        # accept key press event
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)

        self.scale = 1.0
        self.translate = [0, 0]
        self.auto_to_fit = True

        self.x_offset = 0
        self.y_offset = 0

        self.config = None
        self.capture_fps = 0
        self.detection_fps = 0

        # used for scale_to_fit when new file is opened
        self.new_opend = True

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        # self.style().drawPrimitive(QStyle::PE_Widget, opt, p, self);

        # p.begin(self)

        # background
        p.setPen(Qt.NoPen)
        brush = QBrush(QColor(0, 0, 0))
        p.setBrush(brush)

        p.drawRect(self.rect())

        if self.auto_to_fit and self.new_opend and self.frame is not None:
            self.scale_to_fit()
            self.new_opend = False

        # render scene
        self.render_scene(p)

        p.end()

    @classmethod
    def get_sd_and_pctl(cls, is_bpd=False):

        number_sd, measure_percentage = 0, 0
        if cls.ga_sd_pctl_saver:
            try:
                res = cls.ga_sd_pctl_saver.get_sd_and_pctl()
                if is_bpd:
                    number_sd, measure_percentage = res[2:]
                else:
                    number_sd, measure_percentage = res[:2]
            except Exception:
                logger.error(f'{traceback.format_exc()}')

        if -10 < number_sd < 10:
            number_sd_str = f'SD: {np.round(number_sd, 2)}'
            measure_percentage_str = f'Pctl: {np.round(measure_percentage * 100, 2)}%'
        else:
            number_sd_str = 'SD: ****'
            measure_percentage_str = 'Pctl: ****'

        return number_sd_str, measure_percentage_str

    def render_scene(self, painter):
        self.draw_frame(painter, self.frame)

    def update_frame(self, frame, annotation_set):
        self.frame = frame
        self.annotation_set = annotation_set

    def scale_to_fit(self, scale_large_only=True):
        if not self.auto_to_fit or self.frame is None:
            return

        if isinstance(self.frame, np.ndarray):
            height, width = self.frame.shape[:2]
        elif isinstance(self.frame, QPixmap):
            height = self.frame.height()
            width = self.frame.width()
        else:
            return

        if scale_large_only and self.width() >= width and self.height() >= height:
            self.scale = 1.0
            return

        sacle = width / height
        if sacle > self.width() / self.height():
            self.scale = self.width() / width
        elif sacle < self.width() / self.height():
            self.scale = self.height() / height
        else:
            self.scale = 1.0

    def draw_frame(self, painter, frame):
        if frame is None:
            return

        #  convert from numpy(opencv) to QImage
        if type(frame) is QPixmap:
            pixmap = frame
            width = pixmap.width()
            height = pixmap.height()
            width *= self.scale
            height *= self.scale

        else:
            height, width, channels = frame.shape
            frame = self.img_resize(img=frame, width=width, height=height, interpolation=cv.INTER_CUBIC)
            height, width, channels = frame.shape
            bytes_per_line = width * channels
            qImg = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
            if qImg.isNull():
                logger.error('Failed to covert image from opencv to Qimage')
                return
            pixmap = QPixmap.fromImage(qImg.rgbSwapped())

        # scale first and then translate

        self.x_offset = int((self.width() - width) // 2)
        self.y_offset = int((self.height() - height) // 2)

        painter.drawPixmap(self.x_offset, self.y_offset, int(width), int(height), pixmap)
        return width, height

    def img_resize(self, img, width, height, interpolation=cv.INTER_LINEAR):
        img = cv.resize(img, (int(int(width) * self.scale), int(int(height) * self.scale)), interpolation=interpolation)
        return img

    @classmethod
    def draw_annotations(cls, painter, annotations, plane_type='', scale=1):
        if annotations is None or len(annotations) == 0:
            return

        for anno in annotations:
            if anno is None:
                continue

            if anno.name.endswith('清晰'):
                continue

            type_anno = type(anno)
            if type_anno == LineAnnotation:
                cls.draw_line_annotation(painter, anno, scale)
            elif type_anno == BoxAnnotation:
                cls.draw_box_annotation(painter, anno, plane_type)
            elif type_anno == EllipseAnnotation:
                cls.draw_ellipse_annotation(painter, anno)
            elif type_anno == PolygonAnnotation:
                cls.draw_polygon_annotation(painter, anno)

    @classmethod
    def get_font_size(cls, base_size=22):
        return int(base_size * (cls.image_height / 1080))

    @classmethod
    def draw_measure_info(cls, painter, measure_info, show_ruler_info=False, measure_mode='hadlock',
                          image_width=1920, image_height=1080, ga_sd_pctl_saver=None, image_info=None):

        try:

            if measure_info is None:
                return

            cls.image_width = image_width
            cls.image_height = image_height

            cls.image_info = image_info

            if ga_sd_pctl_saver:
                cls.ga_sd_pctl_saver = ga_sd_pctl_saver

            type_measure = type(measure_info)
            if type_measure == FLMeasureInfo:
                cls.draw_fl_measure_info(painter, measure_info)
            elif type_measure == AcMeasureInfo:
                cls.draw_ac_measure_info(painter, measure_info)
            elif type_measure == HcMeasureInfo:
                cls.draw_hc_measure_info(painter, measure_info, measure_mode)
            elif type_measure == HLMeasureInfo:
                cls.draw_hl_measure_info(painter, measure_info)
            elif type_measure == NtMeasureInfo:
                cls.draw_dist_measure_info(painter, measure_info, 'NT: ')
            elif type_measure == CRLMeasureInfo:
                cls.draw_crl_measure_info(painter, measure_info)
            elif type_measure == PLMeasureInfo:
                cls.draw_dist_measure_info(painter, measure_info, 'PL: ')
            elif type_measure == AFIMeasureInfo:
                cls.draw_afi_measure_info(painter, measure_info)
            elif type_measure == XiaonaoMeasureInfo:
                cls.draw_xiaonao_measure_info(painter, measure_info)
            elif type_measure == CenaoMeasureInfo:
                cls.draw_cenao_measure_info(painter, measure_info, measure_mode)
            elif type_measure == ZGZMeasureInfo:
                cls.draw_zgz_measure_info(painter, measure_info)
            elif type_measure == ZGHMeasureInfo:
                cls.draw_zgh_measure_info(painter, measure_info)
            elif type_measure == ZGGJMeasureInfo:
                cls.draw_zggj_measure_info(painter, measure_info)
            elif type_measure == LCMeasureInfo:
                cls.draw_lc_measure_info(painter, measure_info)
            elif type_measure == NoduleMeasureInfo or type_measure == ThyroidMeasureInfo:
                cls.draw_nodule_measure_info(painter, measure_info)
            # elif type_measure == EarlyGynMeasureInfo:
            #     cls.draw_earlygyn_measure_info(painter, measure_info)
            elif type_measure == FukeEarlierMeasureInfo:
                cls.draw_earlier_measure_info(painter, measure_info)
            elif type_measure == JizhuMeasureInfo:
                cls.draw_jizhu_measure_info(painter, measure_info)
            elif type_measure == HeartMeasureInfo:
                cls.draw_heart_measure_info(painter, measure_info)
            elif type_measure == SpectrumMeasureInfo:
                cls.draw_spectrum_measure_info(painter, measure_info)

            if show_ruler_info and hasattr(measure_info, 'ruler_info'):
                cls.draw_ruler_info(painter, measure_info.ruler_info)

        except Exception:
            logger.error(f'{traceback.format_exc()}')

    @classmethod
    def draw_line(cls, painter, start, end, line_width=2, cross_scale=1, dotted_line=False, color=QColor(255, 255, 0)):
        """
        cross_scale: used for draw cross, if cross_scale is 0, cross will not be drawed
        """
        painter.setBrush(Qt.NoBrush)

        # color = QColor(255, 255, 0)
        pen = QPen(color, line_width)

        # 设置虚线样式
        if dotted_line:
            pen.setDashPattern([2, 5])

        painter.setPen(pen)

        # convert to int
        start = [int(start[0]), int(start[1])]
        end = [int(end[0]), int(end[1])]

        painter.drawLine(QPoint(start[0], start[1]), QPoint(end[0], end[1]))

        # draw cross
        if cross_scale > 0:
            cls.draw_cross(painter, start, line_width, cross_scale, active_status=False)
            cls.draw_cross(painter, end, line_width, cross_scale, active_status=False)

    @classmethod
    def draw_line_annotation(cls, painter, line_anno, line_width=2, cross_scale=1, color=QColor(255, 255, 0)):
        """
        cross_scale: cross scale, if cross_scale <= 0, do not draw cross
        """
        if line_anno is None:
            return

        painter.setBrush(Qt.NoBrush)

        if hasattr(line_anno, "is_default_value") and line_anno.is_default_value:
            color = QColor(255, 0, 0)
        # color = QColor(255, 255, 0)
        if line_anno.is_selected() or line_anno.is_highlight():
            color = QColor(0, 255, 0)

        pen = QPen(color, line_width)

        # if line_anno.is_selected():
        # 这一句和线段的虚线有关
        pen.setDashPattern([2, 5])

        painter.setPen(pen)

        start = line_anno.start_point()
        end = line_anno.end_point()

        # convert to int
        start = [int(start[0] + 0.5), int(start[1] + 0.5)]
        end = [int(end[0] + 0.5), int(end[1] + 0.5)]

        painter.drawLine(QPoint(start[0], start[1]), QPoint(end[0], end[1]))

        # draw cross
        if cross_scale >= 0:
            cls.draw_cross(painter, start, line_width, cross_scale,
                           active_status=(line_anno.get_active_endpoint_idx() == 0))
            cls.draw_cross(painter, end, line_width, cross_scale,
                           active_status=(line_anno.get_active_endpoint_idx() == 1))

    @classmethod
    def draw_cross(cls, painter, pt_center, line_width=1, scale=1, color=QColor(255, 255, 0), active_status=False):
        q_color = color  # 黄色
        if active_status:
            q_color = QColor(0, 255, 0)  # 绿色

        pen = QPen(q_color, line_width)
        painter.setPen(pen)
        offset = int(8 / scale)
        painter.drawLine(QPoint(pt_center[0] - offset, pt_center[1]), QPoint(pt_center[0] + offset, pt_center[1]))
        painter.drawLine(QPoint(pt_center[0], pt_center[1] - offset), QPoint(pt_center[0], pt_center[1] + offset))

    @classmethod
    def draw_triangle_list(cls, painter: QPainter, pt_list, active_status=False, down_dir_flag=True):
        """
        dir_flag: True, 尖点朝下，否则，尖点朝上
        """
        q_color = QColor(255, 0, 0)  # 红色
        if active_status:
            q_color = QColor(0, 255, 0)  # 绿色

        pen = QPen(q_color)
        brush = QBrush(q_color)

        painter.setPen(pen)
        painter.setBrush(brush)

        for pt in pt_list:
            if down_dir_flag:
                pt1 = QPoint(pt[0], pt[1])
                pt2 = QPoint(pt[0] - 5, pt[1] - 10)
                pt3 = QPoint(pt[0] + 5, pt[1] - 10)
            else:
                pt1 = QPoint(pt[0], pt[1])
                pt2 = QPoint(pt[0] - 5, pt[1] + 10)
                pt3 = QPoint(pt[0] + 5, pt[1] + 10)
            point = [pt1, pt2, pt3]

            painter.drawPolygon(point)

    @classmethod
    def draw_box_annotation(cls, painter, box_anno, plane_type='', line_width=1):
        painter.setBrush(Qt.NoBrush)

        color = cls.color_manager.get_color(plane_type, box_anno.name)
        pen = QPen(QColor(color[0], color[1], color[2]), line_width)
        painter.setPen(pen)
        font = QFont('Arial', 18)
        painter.setFont(font)

        # a=box_anno.name
        start = box_anno.start_point()
        end = box_anno.end_point()

        pt_start = QPoint(int(start[0] + 0.5), int(start[1] + 0.5))
        pt_end = QPoint(int(end[0] + 0.5), int(end[1] + 0.5))
        painter.drawRect(QRect(pt_start, pt_end))

        # name + score
        name = box_anno.get_name()
        if box_anno.score > 0:
            name += f': {round(box_anno.score * 100)}'

        painter.drawText(pt_start, name)

    @classmethod
    def draw_ellipse_annotation(cls, painter, ellipse_anno, line_width=2, draw_axis=False, color=QColor(255, 255, 0)):
        if ellipse_anno is None:
            return

        painter.setBrush(Qt.NoBrush)

        # 设置画笔
        if hasattr(ellipse_anno, "is_default_value") and ellipse_anno.is_default_value:
            color = QColor(255, 0, 0)

        # color = QColor(255, 255, 0)     # 黄色
        if ellipse_anno.is_selected() or ellipse_anno.is_highlight():
            color = QColor(0, 255, 0)  # 绿色

        pen = QPen(color, line_width)
        pen.setDashPattern([2, 5])  # 用来画虚线，第一个参数是每段实线长度，第二个参数是每段实线的间隔长
        painter.setPen(pen)
        painter.save()

        # 旋转椭圆
        start = ellipse_anno.start_point()
        end = ellipse_anno.end_point()
        pt_center = ellipse_anno.center_point()

        # convert to int
        start = [int(start[0] + 0.5), int(start[1] + 0.5)]
        end = [int(end[0] + 0.5), int(end[1] + 0.5)]
        pt_center = [int(pt_center[0] + 0.5), int(pt_center[1] + 0.5)]

        # logger.info(f'start: {start}, end: {end}, mid: {pt_center}, degree: {round(ellipse_anno.degree(), 2)}')

        painter.translate(pt_center[0], pt_center[1])
        painter.rotate(ellipse_anno.degree())
        painter.translate(-pt_center[0], -pt_center[1])

        painter.drawEllipse(QRect(QPoint(start[0], start[1]), QPoint(end[0], end[1])))
        # painter.drawEllipse(QRect(QPoint(100, 100), QPoint(200, 400)))

        if draw_axis:
            # pen = QPen(QColor(255, 0, 0), line_width)
            # painter.setPen(pen)
            # pen.setColor(QColor(255, 0, 0))

            # pt0, pt1 = ellipse_anno.major_radius_points()
            y = (start[1] + end[1]) // 2
            painter.drawLine(QPoint(start[0], y), QPoint(end[0], y))

            # pt0, pt1 = ellipse_anno.minor_radius_points()
            x = (start[0] + end[0]) // 2
            painter.drawLine(QPoint(x, start[1]), QPoint(x, end[1]))

        # if ellipse_anno.is_selected():
        # draw cross
        cls.draw_cross(painter, (start[0], pt_center[1]), line_width,
                       active_status=(ellipse_anno.get_active_endpoint_idx() == 0))
        cls.draw_cross(painter, (end[0], pt_center[1]), line_width,
                       active_status=(ellipse_anno.get_active_endpoint_idx() == 1))

        painter.restore()

        # pos = ellipse_anno.convert_to_global((start[0], pt_center[1]))
        # cls.draw_cross(painter, pos, line_width)

    @classmethod
    def draw_polygon_annotation(cls, painter: QPainter, polygon_anno, line_width=1):
        if not polygon_anno or polygon_anno.num_of_points() == 0:
            return

        color = QColor(255, 255, 0)  # 黄色
        if polygon_anno.is_selected() or polygon_anno.is_highlight():
            color = QColor(0, 255, 0)  # 绿色

        # 设置画笔
        if line_width > 0:
            pen = QPen(color, line_width)
            painter.setPen(pen)

        brush = QBrush(color)
        painter.setBrush(brush)
        painter.setOpacity(0.2)

        points = []
        pt_center = [0, 0]
        for pt in polygon_anno.all_points():
            points.append(QPoint(int(pt[0]), int(pt[1])))
            pt_center[0] += pt[0]
            pt_center[1] += pt[1]
        painter.drawPolygon(QPolygon(points))

        name = polygon_anno.get_name()

        pt_center[0] /= polygon_anno.num_of_points()
        pt_center[1] /= polygon_anno.num_of_points()

        # inverse color
        color = QColor(255 - color.red(), 255 - color.green(), 255 - color.blue())
        pen = QPen(color, line_width)
        painter.setPen(pen)

        font = painter.font()
        font.setPointSize(11)
        painter.setFont(font)

        # restore opacity
        painter.setOpacity(1.0)
        painter.drawText(QPoint(int(pt_center[0] - len(name) * 5), int(pt_center[1] + 5)), name)

    @classmethod
    def draw_polyline_annotation(cls, painter, polyline_anno, line_width=2, dashed_line=True):
        painter.setBrush(Qt.NoBrush)

        # 设置画笔
        color = QColor(255, 255, 0)  # 黄色
        if polyline_anno.is_selected() or polyline_anno.is_highlight():
            color = QColor(0, 255, 0)  # 绿色
        pen = QPen(color, line_width)
        if dashed_line:
            pen.setDashPattern([2, 5])  # 用来画虚线，第一个参数是每段实线长度，第二个参数是每段实线的间隔长
        painter.setPen(pen)

        # polyline
        points = []
        for pt in polyline_anno.all_points():
            points.append(QPoint(int(pt[0]), int(pt[1])))
        painter.drawPolyline(QPolygon(points))

        if polyline_anno.is_selected():
            # control points
            color = QColor(255, 255, 0)  # 黄色
            pen = QPen(color, line_width)
            painter.setPen(pen)
            for pt in points:
                painter.drawEllipse(pt, 3, 3)

    @classmethod
    def draw_envelop_annotation(cls, painter, envelop_anno, line_width=2, dashed_line=True, down_dir_flag=True):
        painter.setBrush(Qt.NoBrush)

        # 设置画笔
        color = QColor(255, 255, 0)  # 黄色
        if envelop_anno.is_selected() or envelop_anno.is_highlight():
            color = QColor(0, 255, 0)  # 绿色
        pen = QPen(color, line_width)
        if dashed_line:
            pen.setDashPattern([2, 5])  # 用来画虚线，第一个参数是每段实线长度，第二个参数是每段实线的间隔长
        painter.setPen(pen)

        # polyline
        points = []
        for pt in envelop_anno.all_points():
            points.append(QPoint(int(pt[0]), int(pt[1])))
        painter.drawPolyline(QPolygon(points))

        # key points
        key_points = []
        for idx in envelop_anno.key_points_idx:
            pt = envelop_anno.point_at(idx)
            key_points.append(pt)
        cls.draw_triangle_list(painter, key_points, down_dir_flag=down_dir_flag)

    @classmethod
    def draw_roi(cls, painter, roi, text_list=None, text_pos=None):
        if roi is None or len(roi) != 4:
            return

        painter.setBrush(Qt.NoBrush)

        pen = QPen(QColor(255, 0, 0))
        painter.setPen(pen)

        # rectangle for roi
        if roi[2] > 0 and roi[3] > 0:
            painter.drawRect(QRect(QPoint(roi[0], roi[1]), QPoint(roi[0] + roi[2], roi[1] + roi[3])))

        # text
        if text_list:
            if text_pos is None:
                # draw text at top center
                x = roi[0] + roi[2] // 5
                y = roi[1] + 50 if roi[1] > 10 else roi[1] + roi[3] // 10
            else:
                x, y = text_pos

            cls.draw_text(painter, text_list, x, y, 18, QColor(255, 255, 0), is_use_text_shadow=False)

    @classmethod
    def get_info_offset(cls, spectrum=False):
        if cls.image_info and 'use_roi' in cls.image_info:
            pt_start = cls.image_info['use_roi'][:2]
            half_height = math.ceil(cls.image_info['use_roi'][3] / 2)
        else:
            pt_start = ImageInfo.roi[:2]  # 现在没有了类方法offset，改为使用roi[:2]
            half_height = math.ceil(ImageInfo.roi[3] / 2)

        if not spectrum:
            x = pt_start[0] + 50
            y = pt_start[1] + 50

            if pt_start[1] < 5:
                y = pt_start[1] + 140 + half_height
            elif y < half_height:
                y = 140 + half_height
        else:
            x = pt_start[0] + 100
            y = pt_start[1] + 200

        return x, y

    @classmethod
    def draw_fl_measure_info(cls, painter, fl_info):
        if fl_info is None:
            return
        cls.draw_line_annotation(painter, fl_info)

        font_size = cls.get_font_size()

        # fl and ga info in text
        x, y = cls.get_info_offset()

        fl = 'FL: ' + str(round(fl_info.fl, 2)) + ' cm'

        x, y = cls.draw_text(painter, [fl], x, y, font_size=font_size)

        # 计算Fl的孕周
        fl_ga = cls.ga_sd_pctl_saver.get_ga() if cls.ga_sd_pctl_saver else 0

        x, y = cls.draw_ga(painter, fl_ga, x, y, font_size=font_size)

        # 计算Fl的SD和百分位
        number_sd_str, measure_percentage_str = cls.get_sd_and_pctl()

        x, y = cls.draw_text(painter, [number_sd_str, measure_percentage_str], x, y, font_size=font_size)
        #
        if fl_info.all_biometry:
            cls.draw_all_biometry(painter, x, y, font_size=font_size)

    @classmethod
    def draw_hl_measure_info(cls, painter, hl_info):
        if hl_info is None:
            return

        cls.draw_line_annotation(painter, hl_info)

        x, y = cls.get_info_offset()
        hl = 'HL: ' + str(round(hl_info.hl, 2)) + 'cm'

        x, y = cls.draw_text(painter, [hl], x, y, font_size=cls.get_font_size())

        # 计算HL的孕周
        hl_ga = cls.ga_sd_pctl_saver.get_ga() if cls.ga_sd_pctl_saver else 0

        cls.draw_ga(painter, hl_ga, x, y, font_size=cls.get_font_size())

    @classmethod
    def draw_ac_measure_info(cls, painter, ac_info):
        if ac_info is None:
            return
        cls.draw_ellipse_annotation(painter, ac_info)

        # ac and ga info in text
        x, y = cls.get_info_offset()

        font_size = cls.get_font_size()

        ac = 'AC: ' + str(round(ac_info.ac, 2)) + ' cm'
        x, y = cls.draw_text(painter, [ac], x, y, font_size=font_size)

        # 计算AC的孕周
        ac_ga = cls.ga_sd_pctl_saver.get_ga() if cls.ga_sd_pctl_saver else 0

        x, y = cls.draw_ga(painter, ac_ga, x, y, font_size=font_size)

        # 计算AC的SD和百分位
        number_sd_str, measure_percentage_str = cls.get_sd_and_pctl()

        x, y = cls.draw_text(painter, [number_sd_str, measure_percentage_str], x, y, font_size=font_size)

        if ac_info.all_biometry:
            cls.draw_all_biometry(painter, x, y, font_size=font_size)

    @classmethod
    def draw_hc_measure_info(cls, painter, hc_info, measure_mode):

        if hc_info is None:
            return

        hadlock_mode = measure_mode == 'hadlock'
        if hc_info.hc_annotation is not None:
            cls.draw_ellipse_annotation(painter, hc_info.hc_annotation, draw_axis=not hadlock_mode)

        if hadlock_mode and hc_info.hadlock_bpd_annotation:
            cls.draw_line_annotation(painter, hc_info.hadlock_bpd_annotation)

        # hc and bpd and ga info in text
        x, y = cls.get_info_offset()

        # y -= int(120 * (cls.image_width / 1920))
        y -= cls.get_font_size(base_size=120)

        font_size = cls.get_font_size()
        # space = int(35 * (cls.image_width / 1920))
        space = cls.get_font_size(base_size=35)

        # HC
        ofd = hc_info.get_ofd()
        ofd = 'OFD: ' + str(round(ofd, 2)) + ' cm'
        hc = 'HC: ' + str(round(hc_info.hc, 2)) + ' cm'

        # x, y = cls.draw_text(painter, [ofd, hc], x, y, font_size)

        # 计算HC的孕周
        if cls.ga_sd_pctl_saver:
            hc_ga, _ = cls.ga_sd_pctl_saver.get_ga()
        else:
            hc_ga = 0

        # x, y = cls.draw_ga(painter, hc_ga, x, y, font_size=font_size)

        # 计算HC的SD和百分位
        number_sd_str, measure_percentage_str = cls.get_sd_and_pctl()

        # x, y = cls.draw_text(painter, [number_sd_str, measure_percentage_str], x, y, font_size=font_size)

        if hadlock_mode and hc_info.hadlock_bpd_annotation is None or \
                not hadlock_mode and hc_info.hc_annotation is None:
            return x, y

        # h split line
        y += space // 2
        painter.setPen(QPen(QColor(255, 255, 255)))
        painter.drawLine(QPoint(x, y), QPoint(x + 160, y))

        # BPD
        y += space
        bpd = hc_info.get_bpd(measure_mode)

        bpd = 'BPD: ' + str(round(bpd, 2)) + ' cm'
        # x, y = cls.draw_text(painter, [bpd], x, y, font_size=font_size)

        # 计算BPD的孕周
        if cls.ga_sd_pctl_saver:
            _, bpd_ga = cls.ga_sd_pctl_saver.get_ga()
        else:
            bpd_ga = 0

        x, y = cls.draw_ga(painter, bpd_ga, x, y, font_size=font_size)

        # 计算BPD的SD和百分位
        number_sd_str, measure_percentage_str = cls.get_sd_and_pctl(is_bpd=True)

        # x, y = cls.draw_text(painter, [number_sd_str, measure_percentage_str], x, y, font_size=font_size)

        # if hc_info.all_biometry:
        #     x, y = cls.draw_all_biometry(painter, x, y, font_size=font_size)

        return x, y

    @classmethod
    def draw_dist_measure_info(cls, painter, info, prefix='NT: '):
        if info is None:
            return

        # color
        cls.draw_line_annotation(painter, info)

        # fl and ga info in text
        x, y = cls.get_info_offset()

        font_size = cls.get_font_size()

        dist = prefix + str(round(info.measure_length, 2)) + ' cm'
        dist_list = [dist]

        # if prefix == 'CRL: ' and info.is_corrected():
        #     dist = f'Corrected CRL: {round(info.corrected_length, 2)} cm'
        #     dist_list.append(dist)

        x, y = cls.draw_text(painter, dist_list, x, y, font_size=font_size)

        return x, y

    @classmethod
    def draw_crl_measure_info(cls, painter: QPainter, info: CRLMeasureInfo):
        text_list = []
        if info.crl_anno:
            cls.draw_line_annotation(painter, info.crl_anno)
            text_list.append(f'CRL: {round(info.crl_length, 2)} cm')

            if info.center_point and info.corrected_crl_anno:
                cls.draw_line(painter, info.center_point, info.crl_anno.start_point(), line_width=1)
                cls.draw_line(painter, info.center_point, info.crl_anno.end_point(), line_width=1)

        if info.corrected_crl_anno:
            cls.draw_line_annotation(painter, info.corrected_crl_anno, color=QColor(0, 255, 0))
            text_list.append(f'Corrected CRL: {round(info.corrected_crl_length, 2)} cm')

            if info.center_point:
                cls.draw_line(painter, info.center_point, info.corrected_crl_anno.start_point(),
                              line_width=1, color=QColor(0, 255, 0))
                cls.draw_line(painter, info.center_point, info.corrected_crl_anno.end_point(),
                              line_width=1, color=QColor(0, 255, 0))

        x, y = cls.get_info_offset()
        font_size = cls.get_font_size()

        x, y = cls.draw_text(painter, text_list, x, y, font_size=font_size)
        return x, y

    @classmethod
    def draw_afi_measure_info(cls, painter, info):
        if info is None:
            return

        for anno in info.amniotic_annos:
            if anno.is_default_value:
                cls.draw_line_annotation(painter, anno, color=QColor(255, 0, 0))
            else:
                cls.draw_line_annotation(painter, anno)

        # text
        x, y = cls.get_info_offset()

        font_size = cls.get_font_size()
        y -= (4 * font_size)

        af_list = []
        for i, depth in enumerate(info.amniotic_depths):
            af = f'Q{i + 1}: {round(depth, 2)} cm'
            af_list.append(af)

        x, y = cls.draw_text(painter, af_list, x, y, font_size=cls.get_font_size())

        # afi
        if len(af_list) > 1:
            space = font_size

            # h split line
            # y -= space
            painter.setPen(QPen(QColor(255, 255, 255)))
            painter.drawLine(QPoint(x - 10, y), QPoint(x + 150, y))

            y += space * 2 + 4
            afv = f'AFV: {round(info.afv, 2)} cm'
            afi = f'AFI: {round(info.afi, 2)} cm'
            cls.draw_text(painter, [afv, afi], x, y, color=QColor(255, 255, 0), font_size=font_size)

    @classmethod
    def draw_xiaonao_measure_info(cls, painter, info):
        x, y = cls.get_info_offset()
        cls.draw_line_annotation(painter, info.tcd_anno)

        text = f'TCD: {round(info.tcd, 2)} cm'
        # cls.draw_text(painter, [text], x, y, font_size=cls.get_font_size())

    @classmethod
    def draw_cenao_measure_info(cls, painter, info, measure_mode):
        if not FetalBiometry.is_hc_plane_detected:
            x, y = cls.draw_hc_measure_info(painter, info, measure_mode)
        else:
            x, y = cls.get_info_offset()

        if info.lvw_anno is None:
            return

        cls.draw_line_annotation(painter, info.lvw_anno)

        font_size = cls.get_font_size()
        space = font_size + 15

        # h split line
        y += space // 2
        painter.setPen(QPen(QColor(255, 255, 255)))
        painter.drawLine(QPoint(x, y), QPoint(x + 160, y))

        y += space

        text = f'LVW: {round(info.lvw, 2)} cm'
        # cls.draw_text(painter, [text], x, y, font_size=font_size)

    @classmethod
    def draw_text(cls, painter, text_list, x, y, font_size=22, color=QColor(252, 243, 81), space=None,
                  is_use_text_shadow=True):

        if not space:
            # space = int(35 * (cls.image_width / 1920))
            space = cls.get_font_size(base_size=35)

        if is_use_text_shadow:
            for text in text_list:
                painter.save()
                DrawShadow.draw_shadow(painter=painter, text=text, x=x, y=y, font_size=font_size)
                y += space
                painter.restore()
        else:
            # 设置Qpainter
            pen = QPen(color)
            painter.setPen(pen)
            font = QFont('Arial', font_size)
            painter.setFont(font)

            for text in text_list:
                # 绘制文字
                painter.drawText(x, y, text)

                y += space

        return x, y

    @classmethod
    def draw_ga(cls, painter, ga, x, y, color=QColor(252, 243, 81), font_size=22, space=None):

        if not space:
            # space = int(35 * (cls.image_width / 1920))
            space = cls.get_font_size(base_size=35)

        weeks = int(ga)
        days = round((ga - weeks) * 7)
        ga_str = f'GA: {weeks}w{days}d'

        # painter.drawText(x, y, ga_str)

        painter.save()
        DrawShadow.draw_shadow(painter=painter, text=ga_str, x=x, y=y, font_size=font_size)
        painter.restore()

        y += space

        return x, y

    @classmethod
    def draw_all_biometry(cls, painter, x, y, font_size=14, space=None):

        if not space:
            # space = int(35 * (cls.image_width / 1920))
            space = cls.get_font_size(base_size=35)

        pen = QPen(QColor(255, 255, 255))
        painter.setPen(pen)

        # space = font_size + 20
        # split line
        y += space // 2
        painter.drawLine(QPoint(x, y), QPoint(x + 160, y))
        y += space - 3

        # GA, EFW
        efw = 'EFW: ' + str(round(FetalBiometry.get_efw())) + 'g'
        x, y = cls.draw_text(painter, [efw], x, y, font_size=font_size)

        # ga = FetalBiometry.get_composite_ga()
        # cls.draw_ga(painter, ga, x, y, font_size=font_size)

        # split line
        # y += space // 2
        painter.setPen(pen)
        painter.drawLine(QPoint(x, y), QPoint(x + 160, y))
        y += space - 3

        # ratio
        fl_hc_ratio = 'FL/HC: ' + str(round(FetalBiometry.fl_hc_ratio() * 100, 2)) + ' %'
        hc_ac_ratio = 'HC/AC: ' + str(round(FetalBiometry.hc_ac_ratio() * 100, 2)) + ' %'
        return cls.draw_text(painter, [fl_hc_ratio, hc_ac_ratio], x, y, font_size=font_size)

    @classmethod
    def draw_zgz_measure_info(cls, painter, zgz_info):
        if zgz_info is None:
            return

        text_to_show = []
        # 绘制直线
        if zgz_info.zg_bidiameter:
            zg = zgz_info.zg_bidiameter
            cls.draw_bidiameter(painter, zg)

            info = f'Uterus: {zg.major_len:.2f} cm X {zg.minor_len:.2f} cm'
            text_to_show.append(info)

        if zgz_info.nm_anno:
            cls.draw_line_annotation(painter, zgz_info.nm_anno)
            nm = 'NM: ' + str(round(zgz_info.nm, 2)) + 'cm'

            text_to_show.append(nm)

        nodule_anno_list = zgz_info.jl_anno_list
        if nodule_anno_list:
            for i, nodule in enumerate(nodule_anno_list):
                label = nodule.get_name() if nodule.name else f'#{i + 1}'
                # 绘制直线
                cls.draw_bidiameter(painter, nodule, label=label)

                # 增加实际长度在图上
                text_to_show.append(f'{label}: {nodule.major_len:.2f} cm X {nodule.minor_len:.2f} cm')
                if nodule.custom_props and 'cls_props' in nodule.custom_props:
                    prop_list = []
                    for prop in nodule.custom_props['cls_props'].values():
                        prop_list.append(str(prop))
                    text_to_show.append(f'{label}: {"; ".join(prop_list)}')

        # 增加实际长度在图上
        if text_to_show:
            font_size = cls.get_font_size()
            font = QFont('Arial', font_size)
            painter.setFont(font)
            x, y = cls.get_info_offset()
            cls.draw_text(painter, text_to_show, x, y, font_size)

    @classmethod
    def draw_zgh_measure_info(cls, painter, zgh_info):
        if zgh_info is None:
            return

        text_to_show = []
        # 绘制直线
        if zgh_info.zghor_anno:
            cls.draw_line_annotation(painter, zgh_info.zghor_anno)

            info = 'Hor:' + str(round(zgh_info.zghor, 2)) + 'cm'
            text_to_show.append(info)

        nodule_anno_list = zgh_info.jl_anno_list
        if nodule_anno_list:
            for i, nodule in enumerate(nodule_anno_list):
                label = nodule.get_name() if nodule.name else f'#{i + 1}'
                # 绘制直线
                cls.draw_bidiameter(painter, nodule, label=label)

                # 增加实际长度在图上
                text_to_show.append(f'{label}: {nodule.major_len:.2f} cm X {nodule.minor_len:.2f} cm')
                if nodule.custom_props and 'cls_props' in nodule.custom_props:
                    prop_list = []
                    for prop in nodule.custom_props['cls_props'].values():
                        prop_list.append(str(prop))
                    text_to_show.append(f'{label}: {"; ".join(prop_list)}')

        # 增加实际长度在图上
        if text_to_show:
            font_size = cls.get_font_size()
            font = QFont('Arial', font_size)
            painter.setFont(font)
            x, y = cls.get_info_offset()
            cls.draw_text(painter, text_to_show, x, y, font_size)

    @classmethod
    def draw_zggj_measure_info(cls, painter, zggj_info):
        if zggj_info is None:
            return

        text_to_show = []
        # 绘制直线
        if zggj_info.gjmaj_anno:
            cls.draw_line_annotation(painter, zggj_info.gjmaj_anno)
            major = 'Major:' + str(round(zggj_info.gjmaj, 2)) + 'cm'
            text_to_show.append(major)

        if zggj_info.gjmin_anno:
            cls.draw_line_annotation(painter, zggj_info.gjmin_anno)
            minor = 'Minor:' + str(round(zggj_info.gjmin, 2)) + 'cm'
            text_to_show.append(minor)

        if zggj_info.gjx_anno:
            cls.draw_polyline_annotation(painter, zggj_info.gjx_anno)
            gjx = "CX: " + str(round(zggj_info.gjx_len, 2)) + 'cm'
            text_to_show.append(gjx)

        # 增加实际长度在图上
        if text_to_show:
            font_size = cls.get_font_size()
            x, y = cls.get_info_offset()
            cls.draw_text(painter, text_to_show, x, y, font_size)

    @classmethod
    def draw_lc_measure_info(cls, painter, lc_info):
        if lc_info is None:
            return

        # 绘制直线
        text_to_show = []
        if lc_info.lc_bidiameter:
            lc = lc_info.lc_bidiameter
            cls.draw_bidiameter(painter, lc, label='ov', set_color_1st=QColor(0, 0, 255),
                                set_color_2nd=QColor(255, 255, 0), cross_scale=4)
            text_to_show.append(f'OV: {lc.major_len:.2f} cm X {lc.minor_len:.2f} cm')

        if not lc_info.lp_anno_list:
            return

        font_size = cls.get_font_size()
        font = QFont('Arial', font_size)
        painter.setFont(font)

        color_1st = QColor(0, 0, 255)
        color_2nd = QColor(255, 255, 0)
        for i, lp in enumerate(lc_info.lp_anno_list):
            label = f'#{i + 1}'
            if i == 0:
                color_1st = QColor(255, 0, 0)
                color_2nd = QColor(255, 125, 0)
            else:
                color_1st = QColor(125, 125, 125)
                color_2nd = QColor(125, 125, 125)
            # 绘制直线
            cls.draw_bidiameter(painter, lp, label=label, font_size=12, set_color_1st=color_1st,
                                set_color_2nd=color_2nd, cross_scale=4)

            # 增加实际长度在图上
            if lp.major_len > 2.5 or lp.minor_len > 2.5:
                text_to_show.append(f'cyst-{label}: {lp.major_len:.2f} cm X {lp.minor_len:.2f} cm')
            else:
                text_to_show.append(f'{label}: {lp.major_len:.2f} cm X {lp.minor_len:.2f} cm')

        x, y = cls.get_info_offset()
        cls.draw_text(painter, text_to_show, x, y, font_size)

    @classmethod
    def draw_nodule_measure_info(cls, painter, nodule_measure_info):
        nodule_anno_list = nodule_measure_info.nodule_anno_list
        if not nodule_anno_list:
            return

        font_size = cls.get_font_size()
        font = QFont('Arial', font_size)
        painter.setFont(font)

        text_to_show = []
        for i, nodule in enumerate(nodule_anno_list):
            label = nodule.get_name() if nodule.name else f'#{i + 1}'
            # 绘制直线
            cls.draw_bidiameter(painter, nodule, label=label)

            # 增加实际长度在图上
            text_to_show.append(f'{label}: {nodule.major_len:.2f} cm X {nodule.minor_len:.2f} cm')
            if nodule.custom_props and 'cls_props' in nodule.custom_props:
                prop_list = []
                for prop in nodule.custom_props['cls_props'].values():
                    prop_list.append(str(prop))
                text_to_show.append(f'{label}: {"; ".join(prop_list)}')

        x, y = cls.get_info_offset()
        cls.draw_text(painter, text_to_show, x, y, font_size)

    @classmethod
    def draw_bidiameter(cls, painter, bidiameter, label='', font_size=16, set_color_1st=QColor(0, 0, 255),
                        set_color_2nd=QColor(255, 255, 0), cross_scale=8):
        if not bidiameter:
            return

        cls.draw_line_annotation(painter, bidiameter.major_axis, cross_scale=cross_scale, color=set_color_1st)
        cls.draw_line_annotation(painter, bidiameter.minor_axis, cross_scale=cross_scale, color=set_color_2nd)

        if label and (bidiameter.major_axis or bidiameter.minor_axis):
            x, y = bidiameter.min_y_point()

            # 设置Qpainter
            pen = QPen(QColor(252, 243, 81))
            painter.setPen(pen)
            font = QFont('Arial', font_size)
            painter.setFont(font)

            painter.drawText(int(x), int(y - font_size), label)

    @classmethod
    def draw_jizhu_measure_info(cls, painter, jizhu_info):
        if jizhu_info is None:
            return

        font_size = cls.get_font_size()
        if jizhu_info.vetebral_list:
            for i, veteb in enumerate(jizhu_info.vetebral_list):
                cls.draw_ellipse_annotation(painter, veteb)

                # center
                x, y = veteb.center_point()
                cls.draw_text(painter, str(i + 1), int(x - font_size / 2),
                              int(y + font_size + 10), font_size=font_size)

        text_to_show = []
        if jizhu_info.cone2skin_anno:
            cls.draw_line_annotation(painter, jizhu_info.cone2skin_anno)
            text_to_show.append('cone2skin: ' + str(round(jizhu_info.cone2skin_len, 2)) + ' cm')
        if jizhu_info.cone2end_anno:
            cls.draw_line_annotation(painter, jizhu_info.cone2end_anno)
            text_to_show.append('cone2end: ' + str(round(jizhu_info.cone2end_len, 2)) + ' cm')

        x, y = cls.get_info_offset()
        cls.draw_text(painter, text_to_show, x, y, font_size=font_size)

    @classmethod
    def draw_heart_measure_info(cls, painter, heart_info: HeartMeasureInfo):
        text_to_show = []

        # 心轴角度
        if heart_info.ca > 0:
            text_to_show.append('ca: ' + str(round(heart_info.ca)) + ' deg')

        if heart_info.heart_bidiameter_anno:
            cls.draw_bidiameter(painter, heart_info.heart_bidiameter_anno)
            text_to_show.append('tcd: ' + str(round(heart_info.tcd, 2)) + ' cm')

        if heart_info.thorax_bidiameter_anno:
            cls.draw_bidiameter(painter, heart_info.thorax_bidiameter_anno)
            text_to_show.append('ttd: ' + str(round(heart_info.ttd, 2)) + ' cm')

        # 左心室的左右横径、上下横径
        LV_anno = heart_info.LV_bidiameter_anno
        if LV_anno:
            cls.draw_bidiameter(painter, LV_anno)
            text_to_show.append(f'LV: {LV_anno.major_len:.2f}x{LV_anno.minor_len:.2f} cm')

        # 右心室的左右横径、上下横径
        RV_anno = heart_info.RV_bidiameter_anno
        if RV_anno:
            cls.draw_bidiameter(painter, RV_anno)
            text_to_show.append(f'RV: {RV_anno.major_len:.2f}x{RV_anno.minor_len:.2f} cm')

        # 左心房的左右横径、上下横径
        LA_anno = heart_info.LA_bidiameter_anno
        if LA_anno:
            cls.draw_bidiameter(painter, LA_anno)
            text_to_show.append(f'LA: {LA_anno.major_len:.2f}x{LA_anno.minor_len:.2f} cm')

        # 右心房的左右横径、上下横径
        RA_anno = heart_info.RA_bidiameter_anno
        if RA_anno:
            cls.draw_bidiameter(painter, RA_anno)
            text_to_show.append(f'RA: {RA_anno.major_len:.2f}x{RA_anno.minor_len:.2f} cm')

        # 室间隔
        if heart_info.IVS_line_anno:
            cls.draw_line_annotation(painter, heart_info.IVS_line_anno)
            text_to_show.append('IVS: ' + str(round(heart_info.IVS_line, 2)) + ' cm')
        # 左室壁
        if heart_info.LVW_line_anno:
            cls.draw_line_annotation(painter, heart_info.LVW_line_anno)
            text_to_show.append('LVW: ' + str(round(heart_info.LVW_line, 2)) + ' cm')
        # 右室壁
        if heart_info.RVW_line_anno:
            cls.draw_line_annotation(painter, heart_info.RVW_line_anno)
            text_to_show.append('RVW: ' + str(round(heart_info.RVW_line, 2)) + ' cm')
        # 降主动脉和左心房的距离
        if heart_info.DA_LA_anno:
            cls.draw_line_annotation(painter, heart_info.DA_LA_anno)
            text_to_show.append('DA_LA: ' + str(round(heart_info.DA_LA, 2)) + ' cm')
        # 降主动脉内径
        if heart_info.DA_r_anno:
            cls.draw_line_annotation(painter, heart_info.DA_r_anno)
            text_to_show.append('DA_r: ' + str(round(heart_info.DA_r, 2)) + ' cm')
        # 胸腔面积
        if heart_info.thorax_ellipse:
            # cls.draw_line_annotation(painter, heart_info.thorax_ellipse)
            text_to_show.append('Thorax area: ' + str(round(heart_info.thorax_area, 2)) + ' cm^2')
        # 心脏面积
        if heart_info.heart_ellipse:
            # cls.draw_line_annotation(painter, heart_info.heart_ellipse)
            text_to_show.append('Heart area: ' + str(round(heart_info.heart_area, 2)) + ' cm^2')

        # 心胸面积比
        if heart_info.thorax_ellipse and heart_info.heart_ellipse:
            if heart_info.thorax_area != 0:
                heart_thorax = heart_info.heart_area / heart_info.thorax_area
                text_to_show.append('Heart/Thorax: ' + str(round(heart_thorax, 2)))

        x, y = cls.get_info_offset()
        font_size = cls.get_font_size()
        cls.draw_text(painter, text_to_show, x, y, font_size=font_size)

    @classmethod
    def draw_earlygyn_measure_info(cls, painter, early_info):
        if early_info is None:
            return

        text_to_show = []
        # 绘制直线
        if early_info.rsn_bidiameter:
            rsn = early_info.rsn_bidiameter
            cls.draw_bidiameter(painter, rsn)

            info = f'RSN: {rsn.major_len:.2f} cm X {rsn.minor_len:.2f} cm'
            text_to_show.append(info)

        if early_info.lhn_anno:
            cls.draw_line_annotation(painter, early_info.lhn_anno)
            lhn = 'LHN: ' + str(round(early_info.lhn, 2)) + 'cm'

            text_to_show.append(lhn)

        if early_info.pr_anno:
            cls.draw_line_annotation(painter, early_info.pr_anno)
            pr = 'PR: ' + str(round(early_info.pr, 2)) + 'cm'

            text_to_show.append(pr)

        # 增加实际长度在图上
        if text_to_show:
            font_size = cls.get_font_size()
            x, y = cls.get_info_offset()
            cls.draw_text(painter, text_to_show, x, y, font_size)

    @classmethod
    def draw_earlier_measure_info(cls, painter: QPainter, info: FukeEarlierMeasureInfo):
        text_list = []
        if info.earlier_maj_anno:
            cls.draw_line_annotation(painter, info.earlier_maj_anno)
            text_list.append(f'D1: {round(info.earlier_maj, 2)} cm')
        if info.plane_type == '妊娠囊' and info.earlier_min_anno:
            cls.draw_line_annotation(painter, info.earlier_min_anno)
            text_list.append(f'D2: {round(info.earlier_min, 2)} cm')

        x, y = cls.get_info_offset()
        font_size = cls.get_font_size()

        x, y = cls.draw_text(painter, text_list, x, y, font_size=font_size)
        return x, y

    @classmethod
    def draw_spectrum_measure_info(cls, painter, spectrum_info: SpectrumMeasureInfo):
        if spectrum_info.envelop:
            cls.draw_envelop_annotation(painter, spectrum_info.envelop, dashed_line=False,
                                        down_dir_flag=spectrum_info.above_flag)

        for line_anno in spectrum_info.hr_lines:
            cls.draw_line_annotation(painter, line_anno, cross_scale=-1, color=QColor(192, 192, 192))

        text_list = []
        for key, param in spectrum_info.parameters.items():
            if key in ['PSV', 'EDV']:
                text = f'{key}: {param:.2f} cm/s'
            elif key in ['HR']:
                text = f'HR: {int(param)}  bpm'
            elif key in ['SD']:
                text = f'S/D: {param:.2f}'
            else:
                text = f'{key}: {param:.2f}'
            text_list.append(text)

        x, y = cls.get_info_offset(spectrum=True)
        font_size = cls.get_font_size()
        cls.draw_text(painter, text_list, x, y, font_size)

    @classmethod
    def draw_ruler_info(cls, painter, ruler_info):
        if not ruler_info:
            return

        pen = QPen(QColor(255, 0, 0))
        painter.setPen(pen)

        # convert to list
        ruler_info_list = ruler_info
        if not isinstance(ruler_info, list):
            ruler_info_list = [ruler_info]

        for ruler_info in ruler_info_list:
            x0, y0 = ruler_info['startX'], ruler_info['startY']
            x1, y1 = ruler_info['endX'], ruler_info['endY']

            painter.drawLine(QPoint(x0, y0), QPoint(x0 + 20, y0))

            tick_count = ruler_info['count']
            if tick_count > 0:
                y_delta = (y1 - y0) / tick_count
                x_delta = (x1 - x0) / tick_count

                x, y = x0, y0
                for i in range(tick_count - 1):
                    x += x_delta
                    y += y_delta
                    painter.drawLine(QPoint(int(x), int(y)), QPoint(int(x + 20), int(y)))

            painter.drawLine(QPoint(x1, y1), QPoint(x1 + 20, y1))

    def window_to_image_coord(self, x, y):
        x = (x - self.x_offset) / self.scale
        y = (y - self.y_offset) / self.scale

        return x, y

    def set_image_width(self, image_width):
        self.image_width = image_width

    def wheelEvent(self, event):
        # ctrl + wheel: do scale
        if event.modifiers() == Qt.ControlModifier:
            if event.angleDelta().y() > 0:
                self.scale *= 1.1
            else:
                self.scale *= 0.9

            self.update()

            event.accept()
        else:
            super().wheelEvent(event)

    def resizeEvent(self, event) -> None:
        if self.auto_to_fit:
            self.scale_to_fit()
            self.update()
        return super().resizeEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_R:
            self.scale = 1.0
            self.translate = [0, 0]
            self.update()
        elif event.key() == Qt.Key_F:
            if self.auto_to_fit:
                self.scale_to_fit(False)
                self.update()
        else:
            super().keyPressEvent(event)
