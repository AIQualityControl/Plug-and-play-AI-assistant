import platform
import os
import sys
import ctypes
from loguru import logger
import numpy as np
import multiprocessing

from common.config.config import Config
from common.model.SpectrumMeasureInfo import SpectrumMeasureInfo
from common.model.image_info import ImageInfo


class SingleRulerInfo(ctypes.Structure):
    _fields_ = [("rulerUnit", ctypes.c_double), ("startX", ctypes.c_int), ("startY", ctypes.c_int),
                ("endX", ctypes.c_int), ("endY", ctypes.c_int), ("count", ctypes.c_int), ("roi", ctypes.c_int * 4)]


class MultipleRulerInfo(ctypes.Structure):
    _fields_ = [("rulers", SingleRulerInfo * 4)]


class SpectrumInfo(ctypes.Structure):
    _fields_ = [("PSV", ctypes.c_double), ("EDV", ctypes.c_double), ("SD", ctypes.c_double), ("RI", ctypes.c_double),
                ("PI", ctypes.c_double), ("HR", ctypes.c_short), ("above_flag", ctypes.c_bool),
                ("envelop", ctypes.c_short * 1920), ("num_envelop_points", ctypes.c_short),
                ("offsetX", ctypes.c_short),
                ("key_points", ctypes.c_short * 40), ("num_keypoints", ctypes.c_short),
                ("num_valley_points", ctypes.c_short),
                ("hr_xrange", ctypes.c_short * 2), ("hr_yrange", ctypes.c_short * 2),
                ("speed_unit", ctypes.c_double), ("time_unit", ctypes.c_double), ("x_axis", ctypes.c_short)
                ]


class SectorInfo(ctypes.Structure):
    _fields_ = [("leftUpX", ctypes.c_int), ("leftUpY", ctypes.c_int),
                ("leftBottomX", ctypes.c_int), ("leftBottomY", ctypes.c_int),
                ("rightUpX", ctypes.c_int), ("rightUpY", ctypes.c_int),
                ("rightBottomX", ctypes.c_int), ("rightBottomY", ctypes.c_int)]


class RoiInfo(ctypes.Structure):
    _fields_ = [("x", ctypes.c_int), ("y", ctypes.c_int),
                ("width", ctypes.c_int), ("height", ctypes.c_int)]


class RulerRecognizer:
    # load dll for ruler detection
    dll_name = 'ruler_recognizer.so'
    if platform.system().lower() == 'windows':
        dll_name = 'ruler_recognizer.dll'

    # dll_path = os.path.join(os.path.dirname(__file__), dll_name)
    config_path = os.path.join(os.getcwd(), 'capture_core', 'model_config')
    dll_path = os.path.join(config_path, dll_name)
    ruler_config_path = os.path.join(config_path, 'ruler', 'ruler_config.ini')

    process_name = multiprocessing.current_process().name
    logger.debug(f'{process_name}-{os.getpid()}: {dll_path}')

    try:
        rulerDll = ctypes.cdll.LoadLibrary(dll_path)

        detect_ruler_func = rulerDll.detect_ruler
        # detect_ruler_func.restype = ctypes.c_double
        detect_ruler_func.restype = (SingleRulerInfo)

        detect_roibox_func = rulerDll.detect_roi_box
        detect_roibox_func.restype = ctypes.c_bool

        detect_sampling_line_func = rulerDll.detect_sampling_line
        detect_sampling_line_func.restype = ctypes.c_bool

        check_ruler_point_func = rulerDll.check_ruler_point
        check_ruler_point_func.restype = ctypes.c_bool

        detect_spectrum_func = rulerDll.detect_spectrum
        detect_spectrum_func.restype = (SpectrumInfo)

        detect_sector_func = rulerDll.detect_sector
        detect_sector_func.restype = (SectorInfo)

        redetect_spectrum_func = rulerDll.redetect_spectrum
        redetect_spectrum_func.restype = ctypes.c_bool
        redetect_spectrum_func.argtypes = (ctypes.POINTER(SpectrumInfo),)

        init_config_func = rulerDll.init_config
        init_config_func.restype = ctypes.c_bool
        set_ruler_type_func = rulerDll.set_ruler_type

        is_dopler_func = rulerDll.is_dopler
        is_dopler_func.restype = ctypes.c_bool

        dopler_pixels_func = rulerDll.dopler_pixels
        dopler_pixels_func.restype = ctypes.c_int

        is_pseudo_color_image_func = rulerDll.is_pseudo_color_image
        is_pseudo_color_image_func.restype = ctypes.c_bool

        detect_roi_region_func = rulerDll.detect_roi_region
        detect_roi_region_func.restype = (RoiInfo)
    except Exception as e:
        rulerDll = None

        detect_ruler_func = None
        detect_afi_ruler_func = None

        detect_roibox_func = None
        detect_sampling_line_func = None

        check_ruler_point_func = None

        detect_spectrum_func = None
        redetect_spectrum_func = None

        init_config_func = None
        set_ruler_type_func = None

        is_dopler_func = None
        dopler_pixels_func = None
        is_pseudo_color_image_func = None

        detect_roi_region_func = None

        logger.error(str(e))
        # logger.error('Failed to load ruler recognizer dll: ' + dll_path)

    detect_afi_ruler_func = None
    if rulerDll is not None:
        try:
            detect_afi_ruler_func = rulerDll.detect_afi_ruler
            detect_afi_ruler_func.restype = (MultipleRulerInfo)
            # detect_afi_ruler_func.argtypes = []
        except Exception as e:
            logger.warning(str(e))

    # init ruler configuration
    if init_config_func and not init_config_func(ruler_config_path.encode('gbk')):
        logger.error('Failed to init ruler config: ', ruler_config_path)

    def __init__(self):
        '''constructor'''
        pass

    @classmethod
    def set_ruler_type(cls, type: str):
        if RulerRecognizer.set_ruler_type_func:
            RulerRecognizer.set_ruler_type_func(type.encode('utf-8'))

    @classmethod
    @logger.catch
    def detect_ruler(cls, origin_image, roi=None, sm_config: Config = None) -> SingleRulerInfo:
        """
        detect ruler via c-dll.
        gray_image can be color image, the function will convert color image to gray image automatically
        should only use color image if roi is None!!!
        """
        logger.debug("start detecting ruler")
        if RulerRecognizer.detect_ruler_func is None:
            logger.info("end detecting ruler, cause: func is None")
            return None

        if roi is None:
            roi = ImageInfo.roi

        roi_c = (ctypes.c_int * len(roi))(*roi)

        img_gray = origin_image.flatten()
        dataptr = img_gray.ctypes.data_as(ctypes.c_char_p)

        height, width = origin_image.shape[:2]
        channels = origin_image.shape[2] if len(origin_image.shape) > 2 else 1
        try:
            ruler_info = RulerRecognizer.detect_ruler_func(dataptr, width, height, channels, roi_c)
        except Exception as e:
            logger.error(f'failed to detect ruler: {str(e)}')
            return

        logger.debug("ruler detected")

        # update roi in ImageInfo
        new_roi = [ruler_info.roi[0], ruler_info.roi[1], ruler_info.roi[2], ruler_info.roi[3]]
        if new_roi != roi and sm_config:
            logger.info("attempting to change roi from {} to {} in detect ruler function".format(roi, new_roi))
            sm_config.set_detection_roi(new_roi)

        return ruler_info

    @classmethod
    @logger.catch
    def detect_spectrum(cls, image_info: ImageInfo, convert_envelop=True) -> SpectrumMeasureInfo:
        logger.debug("start detecting spectrum")
        if RulerRecognizer.detect_spectrum_func is None:
            logger.info("end detecting spectrum, cause func is None")
            return None

        image = image_info.roi_image()
        img_flat = image.flatten()
        dataptr = img_flat.ctypes.data_as(ctypes.c_char_p)

        height, width = image.shape[:2]
        channels = image.shape[2] if len(image.shape) > 2 else 1
        try:
            spectrum_info = RulerRecognizer.detect_spectrum_func(dataptr, width, height, channels)
        except Exception as e:
            logger.error(f'Failed to detect spectrum: {str(e)}')
            return

        offset = image_info.offset()
        offset = [offset[0] + spectrum_info.offsetX, offset[1]]

        info = cls.convert_to_measure_info(spectrum_info, offset)
        if convert_envelop and info:
            info.convert_envelop()

        points_memory_size = sys.getsizeof(info.envelop.points)
        if points_memory_size > 4096:
            # 4KB at most
            sample_rate = points_memory_size / 4096
            info.envelop.key_points_idx = [int(key_point_idx / sample_rate) for key_point_idx in
                                           info.envelop.key_points_idx]
            info.envelop.points = np.array(info.envelop.points)[
                np.arange(0, len(info.envelop.points), sample_rate).astype(np.int32)].tolist()

        logger.debug("spectrum detected")
        return info

    @classmethod
    def convert_to_measure_info(cls, spectrum_info, offset):
        envelop = spectrum_info.envelop[:spectrum_info.num_envelop_points]

        parameters = {
            "PSV": spectrum_info.PSV,
            "EDV": spectrum_info.EDV,
            "SD": spectrum_info.SD,
            "RI": spectrum_info.RI,
            "PI": spectrum_info.PI,
            "HR": spectrum_info.HR,
        }

        # peak and valley points
        key_points_idx = spectrum_info.key_points[:spectrum_info.num_keypoints]

        HR_lines = [[[spectrum_info.hr_xrange[0] + offset[0], spectrum_info.hr_yrange[0] + offset[1]],
                     [spectrum_info.hr_xrange[0] + offset[0], spectrum_info.hr_yrange[1] + offset[1]]],
                    [[spectrum_info.hr_xrange[1] + offset[0], spectrum_info.hr_yrange[0] + offset[1]],
                     [spectrum_info.hr_xrange[1] + offset[0], spectrum_info.hr_yrange[1] + offset[1]]]]

        info = SpectrumMeasureInfo(parameters, (envelop, key_points_idx, offset), spectrum_info.num_valley_points,
                                   spectrum_info.above_flag, HR_lines,
                                   spectrum_info.speed_unit, spectrum_info.time_unit, spectrum_info.x_axis)

        return info

    @classmethod
    def redetect_spectrum(cls, measure_info: SpectrumMeasureInfo):
        if RulerRecognizer.redetect_spectrum_func is None:
            return None

        # convert from SpectrumMeasureInfo to SpectrumInfo
        info = SpectrumInfo()
        info.num_envelop_points = measure_info.envelop.num_of_points()
        if info.num_envelop_points > 0:
            info.offsetX = int(measure_info.envelop.points[0][0])
            for pt in measure_info.envelop.points:
                info.envelop[int(pt[0] - info.offsetX)] = int(pt[1])

        info.num_keypoints = measure_info.envelop.num_of_keypoints()
        for i, idx in enumerate(measure_info.envelop.key_points_idx):
            info.key_points[i] = idx

        info.num_valley_points = measure_info.num_valley_points
        info.speed_unit = measure_info.ruler_unit
        info.time_unit = measure_info.time_unit
        info.x_axis = measure_info.x_axis

        info.hr_xrange[0] = int(round(measure_info.hr_lines[0].start_point()[0] - info.offsetX))
        info.hr_xrange[1] = int(round(measure_info.hr_lines[1].start_point()[0] - info.offsetX))
        info.hr_yrange[0] = int(measure_info.hr_lines[0].start_point()[1])
        info.hr_yrange[1] = int(measure_info.hr_lines[0].end_point()[1])

        # print(info.hr_xrange[0], info.hr_xrange[1])

        info.above_flag = measure_info.above_flag
        try:
            if not RulerRecognizer.redetect_spectrum_func(info):
                logger.error('Failed to redetect spectrum')
                return
        except Exception as e:
            logger.error(f'Failed to redetect spectrum: {str(e)}')
            return

        # update measure info: envelop points keeps unchanged
        measure_info.parameters = {
            "PSV": info.PSV,
            "EDV": info.EDV,
            "SD": info.SD,
            "RI": info.RI,
            "PI": info.PI,
            "HR": info.HR,
        }

        # peak and valley points
        measure_info.envelop.key_points_idx = info.key_points[:info.num_keypoints]

        pt = measure_info.hr_lines[0].start_point()
        pt[0] = info.hr_xrange[0] + info.offsetX
        measure_info.hr_lines[0].end_point()[0] = pt[0]

        pt = measure_info.hr_lines[1].start_point()
        pt[0] = info.hr_xrange[1] + info.offsetX
        measure_info.hr_lines[1].end_point()[0] = pt[0]

    @classmethod
    @logger.catch
    def detect_afi_ruler(cls, origin_image, afi_lines):
        """
        detect ruler for afi planes
        afi_lines: list of end_points of afi lines, each line has 4 int
        origin_image: can be color image or gray image
        """
        logger.debug("start detecting afi ruler")
        if RulerRecognizer.detect_afi_ruler_func is None:
            ruler_info = RulerRecognizer.detect_ruler(origin_image)
            return [ruler_info]

        img_data = origin_image.flatten()
        dataptr = img_data.ctypes.data_as(ctypes.c_char_p)

        # end points
        if afi_lines:
            points_len = len(afi_lines)

            end_points = ctypes.c_int32 * points_len
            end_points = end_points()
            for i in range(points_len):
                end_points[i] = int(afi_lines[i])
        else:
            # for case without input afi lines
            points_len = 2
            end_points = ctypes.c_int32 * points_len
            end_points = end_points()
            end_points[0] = 100
            end_points[1] = 100

        # machine_type = cls.machine_type.encode('gbk')
        height, width = origin_image.shape[:2]
        channels = origin_image.shape[2] if len(origin_image.shape) > 2 else 1
        try:
            ruler_info = RulerRecognizer.detect_afi_ruler_func(dataptr, width, height, channels,
                                                               end_points, points_len)
            if ruler_info:
                ruler_info_list = [ruler for ruler in ruler_info.rulers if ruler.rulerUnit != -1]
                logger.debug("afi ruler detected")
                return ruler_info_list
        except Exception as e:
            logger.error(f'failed to detect afi ruler: {str(e)}')
            return

    @classmethod
    def detect_roi_box(cls, image):
        """
        whether has roi box in image, the image should be color image
        """
        if RulerRecognizer.detect_roibox_func is None:
            return False, False

        img_flat = image.flatten()
        dataptr = img_flat.ctypes.data_as(ctypes.c_char_p)

        height, width = image.shape[:2]
        channels = image.shape[2] if len(image.shape) > 2 else 1
        try:
            ret = RulerRecognizer.detect_roibox_func(dataptr, width, height, channels)
        except Exception as e:
            logger.error(f'failed to detect roi box: {str(e)}')
            return False

        return ret

    @classmethod
    def detect_roi_region(cls, image):
        """
        detect roi region in image, the image should be color image
        return: roi in xywh
        """
        if RulerRecognizer.detect_roi_region_func is None:
            return False

        img_flat = image.flatten()
        dataptr = img_flat.ctypes.data_as(ctypes.c_char_p)

        height, width = image.shape[:2]
        channels = image.shape[2] if len(image.shape) > 2 else 1
        try:
            ret = RulerRecognizer.detect_roi_region_func(dataptr, width, height, channels)
        except Exception as e:
            logger.error(f'failed to detect roi box: {str(e)}')
            return False

        return [ret.x, ret.y, ret.width, ret.height]

    @classmethod
    def detect_sampling_line(cls, image):
        """
        whether has sampling line in image, the image should be color image
        """
        if RulerRecognizer.detect_sampling_line_func is None:
            return False

        img_flat = image.flatten()
        dataptr = img_flat.ctypes.data_as(ctypes.c_char_p)

        height, width = image.shape[:2]
        channels = image.shape[2] if len(image.shape) > 2 else 1
        ret = RulerRecognizer.detect_sampling_line_func(dataptr, width, height, channels)
        return ret

    @classmethod
    def is_pseudo_color_image(cls, image):
        """
        return:
        """
        if RulerRecognizer.is_pseudo_color_image_func is None:
            return False

        img_flat = image.flatten()
        dataptr = img_flat.ctypes.data_as(ctypes.c_char_p)

        height, width = image.shape[:2]
        channels = image.shape[2] if len(image.shape) > 2 else 1

        return RulerRecognizer.is_pseudo_color_image_func(dataptr, width, height, channels)

    @classmethod
    def is_dopler(cls, image):
        """
        return: (is_dopler, is_pseudo_color)
        """
        if RulerRecognizer.is_dopler_func is None:
            return False, False

        img_flat = image.flatten()
        dataptr = img_flat.ctypes.data_as(ctypes.c_char_p)

        height, width = image.shape[:2]
        channels = image.shape[2] if len(image.shape) > 2 else 1

        is_pseudo_color = ctypes.c_bool(False)
        ret = RulerRecognizer.is_dopler_func(dataptr, width, height, channels, ctypes.byref(is_pseudo_color))

        pseudo_color = bool(is_pseudo_color)
        return ret, pseudo_color

    @classmethod
    def dopler_pixels(cls, image, is_pseudo_color, show_image=False):
        if RulerRecognizer.dopler_pixels_func is None:
            return 0

        img_flat = image.flatten()
        dataptr = img_flat.ctypes.data_as(ctypes.c_char_p)

        height, width = image.shape[:2]
        channels = image.shape[2] if len(image.shape) > 2 else 1
        ret = RulerRecognizer.dopler_pixels_func(dataptr, width, height, channels, is_pseudo_color, show_image)
        return ret

    @classmethod
    def snap_ruler_ticks(cls, image, point0, point1):
        """
        snap ruler ticks automatically
        point0, point1: in and out, should be list
        """
        if RulerRecognizer.check_ruler_point_func is None:
            return False

        # image
        img_flat = image.flatten()
        dataptr = img_flat.ctypes.data_as(ctypes.c_char_p)

        # points
        point_type = ctypes.c_int32 * 2
        pt0 = point_type()
        pt0[0] = int(point0[0])
        pt0[1] = int(point0[1])

        pt1 = point_type()
        pt1[0] = int(point1[0])
        pt1[1] = int(point1[1])

        # machine_type = cls.machine_type.encode('gbk')
        height, width = image.shape[:2]
        channels = image.shape[2] if len(image.shape) > 2 else 1
        ret = RulerRecognizer.check_ruler_point_func(dataptr, width, height, channels, pt0, pt1)

        point0[0] = pt0[0]
        point0[1] = pt0[1]
        point1[0] = pt1[0]
        point1[1] = pt1[1]

        # print(pt0[0], pt0[1], pt1[0], pt1[1])
        return ret

    @classmethod
    def detect_sector(cls, image_info: ImageInfo):
        if RulerRecognizer.detect_sector_func is None:
            return

        # image
        image = image_info.roi_image()
        img_flat = image.flatten()
        dataptr = img_flat.ctypes.data_as(ctypes.c_char_p)

        height, width = image.shape[:2]
        channels = image.shape[2] if len(image.shape) > 2 else 1
        sector_info = RulerRecognizer.detect_sector_func(dataptr, width, height, channels)

        offset = image_info.offset()
        if sector_info.leftUpY == sector_info.leftBottomY and sector_info.leftUpY == 0:
            left_line = None
        else:
            left_line = [[sector_info.leftUpX + offset[0], sector_info.leftUpY + offset[1]],
                         [sector_info.leftBottomX + offset[0], sector_info.leftBottomY + offset[1]]]

        if sector_info.rightUpY == sector_info.rightBottomY and sector_info.rightUpY == 0:
            right_line = None
        else:
            right_line = [[sector_info.rightUpX + offset[0], sector_info.rightUpY + offset[1]],
                          [sector_info.rightBottomX + offset[0], sector_info.rightBottomY + offset[1]]]

        if left_line or right_line:
            return [left_line, right_line]


if __name__ == '__main__':
    image = np.zeros((640, 640), np.uint8)

    # RulerRecognizer.machine_type = '三星'
    RulerRecognizer.detect_ruler(image)

    pt0 = [10, 20]
    pt1 = [30, 40]
    RulerRecognizer.snap_ruler_ticks(image, pt0, pt1)

    print(pt0)
    print(pt1)
