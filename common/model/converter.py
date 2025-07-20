from .CenaoMeasureInfo import CenaoMeasureInfo
from .XiaonaoMeasureInfo import XiaonaoMeasureInfo
from .AFIMeasureInfo import AFIMeasureInfo
from .CRLMeasureInfo import CRLMeasureInfo
from .HLMeasureInfo import HLMeasureInfo
from .NtMeasureInfo import NtMeasureInfo
from .PLMeasureInfo import PLMeasureInfo
from .LineAnnotation import LineAnnotation
from .BoxAnnotation import BoxAnnotation
from .EllipseAnnotation import EllipseAnnotation
from .PolylineAnnotation import PolylineAnnotation
from .PolygonAnnotation import PolygonAnnotation

from .FLMeasureInfo import FLMeasureInfo
from .AcMeasureInfo import AcMeasureInfo
from .HcMeasureInfo import HcMeasureInfo
from .ZGGJMeasureInfo import ZGGJMeasureInfo
from .ZGHMeasureInfo import ZGHMeasureInfo
from .ZGZMeasureInfo import ZGZMeasureInfo
from .LCMeasureInfo import LCMeasureInfo
from .NoduleMeasureInfo import NoduleMeasureInfo
# from .ZGJLMeasureInfo import ZGJLMeasureInfo
from .FukeEarlierMeasureInfo import FukeEarlierMeasureInfo

from .JizhuMeasureInfo import JizhuMeasureInfo
from .measure_info import MeasureInfo

from .ThyroidMeasureInfo import ThyroidMeasureInfo
from .HeartMeasureInfo import HeartMeasureInfo
from .SpectrumMeasureInfo import SpectrumMeasureInfo

from loguru import logger


def annotation_from_json(json_info):
    type = json_info['type'] if 'type' in json_info else 'rect'

    anno = None
    if type in ['rect', 'rectangle'] or type == 2:
        anno = BoxAnnotation.from_json(json_info)
    elif type == 'line' or type == 1:
        anno = LineAnnotation.from_json(json_info)
    elif type == 'ellipse' or type == 3:
        anno = EllipseAnnotation.from_json(json_info)
    elif type == 'polyline' or type == 8:
        anno = PolylineAnnotation.from_json(json_info)
    elif type == 'polygon' or type == 4:
        anno = PolygonAnnotation.from_json(json_info)

    return anno


type2handler = {
    'fl': FLMeasureInfo.from_json,
    'ac': AcMeasureInfo.from_json,
    'hc': HcMeasureInfo.from_json,
    'hl': HLMeasureInfo.from_json,
    'nt': NtMeasureInfo.from_json,
    'crl': CRLMeasureInfo.from_json,
    'afi': AFIMeasureInfo.from_json,
    'pl': PLMeasureInfo.from_json,
    'lv': CenaoMeasureInfo.from_json,
    'tc': XiaonaoMeasureInfo.from_json,
    'zgh': ZGHMeasureInfo.from_json,
    'zgz': ZGZMeasureInfo.from_json,
    'zggj': ZGGJMeasureInfo.from_json,
    'lc': LCMeasureInfo.from_json,
    'nodule': NoduleMeasureInfo.from_json,
    'earlier': FukeEarlierMeasureInfo.from_json,
    'veteb': JizhuMeasureInfo.from_json,
    'thyroid': ThyroidMeasureInfo.from_json,
    'heart': HeartMeasureInfo.from_json,
    'spectrum': SpectrumMeasureInfo.from_json
}


def measure_info_from_json(json_info):
    if not json_info:
        return

    type = json_info['type']
    if type in type2handler:
        info = type2handler[type](json_info)
    else:
        logger.warning(f'measure type is not defined: {type}')
        info = MeasureInfo.from_json(json_info)

    return info
