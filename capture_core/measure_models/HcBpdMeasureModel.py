from ..biometry_measure.EllipseFitting import EllipseFitting
from ..biometry_measure.BPDMeasure import BPDMeasure
from common.FetalBiometry import FetalBiometry
from common.model.EllipseAnnotation import EllipseAnnotation
from common.model.LineAnnotation import LineAnnotation
from common.model.HcMeasureInfo import HcMeasureInfo
from common.model.AcMeasureInfo import AcMeasureInfo
from .DDRNetModel import DDRNetModel


class HcBpdMeasureModel(DDRNetModel):
    def __init__(self, model_file_name, class_mapping_file, config, load_model=True,
                 gpu_id=0, model_dir=r'/data/QC_python/model/'):
        '''constructor'''
        super(HcBpdMeasureModel, self).__init__(model_file_name, class_mapping_file, config,
                                                load_model, gpu_id, model_dir)

    def do_measure(self, mask, roi_image, image_info):
        if mask is None:
            return self.default_measure(roi_image, image_info)

        # hc
        try:
            hc_info = EllipseFitting.fit_ellipse(mask)
        except Exception:
            hc_info = None

        if hc_info is None:
            return self.default_measure(roi_image, image_info)

        if self.plane_type == '上腹部水平横切面':
            return AcMeasureInfo(hc_info[0], hc_info[1], hc_info[2])

        hc_anno = EllipseAnnotation(hc_info[0], hc_info[1], hc_info[2])
        FetalBiometry.is_hc_plane_detected = True

        # bpd
        end_points = hc_anno.minor_radius_points()

        error_type = ''
        try:
            bpd_infos = BPDMeasure.do_measure(roi_image, mask, end_points, measure_mode=self.measure_mode)
            intergrowth_21st_bpd_anno = LineAnnotation(bpd_infos["intergrowth_21st"][0], bpd_infos["intergrowth_21st"][
                1]) if bpd_infos and "intergrowth_21st" in bpd_infos else None

            hadlock_bpd_anno = LineAnnotation(
                bpd_infos["hadlock"][0], bpd_infos["hadlock"][1]) if bpd_infos and "hadlock" in bpd_infos else None

        except Exception:
            bpd_infos = end_points
            error_type = 'HC error'

        # bpd_anno = LineAnnotation(bpd_info[0], bpd_info[1])

        measure_result = HcMeasureInfo(hc_anno, intergrowth_21st_bpd_anno, hadlock_bpd_anno)
        measure_result.error_type = error_type

        return measure_result

    def default_measure(self, roi_image, image_info):

        part_name = '腹壁' if self.plane_type == '上腹部水平横切面' else '颅骨光环'
        bbox = self.get_part_bbox(image_info, part_name)
        if bbox:
            pt_start, pt_end = bbox
        else:
            h, w = roi_image.shape[:2]
            # 缩小矩形的大小
            shrink_factor = 0.87
            w_shrink = w * shrink_factor
            h_shrink = h * shrink_factor

            pt_start = [(w - w_shrink) / 2, (h - h_shrink) / 2]
            pt_end = [(w + w_shrink) / 2, (h + h_shrink) / 2]

        if self.plane_type == '上腹部水平横切面':
            info = AcMeasureInfo(pt_start, pt_end, is_default_value=True)
            info.error_type = 'AC error'
            return info

        # bpd
        hc_anno = EllipseAnnotation(pt_start, pt_end, is_default_value=True)

        # bpd
        end_points = hc_anno.minor_radius_points()
        bpd_anno = LineAnnotation(end_points[0], end_points[1], is_default_value=True)

        FetalBiometry.is_hc_plane_detected = True

        info = HcMeasureInfo(hc_anno, bpd_anno)
        info.error_type = 'HC error'

        return info
