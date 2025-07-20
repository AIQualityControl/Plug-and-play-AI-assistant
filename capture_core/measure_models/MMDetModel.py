# import time
import numpy as np

from .MeasureModel import MeasureModel

from mmdet.apis.inference import init_detector, inference_detector


class MMDetModel(MeasureModel):
    def __init__(self, model_file_name, class_mapping_file, config, load_model=True,
                 gpu_id=0, model_dir=r'/data/QC_python/model/'):
        self.config = config
        # 显示阈值
        # self.result_score_thr = 0.2
        super(MMDetModel, self).__init__(model_file_name, class_mapping_file, config,
                                         load_model, gpu_id, model_dir)

    def load_model(self, model_path, gpu_id, backbone_name):

        cfg_path = self.get_cfg_path()
        # cfg_path = os.path.join(os.path.dirname(sys.argv[0]), 'capture_core', self.config['config_path'])
        device = 'cuda:' + str(gpu_id)
        self.model = init_detector(cfg_path, model_path, device=device)

        return self.model

    def do_segment(self, image_list, image_info_list):

        if self.model is None:
            return

        predictions_list = []
        for image in image_list:
            predict_result = inference_detector(self.model, image)
            type2mask = self.convert_predict_result_to_mask(image, predict_result)
            predictions_list.append(type2mask)

        return predictions_list

    def convert_predict_result_to_mask(self, image, predict_result):

        if isinstance(predict_result, tuple):
            bbox_result, segm_result = predict_result
        else:
            # no segmentation result
            return

        type2mask = {}
        for type_id, bbox_list in enumerate(bbox_result):
            if bbox_list is None or len(bbox_list) == 0:
                continue

            mask_info_list = []
            for bbox, seg in zip(bbox_list, segm_result[type_id]):
                score = bbox[4]
                if score <= self.config["score_threshold"]:
                    continue

                mask = seg.astype(np.uint8)
                mask_info_list.append({
                    'mask': mask,
                    'box': [int(x) for x in bbox[0:4]],
                    'score': score,
                    'polygon': None
                })

            type_name = self.get_annotation_name(type_id)
            type2mask[type_name] = mask_info_list

        return type2mask
