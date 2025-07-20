# import time
import numpy as np

from .MMDetModel import MMDetModel


class SOLOV2Model(MMDetModel):
    def __init__(self, model_file_name, class_mapping_file, config, load_model=True,
                 gpu_id=0, model_dir=r'/data/QC_python/model/'):
        self.config = config
        # 显示阈值
        # self.result_score_thr = 0.2
        super(SOLOV2Model, self).__init__(model_file_name, class_mapping_file, config,
                                          load_model, gpu_id, model_dir)

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
                type_name = self.get_annotation_name(type_id)
                score = bbox[4]
                # if type_id != 18 and score <= self.config["score_threshold"]:
                # 避免删除侧脑室的分割影响测量
                if type_name != "侧脑室后角" and score <= self.config["score_threshold"]:
                    continue

                mask = seg.astype(np.uint8)

                rows, cols = np.where(mask == 1)

                # Calculate the top, left, bottom, and right boundaries
                top = min(rows)
                left = min(cols)
                bottom = max(rows)
                right = max(cols)

                mask_info_list.append({
                    'mask': mask,
                    'box': [left, top, right, bottom],
                    'score': score,
                    'polygon': None
                })

            # print(type_name)
            if mask_info_list:
                type_name = self.get_annotation_name(type_id)
                type2mask[type_name] = mask_info_list

        return type2mask
