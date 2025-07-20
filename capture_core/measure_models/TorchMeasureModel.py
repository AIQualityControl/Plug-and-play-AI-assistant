#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@description:
@date       : 2022/04/30 14:08:56
@author     : Guanghua Tan
@email      : guanghuatan@hnu.edu.cn
@version    : 1.0
'''

from .MeasureModel import MeasureModel
from ..AnnotationIO.aes_cipher import AesCipher
import cv2
import numpy as np
import torch
import io


class TorchMeasureModel(MeasureModel):
    def __init__(self, model_file_name, class_mapping_file, config, load_model=True,
                 gpu_id=0, model_dir=r'/data/QC_python/model/'):
        '''constructor'''

        self.device = gpu_id
        self.padding_value = config['padding_value'] if 'padding_value' in config else 0

        super(TorchMeasureModel, self).__init__(model_file_name, class_mapping_file, config,
                                                load_model, gpu_id, model_dir)

    def load_model_weights(self, model, weights_path, to_cuda=True):
        # weights

        model_data = AesCipher.decipher_model(weights_path)
        if model_data is None:
            return

        bytes_io = io.BytesIO(model_data)

        # 为了macos可以在cpu上运行，加了下面三行
        import os
        if os.name == 'posix':
            pretrained_weights = torch.load(bytes_io, map_location=torch.device('cpu'))
        else:
            pretrained_weights = torch.load(bytes_io)

        # pretrained_weights = torch.load(weights_path)
        model.load_state_dict(pretrained_weights, strict=False)

        model = model.eval()

        # self.model = nn.DataParallel(self.model)
        if to_cuda:
            device = torch.device('cuda:' + str(self.device) if torch.cuda.is_available() else 'cpu')
            model = model.to(device)

        # bytes_io.close()
        return model

    def do_segment(self, image_list, image_info_list):
        if len(image_list) == 0:
            return

        t_w, t_h = self.config['target_width'], self.config['target_height']
        results = []
        for img in image_list:
            mask = self.do_segment_for_image(img, t_w, t_h)
            results.append(mask)
        return results

    def do_segment_for_image(self, img, t_w, t_h):
        """
        t_w: target_width -- == 0: resize according to height, >0: resize to target_width, padding if necessary
        t_h: target_height -- ==0: resize according to width, > 0: resize to target_height, padding if necessary
        """
        original_h, original_w = img.shape[:2]

        resized, roi = self.letterbox(img, (t_h, t_w), padding_value=self.padding_value)
        # convert to color RGB mode
        if len(resized.shape) < 3 or resized.shape[2] == 1:
            resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
        else:
            resized = resized[:, :, ::-1]

        # normalize to [0, 1]
        image_data = resized.astype(np.float32) / 255.0

        # add batch_size dim: batch first
        image_data = np.expand_dims(np.transpose(image_data, (2, 0, 1)), 0)

        with torch.no_grad():
            image = torch.from_numpy(image_data)
            device = torch.device('cuda:' + str(self.device) if torch.cuda.is_available() else 'cpu')
            image = image.to(device)
            # add batch_size dim: batch first
            # image = image.permute(2, 0, 1)
            # image = image.unsqueeze(0)
            pr = self.model(image)
            if type(pr) in [tuple, list]:
                # first output of multiple outputs
                pr = pr[0]
            # the first image
            pr = pr[0]

            # no need to do softmax if only max is chosed
            _, mask = torch.max(pr, dim=0)  # 有的网络输出可能是1分类，这种情况需要做exp并四舍五入
            # mask = mask.squeeze()  # squeeze the batch dimension

            # remove padding
            if roi[0] > 0 or roi[1] > 0:
                mask = mask[roi[1]: roi[1] + roi[3], roi[0]: roi[0] + roi[2]]
            mask = mask.cpu().numpy()

            # pr = F.softmax(pr, dim=0).cpu().numpy()
            # # pr = pr[1, ...]
            # pr = np.argmax(pr, axis=0)
            # # pr = pr.max(axis=0)

            # #   将灰条部分截取掉: remove padding
            # mask = pr[roi[1]: roi[1] + roi[3], roi[0]: roi[0] + roi[2]]

            mask = mask * 255
            mask = mask.astype(np.uint8)
            # 这里我不确定是否会造成性能损失，训练时是先interpolate再做分类
            mask = cv2.resize(mask, (original_w, original_h), interpolation=cv2.INTER_LINEAR)

            return mask

    # @classmethod
    # def letterbox(cls, img, new_shape, padding_value=(114, 114, 114), stride=32):
    #     """
    #     new_shape: (w, h)
    #     """
    #     t_w, t_h = new_shape

    #     MIN_SIZE = stride * 3
    #     # 如果刚好等于stride，有的环境下概率还会报错（神经网络最深层特征分辨率H或W为0），再调大一些能够解决
    #     padded_w = ((t_w + stride - 1) // stride) * stride if t_w > MIN_SIZE else MIN_SIZE
    #     padded_h = ((t_h + stride - 1) // stride) * stride if t_h > MIN_SIZE else MIN_SIZE

    #     dw = (padded_w - t_w) // 2
    #     dh = (padded_h - t_h) // 2

    #     img = cv2.resize(img, (t_w, t_h), interpolation=cv2.INTER_LINEAR)

    #     top = dh
    #     bottom = padded_h - t_h - dh
    #     left = dw
    #     right = padded_w - t_w - dw

    #     padded_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=padding_value)
    #     return padded_img, [left, top, t_w, t_h]
