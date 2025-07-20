import cv2
import numpy as np
from .TorchMeasureModel import TorchMeasureModel
from PIL import Image

import warnings

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms


class DDRNetModel(TorchMeasureModel):
    def __init__(self, model_file_name, class_mapping_file, config, load_model=True,
                 gpu_id=0, model_dir=r'/data/QC_python/model/'):
        '''constructor'''
        super(DDRNetModel, self).__init__(model_file_name, class_mapping_file, config,
                                          load_model, gpu_id, model_dir)

    def load_model(self, model_path, gpu_id, backbone_name):
        from ..thirdpty.DDRNet import hardmseg as hardmseg
        from ..thirdpty.DDRNet.stdc import model_stage as stdc
        from ..thirdpty.DDRNet.fastseg import Unet as unet
        from ..thirdpty.DDRNet.ddrnet_23_slim import build_model
        from ..thirdpty.DDRNet.polyp import pvt

        self.model_name = str(model_path).split('/')[-1]
        if 'hardnet' in self.model_name:
            model = hardmseg.build_model(model_path, False)
        elif 'stdc' in self.model_name:
            model = stdc.build_model(model_path, False)
        elif 'unet' in self.model_name:
            # options: Unet, Att_Unet, R2Att_Unet, R2Unet, NestedUnet, use target_size:256*256
            model = unet.build_model(model_path, False, model_name='Unet')
        elif 'pvt' in self.model_name:
            model = pvt.build_model(model_path)
        else:
            model = build_model(model_path, False)  # by default, it is set to ddrnet

        if 'pvt' not in self.model_name:
            try:
                # add device param!
                pretrained_dict = torch.load(model_path, map_location=torch.device(self.device))
                if 'state_dict' in pretrained_dict:
                    pretrained_dict = pretrained_dict['state_dict']
                model_dict = model.state_dict()
                pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items()
                                   if k[6:] in model_dict.keys()}
                model_dict.update(pretrained_dict)
                model.load_state_dict(model_dict)
            except Exception:
                warnings.warn(f'loading model {model_path} error, wrong state_dict or model is broken!',
                              RuntimeWarning)

        model.eval()
        self.model = model.cuda(self.device) if self.device != 'cpu' else model
        return self.model

    def do_segment(self, image_list, image_info_list):

        if len(image_list) == 0:
            return

        t_w, t_h = self.config['target_width'], self.config['target_height']
        # t_w, t_h = 256, 256
        results = []
        for img in image_list:
            original_h, original_w = img.shape[:2]
            if 'pvt' not in self.model_name:
                resized, roi = self.cv2_padding_resize(img, (t_w, t_h), padding_value=self.padding_value)
                # convert to color RGB mode
                if len(resized.shape) < 3 or resized.shape[2] == 1:
                    resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
                else:
                    resized = resized[:, :, ::-1]

                # normalize to [0, 1]
                image_data = resized.astype(np.float32) / 255.0

                # add batch_size dim: batch first
                image_data = np.expand_dims(np.transpose(image_data, (2, 0, 1)), 0)
                image = torch.from_numpy(image_data)
            else:
                trasnformer = transforms.Compose([
                    transforms.Resize((t_w, t_h)),
                    transforms.ToTensor()])
                image = trasnformer(Image.fromarray(img).convert('RGB'))
                image = torch.unsqueeze(image, dim=0)

            if self.device != 'cpu':
                image = image.to(f'cuda:{str(self.device)}')

            with torch.no_grad():
                pr = self.model(image)
                if 'pvt' in self.model_name:
                    res = F.upsample(pr[0] + pr[1], size=(original_h, original_w), mode='bilinear', align_corners=False)
                    res = res.sigmoid().data.cpu().numpy().squeeze()
                    res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                    if 'pl' in self.model_name:
                        mask = (res > 0.5).astype(np.uint8) * 255
                    else:
                        mask = (res * 255).astype(np.uint8)
                else:
                    if type(pr) in [tuple, list]:
                        # first output of multiple outputs
                        pr = pr[0]
                    # the first image
                    pr = pr[0]

                    # no need to do softmax if only max is chosed
                    _, mask = torch.max(pr, dim=0)  # 有的网络输出可能是1分类，这种情况需要做exp并四舍五入
                    # mask = mask.squeeze()  # squeeze the batch dimension
                    # remove padding
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

                results.append(mask)
                # cv2.imshow('mask show', mask)
                # cv2.waitKey()

        return results
