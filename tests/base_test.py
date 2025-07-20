#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@description:
@date       : 2023/08/04 15:21:54
@author     : Guanghua Tan
@email      : guanghuatan@hnu.edu.cn
@version    : 1.0
'''

import os
import sys
from loguru import logger
import cv2
import numpy as np
import shutil
import subprocess
import SimpleITK

# Allow relative imports when being executed as script.
if __name__ == "__main__" and (__package__ is None or __package__ == ''):
    from pathlib import Path

    project_dir = str(Path(__file__).parents[1])
    sys.path.insert(0, project_dir)
    # import utility.sub_region_detector  # noqa: F401
    __package__ = "pystdplane"

from capture_core.AnnotationIO.converter.annotation_converter import AnnotationConverter
from common.model.image_info import ImageInfo
from common.model.AnnotationSet import AnnotationSet
from common.config.config import Config
from capture_core.QcDetection.utility import math_util
from capture_core.AnnotationIO.annotation_io import save_annotations
from capture_core.ruler.ruler_recognizer import RulerRecognizer

from common.mokey_patch import apply_monkey_patch

apply_monkey_patch()


class BaseTest():
    def __init__(self, root_path, test_with_annotations=True):
        '''constructor'''
        self.test_with_annotations = test_with_annotations

        self.config = Config()
        self.config.init_config()

        self.config.set_machine_type("default")
        self.config.check_consistence

        # RulerRecognizer.set_ruler_type(self.config['machine_type'])

        self.root_path = None
        self.video_path = None
        self.image_name = None

        self.anno_path = None
        self.anno_parser = None

        self.image2anno = {}

        self.show_result = True

        # copy hooks
        copy_git_hooks()

        if not os.path.exists(root_path):
            logger.error('root path does not exist: ' + root_path)
            return

        self.root_path = root_path

        # annotation path
        if os.path.isdir(root_path):
            if test_with_annotations:
                self.anno_path = AnnotationConverter.get_annotation_path(root_path, prefer_gt=True)
        else:
            # video
            root, ext = os.path.splitext(root_path)
            ext = ext.lower()
            # video
            if ext in ('.avi', '.mp4', '.wmv'):
                self.video_path = root_path

                if test_with_annotations:
                    anno_path = root + '.cjson'
                    if not os.path.exists(anno_path):
                        anno_path = root + '_result.json'
                        if not os.path.exists(anno_path):
                            logger.error('annotation path does not exist: ' + anno_path)
                            return
                    self.anno_path = anno_path
            elif ext in ('.jpg', '.bmp', '.png', '.tif', '.jpeg'):
                # image
                self.root_path, self.image_name = os.path.split(root_path)

                if test_with_annotations:
                    self.anno_path = AnnotationConverter.get_annotation_path(self.root_path)
                return

    def __call__(self, *args, **kwds):
        self.run()

    def run(self, start_frame_idx=0, save_result=False, anno_file='annotations.json'):
        """
        for video, start test from frame with idx == start_frame_idx
        """
        if self.root_path is None:
            return

        root_path = self.root_path
        # video
        root, ext = os.path.splitext(root_path)
        ext = ext.lower()
        # video
        if ext in ('.avi', '.mp4', '.wmv'):
            self.video_path = root_path
            anno_path = root + '.cjson'
            if not os.path.exists(anno_path):
                anno_path = root + '_result.json'
                if not os.path.exists(anno_path):
                    logger.error('annotation path does not exist: ' + anno_path)
                    # return
            self.anno_path = anno_path
        if self.test_with_annotations:
            # load annotations 判断是否有annotation（从detection得到的）
            if not self.anno_path:
                logger.error('annotation file does not exist')
            else:
                # 得到每张图片的名称和对应的annotation
                self.anno_parser = AnnotationConverter(self.anno_path, polygon_to_box=False)
                if self.anno_parser.config:
                    if 'ruler_type' in self.anno_parser.config:
                        RulerRecognizer.set_ruler_type(self.anno_parser.config['ruler_type'])
                    elif 'machine_type' in self.anno_parser.config:
                        RulerRecognizer.set_ruler_type(self.anno_parser.config['machine_type'])
                    if 'roi' in self.anno_parser.config:
                        ImageInfo.roi = self.anno_parser.config['roi']
        # 对单张图片的处理
        if self.image_name:
            # test one image only
            image_path = os.path.join(self.root_path, self.image_name)
            if self.image_name.endswith(".dcm"):
                ds = SimpleITK.ReadImage(image_path)
                image = SimpleITK.GetArrayFromImage(ds)[0, :, :, ::-1]  # RGB -> BGR
            else:
                image = AnnotationConverter.cv2_imread(image_path)
            anno_set = self.get_annotation_set(self.image_name)

            self.test_image(image, anno_set, 0, self.image_name)

            cv2.waitKey()

        # 对视频的处理
        elif self.video_path:
            # test each frame of video
            capturer = cv2.VideoCapture(self.video_path)
            total_frames = int(capturer.get(cv2.CAP_PROP_FRAME_COUNT))
            # jump to specified frame
            frame_idx = start_frame_idx

            if frame_idx > 0:
                capturer.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

            ret, image = capturer.read()
            is_paused = False
            self.image2anno = {}
            while ret:
                # print('frame idx', frame_idx)
                anno_set = self.get_annotation_set(str(frame_idx))
                try:
                    print(anno_set.plane_type)
                    if anno_set.plane_type== '颅脑部分切面':
                        
                        result = self.test_image(image, anno_set, frame_idx)
                        if result:
                            self.image2anno[frame_idx] = result
                    else:
                        self.image2anno[frame_idx] = anno_set
                except:
                    result = self.test_image(image, anno_set, frame_idx)
                    if result:
                        self.image2anno[frame_idx] = result
                if is_paused:
                    key = cv2.waitKey()
                else:
                    key = cv2.waitKey()

                # Esc
                if key == 27:
                    break

                # space
                if key == 32:
                    is_paused = not is_paused
                    if is_paused:
                        key = cv2.waitKey()

                    # Esc
                    if key == 27:
                        break
                    if key == 32:
                        is_paused = False

                frame_idx += 1
                ret, image = capturer.read()
            # if len(self.image2anno)>total_frames:
            #     raise RuntimeError(f"[错误] frame_idx 超出视频帧数范围：{len(self.image2anno)} > {total_frames}")
            if save_result:
                video_path, video_name = os.path.split(self.video_path)
                video_name, _ = os.path.splitext(video_name)
                anno_path = os.path.join(video_path, video_name + '_result1.json')
                self.save_result(anno_path)

        # 对文件夹的处理
        elif self.root_path:
            # test each image in this folder
            frame_idx = 0
            for image_name in os.listdir(self.root_path):

                _, ext = os.path.splitext(image_name)
                ext = ext.lower()
                if ext in ('.jpg', '.bmp', '.png', '.tif', '.jpeg', ".dcm"):
                    print(image_name)

                    image_path = os.path.join(self.root_path, image_name)
                    try:
                        if ext in (".dcm"):
                            ds = SimpleITK.ReadImage(image_path)
                            image = SimpleITK.GetArrayFromImage(ds)[0, :, :, ::-1]  # RGB -> BGR
                        else:
                            image = AnnotationConverter.cv2_imread(image_path)
                    except cv2.error:
                        continue

                    anno_set = self.get_annotation_set(image_name)

                    # 测量的入口
                    result = self.test_image(image, anno_set, frame_idx, image_name)
                    if result:
                        self.image2anno[image_name] = result

                    frame_idx += 1

                    key = cv2.waitKey() if self.show_result else -1
                    if key == 27:
                        break

            if save_result:
                anno_path = os.path.join(self.root_path, anno_file)
                self.save_result(anno_path)

        # close
        self.close()

    def test_image(self, image, annoset, frame_idx, image_name=None):
        # raise Exception('test_image should be overrided by subclass')
        pass

    def save_result(self, output_path):
        if not output_path or not self.image2anno:
            return

        save_annotations(output_path, self.image2anno)

    def close(self):
        pass

    @classmethod
    def get_image_info(cls, image, annoset=None):
        image_info = ImageInfo(image)
        image_info.anno_set = annoset
        return image_info

    def get_annotation_set(self, image_name):
        if self.anno_parser is None:
            return

        anno_set = self.anno_parser.get_annotation_set(image_name)
        anno_set = AnnotationSet.from_json(anno_set) if anno_set else None
        return anno_set

    @classmethod
    def get_type2mask(cls, anno_set, image_shape=None):
        """
        image_shape: [h, w], if image_shape is specified, construct mask from polygon
        """
        annotations = anno_set.get_polygon_annotations() if isinstance(anno_set, AnnotationSet) \
            else cls.get_polygon_annotations(anno_set)
        if not annotations:
            return {}

        type2mask = {}
        for anno in annotations:
            name = anno.name

            if len(anno.points) < 3:
                continue

            mask = None
            polygon = np.array(anno.points, np.int32)
            if image_shape:
                mask = np.zeros(image_shape, np.uint8)

                cv2.fillPoly(mask, [polygon], (255,))

            bbox = math_util.boundingbox(polygon)
            # draw on mask
            mask_info = {
                'mask': mask,
                'box': [*bbox[0], *bbox[1]],
                'score': 1.0,
                'polygon': None
            }
            if name in type2mask:
                type2mask[name].append(mask_info)
            else:
                type2mask[name] = [mask_info]

        return type2mask

    @classmethod
    def get_polygon_annotations(cls, anno_set, anno_name=''):
        if isinstance(anno_set, AnnotationSet):
            return anno_set.get_polygon_annotations(anno_name)

        poly_annotations = []
        for anno in anno_set['annotations']:
            if anno['type'] == 4 and (not anno_name or anno_name == anno['name']):
                poly_annotations.append(anno)

        return poly_annotations

    @classmethod
    def get_mask_info(cls, anno_set, image_shape, ann_name=''):
        """
        image_shape: [h, w]
        """
        poly_annos = cls.get_polygon_annotations(anno_set, ann_name)
        if not poly_annos:
            return

        mask_list = []
        for anno in poly_annos:
            polygon = np.array(anno.points, np.int32)
            mask = np.zeros(image_shape, np.uint8)
            cv2.fillPoly(mask, [polygon], (255,))

            mask_list.append(mask)

        return mask_list

    @classmethod
    def crop_image(cls, image, crop_pixels=10):
        if crop_pixels <= 0:
            return image

        height, width = image.shape[:2]
        if height > 2 * crop_pixels and width > 2 * crop_pixels:
            return image[crop_pixels:height - crop_pixels, crop_pixels:width - crop_pixels]

        return image


def copy_git_hooks():
    current_dir = os.path.dirname(__file__)
    # parent path
    current_dir = os.path.dirname(current_dir)

    pre_commit = os.path.join(current_dir, 'package_tool', 'git_hooks', 'pre-commit')
    if not os.path.exists(pre_commit):
        return

    # add +x
    if os.name == 'posix':
        subprocess.run(f'chmod +x {pre_commit}', shell=True, stdout=subprocess.PIPE).stdout

    copy_to_submodule_hooks(pre_commit, os.path.join(current_dir, '.git'))


def copy_to_submodule_hooks(pre_commit, sub_module_path):
    # copy to each module
    hooks_dir = os.path.join(sub_module_path, 'hooks')
    if not os.path.exists(hooks_dir):
        return

    shutil.copy(pre_commit, hooks_dir)

    # follow to each submodule
    module_dir = os.path.join(sub_module_path, 'modules')
    if not os.path.exists(module_dir):
        return

    for sub_module in os.listdir(module_dir):
        sub_module_path = os.path.join(module_dir, sub_module)
        if os.path.isdir(sub_module_path):
            copy_to_submodule_hooks(pre_commit, sub_module_path)


if __name__ == '__main__':
    root_path = r'H:\video_problem\jizhu'
    tester = BaseTest(root_path, True)
    tester.run()
