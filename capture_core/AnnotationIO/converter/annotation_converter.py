#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@description:
@date       : 2022/03/25 19:23:34
@author     : Guanghua Tan
@email      : guanghuatan@hnu.edu.cn
@version    : 1.0
'''

import os
import cv2
import sys

import numpy as np
from loguru import logger

# Allow relative imports when being executed as script.
if __name__ == "__main__" and (__package__ is None or __package__ == ''):
    project_dir = os.path.join(os.path.dirname(__file__), '..')
    sys.path.insert(0, project_dir)

    __package__ = "AesCipher.converter"

from ..annotation_io import load_annotations, get_annotation_path


class AnnotationConverter:

    def __init__(self, annotation_path=None, polygon_to_box=False):
        """
        annotation_path: annotation file in json
        """
        self.image_annotations = {}
        self.config = {}

        # dir of the image folder
        self.root_path = ''
        self.polygon_to_box = polygon_to_box

        self.name2id = {}
        if not annotation_path:
            return

        # read annotations
        image_annotations = load_annotations(annotation_path, polygon_to_box)
        if image_annotations:
            if isinstance(image_annotations, (tuple, list)):
                self.image_annotations, self.config = image_annotations
            else:
                self.image_annotations = image_annotations

        # root path
        if os.path.isdir(annotation_path):
            self.root_path = annotation_path
        else:
            self.root_path, _ = os.path.split(annotation_path)

    @classmethod
    def cv2_imread(cls, image_path, flags=cv2.IMREAD_COLOR):
        """
        flags: IMREAD_UNCHANGED, IMREAD_COLOR, IMREAD_GRAY
        color image is stored in B G R format
        """
        img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), flags=flags)
        # if len(img.shape) > 2 and img.shape[2] > 3:
        #     img = img[..., :3]
        return img

    @classmethod
    def cv2_imwrite(cls, image_path, image):
        path, ext = os.path.splitext(image_path)
        cv2.imencode(ext, image)[1].tofile(image_path)

    def read_class_mapping(self, path):
        self.name2id = {}
        if not os.path.exists(path):
            logger.error(f'class mapping file does not exist: {path}')
            return

        with open(path, 'r', encoding='utf-8') as fs:
            for line in fs.readlines():
                # comment
                if not line or line.startswith('#'):
                    continue
                items = line.split(',')
                self.name2id[items[0]] = int(items[1])
            return True

    def anno_name_to_id(self, anno_name):
        return self.name2id.get(anno_name, -1)

    @classmethod
    def get_annotation_path(cls, base_path, prefer_gt=False):
        return get_annotation_path(base_path, prefer_gt)

    @classmethod
    def clear_directory(cls, path):
        logger.info(f'deleting files in {path} ...')
        # clear images in the directory
        for file_name in os.listdir(path):
            file_path = os.path.join(path, file_name)
            if os.path.isdir(file_path):
                os.removedirs(file_path)
            else:
                os.remove(file_path)

    def get_annotation_set(self, image_name):
        if image_name not in self.image_annotations:
            return None

        if 'annosets' not in self.image_annotations[image_name]:
            return None

        anno_set = self.image_annotations[image_name]['annosets']
        if not anno_set:
            return

        # use the first annotation set only
        anno_set = anno_set[0]
        return anno_set

    def get_polygon_annotations(self, image_name, anno_name=None):
        """
        if anno_name is not specified, return all polygon annotations
        if anno_name is specified, return polygon annotations with specified anno_name
        """
        if image_name not in self.image_annotations:
            return None

        if 'annosets' not in self.image_annotations[image_name]:
            return None

        anno_set = self.image_annotations[image_name]['annosets']
        if not anno_set:
            return

        # use the first annotation set only
        anno_set = anno_set[0]
        if 'annotations' not in anno_set:
            return

        poly_annotations = []
        for anno in anno_set['annotations']:
            if anno['type'] == 4 and (not anno_name or anno_name == anno['name']):
                poly_annotations.append(anno)

        return poly_annotations

    def get_ruler_info(self, image_name):
        if image_name not in self.image_annotations:
            return None

        if 'annosets' not in self.image_annotations[image_name]:
            return None

        anno_set = self.image_annotations[image_name]['annosets']
        if not anno_set:
            return None

        # use the first annotation set only
        anno_set = anno_set[0]
        if 'measure_results' not in anno_set:
            return None

        if 'ruler_info' not in anno_set['measure_results']:
            return None

        return anno_set['measure_results']['ruler_info']

    def get_roi(self):
        if 'roi' in self.config:
            return self.config['roi']
        return None

    def get_ruler_type(self):
        if 'ruler' in self.config:
            return self.config['ruler']['type']

        return 'right'

    def convert(self, target_path, ignore_image_with_no_anno=True):
        pass


if __name__ == '__main__':
    path = r'F:\datasets\BUltrasonic\0_20200613_095808_Trim.json'
    path = r'C:\Users\guang\Desktop\test\7th(黄文兰)\7th\annotations.cjson'
    converter = AnnotationConverter(path)
