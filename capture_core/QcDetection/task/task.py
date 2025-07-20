import os
import cv2

from loguru import logger
import random
import csv

import multiprocessing
import importlib

import pydicom
import numpy as np
import requests


class Task:
    def __init__(self, model_dir, image_dir, db_config=None):
        '''constructor'''
        self.image_dir = image_dir
        self.model_dir = model_dir
        self.db_config = db_config
        self.config = db_config

        self.my_db = None
        self.model_list = []
        self.model_map = {
            '早孕期': {},
            '中晚孕期': {},
            '妇科': {},
            '甲状腺': {}
        }

        self.redis = None

        self.name_to_db_id_map = {}
        self.db_id_to_name_map = {}

        # 每个大类包含哪些小类
        self.image_sub_types = {}
        # 每个小类对应的大类
        self.sub_type_mapping = {}
        # 每个小类对应的阈值，如：{'小脑水平横切面': {'capture_threshold': 60, 'conf_threshold': 60}}
        # self.sub_type_thresh = {}

        self.is_finished = False

    # override function call
    def __call__(self, options=None):

        if self.init(options):
            self.run()
        else:
            process_name = multiprocessing.current_process().name
            logger.error(f"Failed to init process: {process_name}-{os.getpid()}")

    # dynamically creating model
    def create_model(self, params, gpu_id, load_model=True, package_root='qc_models'):

        path = f"{package_root}.{params['model_class']}"
        module = importlib.import_module(path)
        # eary.NtCtrlModel --> NtCtrlModel
        model_class = params['model_class'].split('.')[-1]
        class_meta = getattr(module, model_class)

        model = class_meta(model_dir=self.model_dir, gpu_id=gpu_id, load_model=load_model, **params['params'])

        if model:
            model.init_name_id_map(self.name_to_db_id_map, self.db_id_to_name_map)

        if load_model and (model is None or not model.is_inited()):
            logger.error('Failed to load: ' + path)

        return model

    @classmethod
    def get_session(cls):
        import tensorflow as tf

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        return tf.Session(config=config)

    def specify_gpu_device(self, gpu_id):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        # import keras

        # # # set the modified tf session as backend in keras
        # keras.backend.tensorflow_backend.set_session(self.get_session())

    def construct_name2id_map(self):

        if self.my_db is None or not self.my_db.is_connected():
            # raise Exception('db is not connected')
            logger.error('db is not connected, can be used only for testing with local images')
            return

        # mapping from name to db_id
        self.name_to_db_id_map = {}
        self.db_id_to_name_map = {}
        # plane_names = model_list[0].all_annotation_names()

        for plane_id, plane_name in self.my_db.query_all_plane_ids():
            self.name_to_db_id_map[plane_name] = plane_id
            self.db_id_to_name_map[plane_id] = plane_name

    def id_of_image_type(self, image_type):
        """
        return the id in db of the image type
        """
        if image_type in self.name_to_db_id_map:
            return self.name_to_db_id_map[image_type]
        return -1

    def image_type_of_id(self, type_id):
        """
        return the name of image type given the type id
        """
        if type_id in self.db_id_to_name_map:
            return self.db_id_to_name_map[type_id]
        return '未知'

    def get_target_image_size(self):
        if self.model_list:
            model = self.model_list[0]
            if model is not None:
                config = self.model_list[0].config
                return config['target_width'], config['target_height']
        return 640, 640

    def read_image(self, image_name):
        """
        read image from url or local file
        """
        tmp_split_word = image_name.split('/')
        if 'dicom' in tmp_split_word or 'http:' in tmp_split_word:
            url = image_name
            response = requests.get(url)
            if response.status_code == 200:
                img_dcm = response.content
                ds = pydicom.dcmread(img_dcm, force=True)
                img = ds.pixel_array
                image = np.array(img)
                return image
            else:
                logger.error('does not get image from: ' + url)
                return
        else:
            image_path = os.path.join(self.image_dir, image_name)
            if not os.path.exists(image_path):
                logger.error('image does not exist: ' + image_path)
                return

            # image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), flags=cv2.IMREAD_UNCHANGED)
            # image = image[:,:,:3]
            try:
                image = cv2.imread(image_path)
                return image
            except Exception:
                with open("/data/ultrasonic_project/srccode/opecv_error.txt", "a+", encoding='utf-8') as f:
                    f.write(f'===maybe opencv error:{image_path}===\n')
                return None

    def _get_image(self, image_name, from_image, image_map):
        """
        docstring
        """
        # image_path = os.path.join(self.image_dir, image_name)
        if from_image > 0:
            # sub images
            if from_image in image_map:
                # already read into memory
                image = image_map[from_image]
            else:
                image = self.read_image(image_name)
                image_map[from_image] = image

                # cv2.imshow('original multiple image', image)

        else:
            image = self.read_image(image_name)

        # cv2.waitKey()
        return image

    def _get_image_and_roi(self, image_name, roi, from_image, image_map):
        """
        docstring
        """
        # image_path = os.path.join(self.image_dir, image_name)
        croped_with_roi = False
        if from_image > 0:
            # sub images
            if from_image in image_map:
                # already read into memory
                image = image_map[from_image]
            else:
                image = self.read_image(image_name)
                image_map[from_image] = image

                # cv2.imshow('original multiple image', image)

            # get roi
            if image is not None:
                image = image[int(roi[1]): int(roi[1]) + int(roi[3]), int(roi[0]): int(roi[0]) + int(roi[2])]
                # roi = [0, 0, roi[2], roi[3]]
                croped_with_roi = True

        else:
            image = self.read_image(image_name)

            # roi = [left_, top, width, height]

        # 添加一个位用于判断是否已经用roi框进行裁剪了
        roi.append(croped_with_roi)
        # cv2.rectangle(image, (roi[0], roi[1]), (roi[0] + roi[2], roi[1] + roi[3]), (255, 0, 255))
        # cv2.imshow('image with roi', image)

        # cv2.waitKey()
        return image, roi

    @classmethod
    def add_ai_log_filter(cls, _logger, log_path, log_level='INFO'):
        _logger.add(log_path, backtrace=True, diagnose=True, enqueue=True, level=log_level,
                    rotation='00:00', encoding='utf-8', retention='10 days',
                    filter=lambda x: 'common' in x['name'] or 'capture_core' in x['name'] or 'QcDetection' in x['name'],
                    )

    @classmethod
    def add_model_create_log_filter(cls, _logger, log_path, log_level='INFO'):
        # filter=lambda x: 'create_model' in x['function'],
        _logger.add(log_path, backtrace=True, diagnose=True, enqueue=True, level=log_level,
                    rotation='00:00', encoding='utf-8', retention='10 days',
                    filter=lambda x: 'task.task' in x['name'],
                    )

    def init(self, options=None):
        if 'log_path' in options:
            log_path = options['log_path']
            model_log_path = None
            if isinstance(log_path, (tuple, list)):
                if len(log_path) == 1:
                    log_path = log_path[0]
                elif len(log_path) > 1:
                    log_path, model_log_path = log_path[0: 2]

            log_level = options['log_level'] if 'log_level' in options else 'INFO'
            if log_path:
                Task.add_ai_log_filter(logger, log_path, log_level)

            if model_log_path:
                Task.add_model_create_log_filter(logger, model_log_path, log_level)

        if self.db_config is None:
            if self.config is None or 'shared_memory' not in self.config:
                logger.warning(
                    'database connection parameter is not specified, only can be used for testing with local images')

            # plane name to db id
            self.construct_name2id_map()

            return True

        # MySQL
        db_config = self.db_config['db'] if 'db' in self.db_config else self.db_config

        from db.db import MySqlDb
        self.my_db = MySqlDb(db_config=db_config)
        if not self.my_db.is_connected():
            # raise Exception('db is not connected')
            logger.error('db is not connected')
            return False

        # plane name to db id
        self.construct_name2id_map()

        # redis
        if 'redis' in self.db_config:
            # whether use redis
            redis_config = self.db_config['redis']
            if redis_config is None:
                return True

            from db.redis import MyRedis
            self.redis = MyRedis(redis_config)
            if not self.redis.is_connected():
                logger.error('Redis is not connected. Please start redis server firstly')
                return False

        return True

    def init_image_subtypes(self, model_params):

        for plane_type in model_params:
            if plane_type == 'classification':
                continue

            model_param = model_params[plane_type]
            if 'sub_types' not in model_param:
                continue

            sub_types = model_param['sub_types']
            if len(sub_types) < 1:
                continue

            # construct mapping from sub_type to parent
            if plane_type in self.name_to_db_id_map:
                parent = self.id_of_image_type(plane_type)
            else:
                parent = self.id_of_image_type(sub_types[0])

            sub_type_ids = []
            for sub_type in sub_types:
                sub_id = self.id_of_image_type(sub_type)
                sub_type_ids.append(sub_id)

                self.sub_type_mapping[sub_id] = parent

            self.image_sub_types[plane_type] = sub_type_ids

    def parent_type(self, sub_id):
        sub_id = abs(sub_id)
        return self.sub_type_mapping[sub_id] if sub_id in self.sub_type_mapping else sub_id

    def sub_types(self, image_type):
        if image_type not in self.image_sub_types:
            return []
        return self.image_sub_types[image_type]

    def run(self):
        pass

    # ///////////////////// used for testing with local images ////////////////

    def run_with_local_images(self, images_list=None, batch_size=8, visualize_local=False,
                              classMapping_local=None, save_json=False):
        '''
        images_list == None: test all images in the image directory
        images_list: list of image name or list of images: test the image list
        '''
        self.visualize_local = visualize_local
        self.classMapping_local = classMapping_local
        self.save_json = save_json
        if images_list is None:
            # iterate the image directory
            return self.run_with_local_dir(batch_size)
        elif not isinstance(images_list, list):
            images_list = [images_list]

        end_idx = 0
        start_idx = 0
        while True:
            end_idx += batch_size
            if end_idx >= len(images_list):
                self._run_with_batch_helper(images_list[start_idx:])
                return
            else:
                self._run_with_batch_helper(images_list[start_idx:end_idx])
            start_idx = end_idx

    def run_with_local_dir(self, batch_size=8):
        '''
        iterate all images in the directory to test with batches
        '''
        from yoloair.utils.general import increment_path

        save_dir = increment_path('./runs/exp', exist_ok=False)
        save_dir.mkdir(parents=True, exist_ok=True)  # make dir
        images_to_detect = []
        for image_name in os.listdir(self.image_dir):
            if any(image_name.endswith(ext) for ext in ['.jpg', 'png', 'jpeg']):
                # image = self.read_image(image_name)
                if image_name is None:
                    continue

                images_to_detect.append(image_name)
                if len(images_to_detect) == batch_size:
                    self._run_with_batch_helper(images_to_detect, save_dir=save_dir)
                    images_to_detect = []

        if len(images_to_detect) > 0:
            self._run_with_batch_helper(images_to_detect, save_dir=save_dir)

    def _run_with_batch_helper(self, images_list, roi_list=None, save_dir=''):
        '''
        images_list can be list of images or image_name
        '''
        if images_list is None or len(images_list) == 0:
            return

        images_to_detect = images_list
        if isinstance(images_list[0], str):
            images_to_detect = []
            for image_name in images_list:
                image = self.read_image(image_name)
                if image is None:
                    continue
                images_to_detect.append(image)

        result_params, annotations = self.run_with_batch(images_to_detect, roi_list)

        import json
        if self.save_json:
            annotations_path = os.path.join(save_dir, 'annotations.json')
            if not os.path.exists(annotations_path):
                content = {
                    'annotations': {},
                }
            else:
                f = open(annotations_path, 'r', encoding='utf-8')
                content = json.load(f)
                f.close()

        if (self.visualize_local or self.save_json) and self.classMapping_local is not None:
            colors = {}
            keys = {}
            for planeType in self.classMapping_local:
                colors[planeType] = [[random.randint(0, 255) for _ in range(3)]
                                     for _ in self.classMapping_local[planeType]]
                keys[planeType] = list(self.classMapping_local[planeType].keys())
            # colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.classMapping_local]
            # keys = list(self.classMapping_local.keys())
            standards = ['标准', '基本标准', '非标准']

            for imageName, img0, annotation, result_param in zip(images_list, images_to_detect, annotations,
                                                                 result_params):
                img = img0.copy()
                parts = annotation['auto_annotations']
                plane_type = result_param['plane_type']
                if plane_type == '其它':
                    continue
                auto_score = result_param['auto_score']
                confidence = result_param['confidence']
                info = f'score:{auto_score:0.2f},conf:{confidence:0.2f}'
                if self.save_json:
                    content['annotations'][imageName] = {
                        'bodyPart': plane_type,
                        'standard': standards[result_param['auto_std_type'] - 1],
                        'auto_score': auto_score,
                        'confidence': confidence,
                        'annotations': [],
                    }
                for part in parts:
                    KAS_info = eval(part)
                    name = KAS_info['name']
                    if name not in self.classMapping_local[plane_type]:
                        print(f'{name} not in classMapping')
                        continue
                    #
                    label = f'{self.classMapping_local[plane_type][name]} {KAS_info["score"]:.2f}'
                    index = keys[plane_type].index(name)
                    vertex = eval(KAS_info['vertex'])
                    if self.visualize_local:
                        self.plot_one_box(vertex, img, label=label, color=colors[plane_type][int(index)],
                                          line_thickness=3)
                    if self.save_json:
                        content['annotations'][imageName]['annotations'].append({
                            'name': name,
                            'vertex': [[vertex[0], vertex[1]], [vertex[2], vertex[3]]],
                            'score': float(KAS_info["score"]),
                        })
                if self.visualize_local:
                    self.plot_one_box([0, 0, 0, 0], img=img, label=info, color=colors[plane_type][0], line_thickness=3,
                                      no_rectangle=True)
                    cv2.imwrite(os.path.join(save_dir, imageName), img)
            if self.save_json:
                f = open(os.path.join(save_dir, 'annotations.json'), 'w', encoding='utf-8')
                json.dump(content, f, ensure_ascii=False, indent=4)
                f.close()

        # 修改 os.path.join(save_dir, imageName)
        fff = open(os.path.join(save_dir, 'description.csv'), 'a+', encoding='utf-8')
        csv_writer = csv.writer(fff)
        for i, image_name in enumerate(images_list):
            auto_score = result_params[i]['auto_score']
            auto_std_type = result_params[i]['auto_std_type']
            description = annotations[i]['auto_annotations']
            reason = annotations[i]['auto_reason']
            csv_writer.writerow([image_name, str(auto_score), str(auto_std_type), str(description), str(reason)])
            print(image_name, auto_score, str(auto_std_type), description)
        fff.close()
        return True

    def run_with_batch(self, images_list, roi_list=None):
        pass

    def plot_one_box(self, x, img, color=None, label=None, line_thickness=3, no_rectangle=False):
        # Plots one bounding box on image img
        tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
        color = color or [random.randint(0, 255) for _ in range(3)]
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        if no_rectangle:
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = t_size[0], t_size[1] + 3
            cv2.rectangle(img, (0, 0), c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(img, label, (0, c2[1]), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
            return
        cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        if label:
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


if __name__ == '__main__':
    pass
