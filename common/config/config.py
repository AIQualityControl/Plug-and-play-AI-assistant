#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os
import json
from pathlib import Path
from typing import Union

from loguru import logger
from configparser import ConfigParser
from multiprocessing import Manager, current_process

from .midlate_model_config import model_params as midlate_model_params

classify_model_params = {
    'second and third trimesters': midlate_model_params['classification']
}

detection_model_params = {
    'second and third trimesters': midlate_model_params['detection']
}

measure_model_params = {
    'second and third trimesters': midlate_model_params['measure']
}

FETAL_KIND_2_IDX = {
    'first trimester': 1,
    'second and third trimesters': 2,
    'fetal echocardiogram': 4
}
IDX_2_FETAL_KIND = {id: kind for kind, id in FETAL_KIND_2_IDX.items()}


def normalized_fetal_kind(fetal_kind):
    if fetal_kind == 'fetal echocardiogram':
        return 'second and third trimesters'

    return fetal_kind


AUTO_CAPTURE_MODE = 0
MANUAL_CAPTURE_MODE = 1

AMNIOTIC_MEASURE_MODE = 2
LC_CAPTURE_MODE = 3
DICOM_TRANSFER_MODE = 4
SPECTURM_MEASURE_MODE = 5


class Config:
    def __init__(self, use_shared_config=False):
        '''constructor'''
        self.common_config = {}
        self.ruler_config = None
        self.last_machine_type = None
        self.last_detection_roi = None
        self.last_video_card_crop = False

        self.init_config(use_shared_config)

    @property
    def fetal_kind(self):
        return self.common_config['fetal_kind'][0]

    @property
    def fetal_kind_list(self):
        return self.common_config['fetal_kind']

    @property
    def machine_type(self):
        return self.common_config['machine_type']

    def __contains__(self, item):
        return item in self.common_config

    def __setitem__(self, key, value):
        self.common_config[key] = value

    def __getitem__(self, item):
        return self.common_config[item]

    def init_config(self, use_shared_config=False):
        root_path = Path.cwd().joinpath('capture_core', 'model_config')
        if not root_path.exists():
            rel_root_path = Path.cwd().joinpath('model_config')
            if not rel_root_path.exists():
                # create config path
                root_path.mkdir(parents=True)
            else:
                root_path = rel_root_path
            logger.info('config path does not exist: ' + str(root_path))

        config_path = root_path / 'config.json'

        config = {}
        with open(config_path, 'r', encoding='utf-8') as fs:
            try:
                config = json.load(fs)
            except Exception as e:
                print(e)
                logger.warning(f'Failed to init configuration: {config_path}, use default config')

        # default value
        # roi
        self.set_default_value(config)

        # write default config to config_path
        if not config_path.exists():
            with open(config_path, 'w', encoding='utf-8') as fs:
                json.dump(config, fs, ensure_ascii=False, indent=4)

        # read commit time
        commit_path = root_path / 'commit_info.txt'
        if commit_path.exists():
            with open(commit_path, 'r', encoding='utf-8') as fs:
                commit_time = fs.readline()
                if commit_time:
                    config['commit_time'] = commit_time.strip()

        if use_shared_config and config:
            self.common_config = Manager().dict()
            for k, v in config.items():
                self.common_config[k] = v
        else:
            self.common_config = config

        # ruler config
        ruler_config = self.init_ruler_config()

        # dict: {machine_type: roi}
        self.ruler_config = {machine_type: json.loads(ruler_config[machine_type]["roi"])
                             for machine_type in ruler_config.sections()}

        if self.ruler_config and config["machine_type"] not in self.ruler_config:
            config['machine_type'] = 'default'
            logger.error('config does not contain machine type: ' + config['machine_type'] + ', default is used')

    def init_ruler_config(self):
        root_path = Path.cwd().joinpath('capture_core', 'model_config', 'ruler')
        if not root_path.exists():
            rel_root_path = Path.cwd().joinpath('model_config', 'ruler')
            if rel_root_path.exists():
                root_path = rel_root_path
            else:
                logger.error('config path does not exist: ' + str(root_path))
                return

        config_path = root_path / 'ruler_config.ini'
        if not config_path.exists():
            logger.error('config path does not exist: ' + str(config_path))
            return

        parser = ConfigParser()
        parser.read(config_path, encoding="utf-8")

        return parser

    @classmethod
    def set_default_value(cls, config):
        if 'show_detection_roi' not in config:
            config['show_detection_roi'] = False

        if 'show_seg_roi' not in config:
            config['show_seg_roi'] = False

        if 'save_to_db' not in config:
            config['save_to_db'] = False

        if 'save_video' not in config:
            config['save_video'] = False

        if 'check_cont_frames' not in config:
            config['check_cont_frames'] = False

        if 'show_annotations' not in config:
            config['show_annotations'] = True

        if 'debug' not in config:
            config['debug'] = False

        if 'load_model' not in config:
            config['load_model'] = True

        if 'replace_task' not in config:
            config['replace_task'] = True

        if 'measure_mode' not in config:
            config['measure_mode'] = 'hadlock'

        if 'correct_CRL' not in config:
            config['correct_CRL'] = False

        if 'detection_with_thread' not in config:
            config['detection_with_thread'] = True

        if 'gpu_id' not in config:
            config['gpu_id'] = 0

        if 'log_level' not in config:
            config['log_level'] = 'INFO'

        if "video_card_crop" not in config:
            config["video_card_crop"] = True

        if 'show_gt' not in config:
            config['show_gt'] = False

        if 'measure_heart_spine' not in config:
            config['measure_heart_spine'] = False

        if 'shared_memory' not in config:
            config['shared_memory'] = {
                'video_shape': (1080, 1920, 3),
                'image_queue_length': 8,
                'display_queue_length': 16,
                'image_queue_name': 'image_queue',
                'display_queue_name': 'display_queue'
            }

        if 'fetal_kind' not in config:
            config['fetal_kind'] = ['second and third trimesters']
        elif isinstance(config['fetal_kind'], str):
            fetal_kind_list = config['fetal_kind'].split(',')
            config['fetal_kind'] = [fetal_kind.strip() for fetal_kind in fetal_kind_list]

        # use machine type to determine roi and ruler information
        if 'machine_type' not in config:
            if 'ruler_type' in config:
                config['machine_type'] = config['ruler_type']
                del config["ruler_type"]  # delete old "ruler_type"
            elif 'ruler' in config:
                config['machine_type'] = config['ruler']['type']
                del config["ruler"]  # delete old "ruler"
            else:
                config['machine_type'] = 'default'

        if 'roi' not in config:
            config['roi'] = [300, 75, 1300, 800]
        # clip roi to fit in video shape
        height, width = config['shared_memory']['video_shape'][:2]
        if config['roi'][0] + config['roi'][2] > width:
            config['roi'][2] = width - config['roi'][0]
        if config['roi'][1] + config['roi'][3] > height:
            config['roi'][3] = height - config['roi'][1]

    def save_config(self):
        root_path = Path.cwd().joinpath('capture_core', 'model_config')
        if not root_path.exists():
            rel_root_path = Path.cwd().joinpath('model_config')
            if not rel_root_path.exists():
                # create config path
                root_path.mkdir(parents=True)
            else:
                root_path = rel_root_path
            logger.info('config path does not exist: ' + str(root_path))

        config_path = root_path / 'config.json'
        with open(config_path, 'w', encoding='utf-8') as fs:
            json.dump(dict(self.common_config), fs, ensure_ascii=False, indent=4)

    def to_json(self):
        return json.dumps(dict(self.common_config), ensure_ascii=False)

    def get_detection_roi(self):
        return self.last_detection_roi

    def get_machine_type(self):
        return self.last_machine_type

    def get_video_card_crop(self):
        return self.last_video_card_crop

    def set_machine_type(self, machine_type=None, need_ruler=True):
        try:
            if machine_type is None:
                return False
            if machine_type not in self.ruler_config:
                logger.warning(f"{machine_type} does not exist.")
                return False

            self.common_config["machine_type"] = machine_type

            return self.check_machine_type(need_ruler)

        except Exception as e:
            logger.exception(e)

    def set_video_card_crop(self, video_card_crop: bool):
        try:
            self.common_config["video_card_crop"] = video_card_crop

            return self.check_video_card_crop()

        except Exception as e:
            logger.exception(e)

    def set_detection_roi(self, roi: Union[list, tuple]):

        try:
            self.common_config["roi"] = roi

            return self.check_detection_roi()
        except Exception as e:
            logger.exception(e)

    def check_machine_type(self, need_ruler=True):
        """
        whether machine type is changed
        """
        new_machine_type = self.common_config["machine_type"]
        if self.last_machine_type == new_machine_type:
            return False

        # ruler
        if need_ruler:
            from capture_core.ruler.ruler_recognizer import RulerRecognizer
            RulerRecognizer.set_ruler_type(new_machine_type)

        old_machine_type = self.last_machine_type
        self.last_machine_type = new_machine_type

        process_name = current_process().name

        # update roi
        # if process_name == "MainProcess":
        self._update_detection_roi()

        logger.info(f'{process_name}-{os.getpid()}: change machine_type '
                    f'from {old_machine_type} to {new_machine_type} with roi {self.last_detection_roi}')

        return True

    def check_video_card_crop(self):
        """
        whether video_card_crop is changed
        """
        video_card_crop = self.common_config['video_card_crop']
        if self.last_video_card_crop == video_card_crop:
            return False

        old_video_crop = self.last_video_card_crop
        self.last_video_card_crop = video_card_crop

        process_name = current_process().name

        # if process_name == "MainProcess":
        self._update_detection_roi()
        logger.info(f'{process_name}-{os.getpid()}: change video card crop '
                    f'from {old_video_crop} to {video_card_crop} with roi {self.last_detection_roi}')

        return True

    def check_detection_roi(self):
        # roi can be changed by ruler detection
        # self.common_config['video_card_crop']
        if self.last_detection_roi != self.common_config['roi']:
            new_detection_roi = self.common_config['roi']

            from common.model.image_info import ImageInfo
            ImageInfo.roi = new_detection_roi

            process_name = current_process().name
            logger.info(
                f'{process_name}-{os.getpid()}: change roi from {self.last_detection_roi} to {new_detection_roi}')

            self.last_detection_roi = new_detection_roi

            return True

        return False

    def _update_detection_roi(self):
        self.last_video_card_crop = self.common_config['video_card_crop']
        if self.last_video_card_crop:
            self.last_detection_roi = self.common_config['roi']
        else:
            machine_type = self.common_config['machine_type']
            self.last_detection_roi = self.ruler_config[machine_type]

        from common.model.image_info import ImageInfo
        ImageInfo.roi = self.last_detection_roi

    def check_consistence(self, need_ruler=True):
        # can not use short-cut judgement, since roi maybe not updated
        if self.check_machine_type() or self.check_video_card_crop() or self.check_detection_roi():
            return False
        return True

    def get_machine_type_list(self):
        if self.ruler_config is None:
            return []

        return self.ruler_config.keys()


def init_detection_planes():
    planes = {}
    root_path = Path.cwd().joinpath('capture_core', 'model_config')
    if not root_path.exists():
        rel_root_path = Path.cwd().joinpath('model_config')
        if not rel_root_path.exists():
            # create config path
            root_path.mkdir(parents=True)
        else:
            root_path = rel_root_path
        logger.info('config path does not exist: ' + str(root_path))

    detectPlanes_path = root_path / 'detectPlanes.json'
    with open(detectPlanes_path, 'r', encoding='utf-8') as fs:
        try:
            planes = json.load(fs)
        except Exception as e:
            print(e)
            logger.warning(f'Failed to init detectPlanes: {detectPlanes_path}, use default detect planes')

    return planes


def get_platform_info():
    import psutil
    import torch
    import winreg
    total_memory_size = str(round(psutil.virtual_memory().total / (1024 ** 3), 2)) + "GB"

    key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
    # QueryValueEx 
    cpu_name = winreg.QueryValueEx(key, "ProcessorNameString")  # get cpu name

    cuda_device_count = torch.cuda.device_count()
    cuda_device_list = [torch.cuda.get_device_properties(i) for i in range(cuda_device_count)]

    return {
        "MemorySize": total_memory_size,
        "CPU": cpu_name[0],
        "Logical Cores": psutil.cpu_count(logical=True),
        "Physical Cores": psutil.cpu_count(logical=False),
        "Freq": psutil.cpu_freq(percpu=True),
        "GPU": cuda_device_list,
    }


PLANES_TO_FILTER = [
    'All', 
    'brain', 
    'bone', 
    'kidney', 
    'heart', 
    'spine', 
    'foot', 
    'leg', 
    'umbilical cord', 
    'thalamic transverse plane', 
    'lateral ventricular transverse plane',
    'cerebellar transverse plane',
    'horizontal cross-section of upper abdomen', 
    'gallbladder level transverse plane', 
    'cervix', 
    'placenta', 
    'humerus long-axis plane', 
    'ulna and radius long-axis plane',
    'femur long-axis plane', 
    'tibia and fibula long-axis plane', 
    'forearm long-axis plane', 
    'calf long-axis plane',
    'calf coronal plane', 
    'foot plantar plane', 
    'palm plane', 
    'uterus', 
    'ovary', 
    'uterus longitudinal plane', 
    'ovary plane', 
    'uterus transverse plane',
    'uterus cervix longitudinal plane',
    'facial median sagittal plane', 
    'facial coronal plane', 
    'bilateral eyeballs horizontal transverse plane', 
    'nasolabial coronal plane'
]

PLANES_TO_REJUDGMENT = [
    'femur long-axis plane', 
    'humerus long-axis plane', 
    'thalamic transverse plane', 
    'horizontal cross-section of upper abdomen', 
    'cerebellar transverse plane',
    'lateral ventricular transverse plane',
    'fetal NT plane', 
    'fetal crown-rump length measurement plane'
]

UNNORMAL_PLANE_TYPES = [
    'uterus longitudinal plane_polyp', 
    'uterus transverse plane_polyp', 
    'uterus longitudinal plane_myoma', 
    'uterus transverse plane_myoma',
    'cleft palate-hard palate oblique coronal plane', 
    'cleft palate-soft palate oblique coronal plane', 
    'cleft lip-nasolabial coronal plane'
]

if __name__ == '__main__':
    info = get_platform_info()

    print(info)
