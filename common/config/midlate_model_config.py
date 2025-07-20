#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@description:
@date       : 2023/04/29 15:07:09
@author     : Guanghua Tan
@email      : guanghuatan@hnu.edu.cn
@version    : 1.0
'''

model_params = {
    'classification': [
        {
            'model_class': 'ClassificationModel_Swin',  # 'SwinOnnx'
            'detect_with_roi': False,  # detect with roi,
            'capture_threshold': 80,
            'params': {
                'model_file_name': 'midlate_classify_24_11_11.pth',
                'class_mapping_file': 'midlate_classmapping_26.csv',
                'config': {
                    'backbone': 'resnet50',
                    'agnostic_nms': False,
                    'stride': 32,
                    'max_detections': 2,
                    'target_width': 224,
                    'target_height': 224,
                    'score_threshold': 0.2,
                    'iou_threshold': 0.45,
                    'fetal_kind': '中晚孕期',
                    'config_path': 'swin_transformer/swin_small_patch4_window7_224.yaml'
                }
            }
        },
        {
            'model_class': 'midlate.ManualClassification',
            'detect_with_roi': False,  # detect with roi,
            'capture_threshold': 80,
            'params': {
                'model_file_name': 'midlate_manual_classification.pt',
                'class_mapping_file': 'midlate_manual_classification.csv',
                'config': {
                    'backbone': 'YOLOV8',
                    'agnostic_nms': False,
                    'stride': 32,
                    'max_detections': 300,
                    'target_width': 640,
                    'target_height': 640,
                    'score_threshold': 0.1,
                    'iou_threshold': 0.45
                }
            }
        }
    ],
    'detection': {
        '颅脑部分切面': {
            'sub_types': ['丘脑水平横切面', '侧脑室水平横切面', '小脑水平横切面', '透明隔腔水平横切面', '颅顶部横切面'],
            'model_list': [{
                'model_class': 'midlate.LunaoALLModel',
                'detect_with_roi': False,  # used to indicate whether has to extract roi
                'capture_threshold': 65,
                'params': {
                    # 'model_file_name': r"G:\Axforce\bestval数据清洗后的.pt",
                    # 'model_file_name': r"last.pt",
                    'model_file_name': r'C:\Users\wanghang\Desktop\修改了透明隔切面的标签.pt',
                    'class_mapping_file': 'lunao_classmapping.csv',
                    # 'parts_weight_mapping': {'透明隔腔': 40, "脑中线": 20, "大脑外侧裂": 30, '丘脑': 30, "脉络丛": 20 },
                    'config': {
                        'backbone': 'YOLOV8',
                        'agnostic_nms': True,
                        'stride': 32,
                        'max_detections': 25,
                        'target_width': 640,
                        'target_height': 640,
                        'score_threshold': 0.1,
                        'iou_threshold': 0.45
                    }
                }
            }]
        },
    },
    'measure': {
        '股骨肱骨测量': {
            'sub_types': ['股骨长轴切面', '肱骨长轴切面'],
            'model_class': 'FLHLMeasureModel',
            'detect_with_roi': True,
            'params': {
                'model_file_name': r"pvt_bone_1206.pth",
                'class_mapping_file': '',
                'config': {
                    'backbone': 'resnet50',
                    'padding_value': 128,
                    'target_width': 352,
                    'target_height': 352
                }
            }
        },
        '头围腹围测量': {
            'sub_types': ['丘脑水平横切面', '上腹部水平横切面'],
            'model_class': 'HcBpdMeasureModel',
            'detect_with_roi': True,
            'params': {
                'model_file_name': r'pvt_HC-AC_1206.pth',
                'class_mapping_file': '',
                'config': {
                    'backbone': 'resnet50',
                    'padding_value': 0,
                    'target_width': 640,
                    'target_height': 640,
                    # 'measure_mode': 'intergrowth-21st'  # options: intergrowth-21st or hadlock
                }
            }
        },
        '颅脑测量切面': {
            'sub_types': ['小脑水平横切面', '侧脑室水平横切面'],
            'model_class': 'LunaoMeasureModel',
            'detect_with_roi': False,
            'params': {
                'model_file_name': r'Head_solov2_231005.pth',
                'class_mapping_file': 'lunao_measure_classmapping_solo.csv',
                'config': {
                    'backbone': 'resnet50',
                    'score_threshold': 0.3,
                    'target_width': 640,
                    'target_height': 640,
                    'config_path': r'mmdetection/solov2_r50_fpn_1x_coco.py'
                }
            }
        },
    }
}
