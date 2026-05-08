#!/usr/bin/env python
# -*- encoding: utf-8 -*-

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
                    'fetal_kind': 'second and third trimesters',
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
        'cranial plane': {
            'sub_types': ['thalamic transverse plane', 'lateral ventricular transverse plane', 'cerebellar transverse plane', 'fan-shaped region', 'cranial vault transverse plane'],
            'model_list': [{
                'model_class': 'midlate.LunaoALLModel',
                'detect_with_roi': False,  # used to indicate whether has to extract roi
                'capture_threshold': 65,
                'params': {
                    'model_file_name': 'lunao.pt',
                    'class_mapping_file': 'lunao_classmapping.csv',
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
        'Long-axis plane of the femur or humerus': {
            'sub_types': ['femur long-axis plane', 'humerus long-axis plane'],
            'model_class': 'midlate.BoneModel',
            'measure_threshold': 80,
            'capture_threshold': {'femur long-axis plane': 80, 'humerus long-axis plane': 80, 'femur long-axis plane(1)': 60, 'humerus long-axis plane(1)': 60},
            'detect_with_roi': False,  # detect with roi,
            'params': {
                'model_file_name': 'yolo_bone_0516.pt',
                'class_mapping_file': 'bone_classmapping.csv',
                'config': {
                    'backbone': 'resnet50',
                    'agnostic_nms': False,
                    'stride': 32,
                    'max_detections': 10,
                    'target_width': 640,
                    'target_height': 640,
                    'score_threshold': 0.2,
                    'iou_threshold': 0.45
                }
            }
        },
    },
    'measure': {
        'femur humerus measurement': {
            'sub_types': ['femur long-axis plane', 'humerus long-axis plane'],
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
        'Head circumference and abdominal circumference measurement': {
            'sub_types': ['thalamic transverse plane', "Horizontal cross-section of upper abdomen"],
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
        'cranial measurement plane': {
            'sub_types': ['cerebellar transverse plane', 'lateral ventricular transverse plane'],
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
