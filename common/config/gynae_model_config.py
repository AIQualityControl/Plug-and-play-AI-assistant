#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@description:
@date       : 2023/04/29 15:07:03
@author     : Guanghua Tan
@email      : guanghuatan@hnu.edu.cn
@version    : 1.0
'''

model_params = {
    'classification': [{
        'model_class': 'ClassificationModel_Swin_gyn',
        'detect_with_roi': False,  # detect with roi,
        'capture_threshold': 80,
        'params': {
            'model_file_name': 'best_fuke_model_24_10_12.pth',
            'class_mapping_file': 'classmapping_classification_swin_gyn.csv',
            'config': {
                'backbone': 'resnet50',
                'agnostic_nms': False,
                'stride': 32,
                'max_detections': 1,
                'target_width': 224,
                'target_height': 224,
                'score_threshold': 0.2,
                'iou_threshold': 0.45,
                'fetal_kind': '妇科',
                'config_path': 'swin_transformer/swin_small_patch4_window7_224_gyn.yaml'
            }
        }
    }],
    'detection': {
        # 妇科的5大切面的配置：卵巢切面、子宫横切面、子宫纵切面、子宫宫颈纵切面、卵巢切面（1）
        '卵巢切面': [{
            'model_class': 'gynae.LCModel',
            'conf_threshold': 60,
            'capture_threshold': 60,
            'detect_with_roi': False,  # detect with roi,
            'params': {
                'model_file_name': 'zg_lc_230710.pt',
                'class_mapping_file': 'luanchao_classmapping.csv',
                'config': {
                    'backbone': 'resnet50',
                    'agnostic_nms': False,
                    'stride': 32,
                    'max_detections': 25,
                    'target_width': 640,
                    'target_height': 640,
                    'score_threshold': 0.2,
                    'iou_threshold': 0.45,
                }
            }
        }],

        '子宫横切面': {
            'sub_types': ['子宫横切面_肌瘤', '子宫横切面_息肉'],
            'model_list': [{
                # 'model_class': 'gynae.ZGHModel',
                'model_class': 'gynae.ZGHJLModel',
                'capture_threshold': 60,
                'conf_threshold': 60,
                'detect_with_roi': False,  # detect with roi,
                'params': {
                    # 'model_file_name': 'zgh_zgjl_230904.pt',
                    # 'model_file_name': 'zgjl_241220.pt',
                    'model_file_name': 'zgjl_rsl_250103.pt',
                    # 'class_mapping_file': 'zhigong_heng_classmapping.csv',
                    'class_mapping_file': 'zhigong_hen_classmapping_241225.csv',
                    'config': {
                        # 'backbone': 'resnet50',
                        'backbone': 'YOLOV8',
                        'agnostic_nms': False,
                        'max_detections': 10,
                        'stride': 32,
                        'target_width': 640,
                        'target_height': 640,
                        'score_threshold': 0.05,
                        'iou_threshold': 0.45
                    }
                }
            }]},

        '子宫纵切面': {
            'sub_types': ['子宫纵切面_肌瘤', '子宫纵切面_息肉'],
            'model_list': [{
                # 'model_class': 'gynae.ZGZModel',
                'model_class': 'gynae.ZGZJLModel',
                'capture_threshold': 60,
                'conf_threshold': 60,
                'detect_with_roi': False,  # detect with roi,
                'params': {
                    # 'model_file_name': 'zgz_zgjl_230904.pt',
                    # 'model_file_name': 'zgjl_0724n.pt',
                    # 'model_file_name': 'zgjl_241220.pt',
                    'model_file_name': 'zgjl_rsl_250103.pt',
                    # 'class_mapping_file': 'zhigong_zong_classmapping.csv',
                    'class_mapping_file': 'zhigong_zong_classmapping_241018.csv',
                    'config': {
                        # 'backbone': 'resnet50',
                        'backbone': 'YOLOV8',
                        'agnostic_nms': False,
                        'max_detections': 10,
                        'stride': 32,
                        'target_width': 640,
                        'target_height': 640,
                        'score_threshold': 0.2,
                        'iou_threshold': 0.45
                    }
                }
            }]
        },

        '子宫宫颈纵切面': [{
            'model_class': 'gynae.ZGGJModel',
            'capture_threshold': 60,
            'conf_threshold': 60,
            'detect_with_roi': False,  # detect with roi,
            'params': {
                # 'model_file_name': 'zg_zggj_230710.pt',
                'model_file_name': 'jyd_gjnk_241016.pt',
                'class_mapping_file': 'zhigong_gjz_classmapping.csv',
                'config': {
                    'backbone': 'YOLOV8',
                    'agnostic_nms': False,
                    'max_detections': 10,
                    'stride': 32,
                    'target_width': 640,
                    'target_height': 640,
                    'score_threshold': 0.2,
                    'iou_threshold': 0.45
                }
            }
        }],

        '早孕期切面': [{
            'sub_types': ['妊娠囊最大切面', '胚芽最大切面', '卵黄囊最大切面'],
            'model_class': 'gynae.EarlierMeasureModel',
            'detect_with_roi': True,
            'params': {
                'model_file_name': 'fuke_earlier_seg_240923.ckpt',
                'class_mapping_file': 'fuke_earlier_measure.csv',
                'config': {
                    'backbone': 'earlierunet',
                    'padding_value': 0,
                    'target_width': 416,
                    'target_height': 320
                }
            }
        }]

    },
    'measure': {
        # 正常切面

        '子宫纵切面': {
            'model_class': 'ZGZMeasureModel',
            'detect_with_roi': False,
            'params': {
                'model_file_name': 'zgz_4segn241125.pt',
                'class_mapping_file': 'zgz_measure_classmapping.csv',
                'config': {
                    'backbone': 'resnet50',
                    'score_threshold': 0.1,
                    'iou_threshold': 0.25,
                    'target_width': 640,
                    'target_height': 640,
                }
            }
        },
        '子宫横切面': {
            'model_class': 'ZGHMeasureModel',
            'detect_with_roi': False,
            'params': {
                'model_file_name': 'zgh_F240409n.pt',
                'class_mapping_file': 'zgh_measure_classmapping.csv',
                'config': {
                    'backbone': 'resnet50',
                    'score_threshold': 0.1,
                    'iou_threshold': 0.25,
                    'target_width': 640,
                    'target_height': 640,
                }
            }
        },
        '子宫宫颈纵切面': {
            'model_class': 'ZGGJMeasureModel',
            'detect_with_roi': False,
            'params': {
                'model_file_name': 'gjz_F240408n.pt',
                'class_mapping_file': 'zggj_measure_classmapping.csv',
                'config': {
                    'backbone': 'resnet50',
                    'score_threshold': 0.1,
                    'iou_threshold': 0.25,
                    'target_width': 640,
                    'target_height': 640,
                }
            }
        },
        '卵巢切面': {
            'model_class': 'LCMeasureModel',
            'detect_with_roi': False,
            'params': {
                'model_file_name': 'lc_s_cm15_241125.pt',
                'class_mapping_file': 'lc_measure_classmapping.csv',
                'config': {
                    'backbone': 'resnet50',
                    'score_threshold': 0.1,
                    'iou_threshold': 0.25,
                    'target_width': 640,
                    'target_height': 640,
                }
            }
        },

        # 异常结构

        '子宫纵切面_息肉': {
            'model_class': 'ZGXRMeasureModel',
            'detect_with_roi': False,
            'params': {
                'model_file_name': 'zgxr_z_n0531-seg.pt',
                'class_mapping_file': 'zgxr_zgz_measure_classmapping.csv',
                'config': {
                    'backbone': 'resnet50',
                    'score_threshold': 0.1,
                    'iou_threshold': 0.25,
                    'target_width': 640,
                    'target_height': 640,
                }
            }
        },
        '子宫横切面_息肉': {
            'model_class': 'ZGXRMeasureModel',
            'detect_with_roi': False,
            'params': {
                'model_file_name': 'zgh-zgxr-segm-240530.pt',
                'class_mapping_file': 'zgxr_zgh_measure_classmapping.csv',
                'config': {
                    'backbone': 'resnet50',
                    'score_threshold': 0.1,
                    'iou_threshold': 0.25,
                    'target_width': 640,
                    'target_height': 640,
                }
            }
        },
        '子宫纵切面_肌瘤': {
            'model_class': 'ZGJLMeasureModel',
            'detect_with_roi': False,
            'params': {
                'model_file_name': 'zgjl_z_n0603-seg.pt',
                # 'model_file_name': 'zgz_zgjl_seg241203n.pt',
                'class_mapping_file': 'zgjl_zgz_measure_classmapping.csv',
                'config': {
                    'backbone': 'resnet50',
                    'score_threshold': 0.1,
                    'iou_threshold': 0.25,
                    'target_width': 640,
                    'target_height': 640,
                }
            }
        },
        '子宫横切面_肌瘤': {
            'model_class': 'ZGJLMeasureModel',
            'detect_with_roi': False,
            'params': {
                'model_file_name': 'zgh-zgjl-segM-240602.pt',
                'class_mapping_file': 'zgjl_zgh_measure_classmapping.csv',
                'config': {
                    'backbone': 'resnet50',
                    'score_threshold': 0.1,
                    'iou_threshold': 0.25,
                    'target_width': 640,
                    'target_height': 640,
                }
            }
        },

        # 早孕期测量
        '早孕期切面': {
            'sub_types': ['卵黄囊最大切面', '胚芽最大切面', '妊娠囊最大切面'],
            'model_class': 'EarlyGynMeasureModel',
            'detect_with_roi': True,
            'params': {
                'model_file_name': 'fuke_earlier_seg_240725.ckpt',
                'class_mapping_file': 'gyn_early_measure_classmapping.csv',
                'config': {
                    'backbone': 'earlierunet',
                    'padding_value': 0,
                    'target_width': 416,
                    'target_height': 320
                }
            }
        }
    }
}
