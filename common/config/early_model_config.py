#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@description:
@date       : 2023/04/29 15:07:15
@author     : Guanghua Tan
@email      : guanghuatan@hnu.edu.cn
@version    : 1.0
'''

model_params = {
    'classification': [
        {
            'model_class': 'ClassificationModel_Swin',
            'detect_with_roi': False,  # detect with roi,
            'capture_threshold': 80,
            'params': {
                'model_file_name': 'zaoyun_classify_model_24_08_22.pth',
                'class_mapping_file': 'zaoyun_classmapping_11.csv',
                'config': {
                    'backbone': 'resnet50',
                    'agnostic_nms': False,
                    'stride': 32,
                    'max_detections': 2,
                    'target_width': 224,
                    'target_height': 224,
                    'score_threshold': 0.2,
                    'iou_threshold': 0.45,
                    'fetal_kind': '早孕期',
                    'config_path': 'swin_transformer/swin_small_patch4_window7_224_zaoyun.yaml'
                }
            }
        }
        # {
        #     'model_class': 'ClassificationModel_Yolo',
        #     'conf_threshold': 0,
        #     'capture_threshold': 60,
        #     'detect_with_roi': True,  # detect with roi,
        #     'params': {
        #         'model_file_name': 'early_cls13_yolov5.pt',
        #         'class_mapping_file': 'early_cls13_yolov5_classmapping.csv',
        #         'config': {
        #             'backbone': 'resnet50',
        #             'agnostic_nms': False,
        #             'stride': 32,
        #             'max_detections': 4,
        #             'target_width': 160,
        #             'target_height': 160,
        #             'score_threshold': 0.2,
        #             'iou_threshold': 0.45
        #         }
        #     }
        # }
    ],
    'detection': {
        '双眼球及双耳冠状切面': [{
            'model_class': 'early.EyeEarEarlyModel',
            'conf_threshold': 45,
            'capture_threshold': 60,
            'detect_with_roi': False,  # detect with roi,
            'params': {
                'model_file_name': 'best_sysr.pt',
                # 'model_file_name': 'resnet50_csv_50_detect_eye.h5',
                'class_mapping_file': 'eyeear_early_classmapping.csv',
                'config': {
                    'backbone': 'resnet50',
                    'agnostic_nms': False,
                    'stride': 32,
                    'max_detections': 10,
                    'target_width': 640,
                    'target_height': 640,
                    'score_threshold': 0.25,
                    'iou_threshold': 0.45
                }
            }
        }],
        '鼻后三角冠状切面': [{
            'model_class': 'early.NosetriangleEarlyModel',
            'conf_threshold': 45,
            'capture_threshold': 60,
            'detect_with_roi': False,  # detect with roi,
            'params': {
                'model_file_name': 'best_oneBT_twoYE.pt',
                'class_mapping_file': 'nosetriangle_early_classmapping.csv',
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
        }],
        '尺桡骨冠状切面、胫腓骨冠状切面、小腿矢状切面': {
            'sub_types': ['上肢长轴切面', '双上肢长轴切面', '上臂长轴切面', '前臂长轴切面',
                          '下肢长轴切面', '双下肢长轴切面', '大腿长轴切面', '小腿长轴切面'],
            'model_list': [{
                'model_class': 'early.LimbEarlyModel',
                'capture_threshold': 60,
                'detect_with_roi': False,
                'params': {
                    'model_file_name': 'sx_merge_exp4.pt',  # shangzhi_early.pt
                    'class_mapping_file': 'limb_early_classmapping.csv',
                    'config': {
                        'backbone': 'resnet50',
                        'agnostic_nms': False,
                        'stride': 32,
                        'max_detections': 15,
                        'target_width': 640,
                        'target_height': 640,
                        'score_threshold': 0.25,
                        'iou_threshold': 0.4
                    }
                }
            }],
        },
        '脐带腹壁入口处横切面': [{
            'model_class': 'early.QikongEarlyModel',
            'capture_threshold': 70,
            'detect_with_roi': False,  # detect with roi,
            'params': {
                'model_file_name': 'qdfb_early.pt',
                'class_mapping_file': 'qikong_early_classmapping.csv',
                'config': {
                    'backbone': 'resnet50',
                    'agnostic_nms': True,

                    'stride': 32,
                    'max_detections': 10,
                    'target_width': 640,
                    'target_height': 640,
                    'score_threshold': 0.2,
                    'iou_threshold': 0.45,

                }
            }
        }],
        '膀胱水平横切面彩色多普勒': [{
            'model_class': 'early.Bladder_zao_Model',
            'capture_threshold': 70,
            'detect_with_roi': False,  # detect with roi,
            'params': {
                'model_file_name': 'pgsp_best_20230619.pt',
                'class_mapping_file': 'bladder_zao_classmapping.csv',
                'config': {
                    'backbone': 'resnet50',
                    'agnostic_nms': True,

                    'stride': 32,
                    'max_detections': 10,
                    'target_width': 640,
                    'target_height': 640,
                    'score_threshold': 0.2,
                    'iou_threshold': 0.45,

                }
            }
        }],
        # '颅脑部分切面': {
        #     'sub_types': ['侧脑室水平横切面', '小脑冠状切面'],
        #     'model_list': [{
        #         'model_class': 'early.LunaoEarlyModel',
        #         'detect_with_roi': False,  # detect with roi
        #         'capture_threshold': 60,
        #         'conf_threshold': 45,
        #         'params': {
        #             'model_file_name': 'best_lunao_early_230907.pt',
        #             'class_mapping_file': 'lunao_early_classmapping.csv',
        #             'config': {
        #                 'backbone': 'resnet50',
        #                 'agnostic_nms': True,
        #                 'stride': 32,
        #                 'max_detections': 10,
        #                 'target_width': 640,
        #                 'target_height': 640,
        #                 'score_threshold': 0.2,
        #                 'iou_threshold': 0.45,
        #             }
        #         },
        #     }],
        # },
        '颅脑部分切面': {
            'sub_types': ['侧脑室水平横切面', '小脑冠状切面', '中脑导水管水平横切面'],
            'model_list': [{
                'model_class': 'early.LunaoEarlyModel',
                'detect_with_roi': False,  # detect with roi
                'capture_threshold': 60,
                'conf_threshold': 45,
                'params': {
                    'model_file_name': 'best_lunao_early_240515.pt',
                    'class_mapping_file': 'lunao_early_classmapping.csv',
                    'config': {
                        'backbone': 'YOLOV8',
                        'agnostic_nms': True,
                        'stride': 32,
                        'max_detections': 10,
                        'target_width': 640,
                        'target_height': 640,
                        'score_threshold': 0.2,
                        'iou_threshold': 0.45,
                    }
                },
            }],
        },
        '宫颈内口矢状切面': [{
            'model_class': 'early.Gongjing_z_Model',
            'capture_threshold': 60,
            'detect_with_roi': False,  # detect with roi,
            'params': {
                'model_file_name': 'gjnk_z_241021s.pt',
                'class_mapping_file': 'gj_z_classmapping.csv',
                'config': {
                    'backbone': 'YOLOV8',
                    'agnostic_nms': True,
                    'stride': 32,
                    'max_detections': 10,
                    'target_width': 640,
                    'target_height': 640,
                    'score_threshold': 0.2,
                    'iou_threshold': 0.45,

                }
            }
        }],
        '胎盘脐带插入口切面': [{
            'model_class': 'early.Taipan_z_Model',
            'capture_threshold': 60,
            'detect_with_roi': False,  # detect with roi,
            'params': {
                'model_file_name': 'tp_z_618.pt',
                'class_mapping_file': 'tp_z_classmapping.csv',
                'config': {
                    'backbone': 'resnet50',
                    'agnostic_nms': True,
                    'stride': 32,
                    'max_detections': 10,
                    'target_width': 640,
                    'target_height': 640,
                    'score_threshold': 0.2,
                    'iou_threshold': 0.45,

                }
            }
        }],

        '上腹部、胆囊水平横切面': {
            'sub_types': ['上腹部水平横切面'],
            'model_list': [{
                'model_class': 'early.FuweiModelEarly',
                'detect_with_roi': False,
                'capture_threshold': 60,
                'params': {
                    'model_file_name': 'fubu_early.pt',
                    'class_mapping_file': 'FuWeiClassMappingEarly.csv',
                    'config': {
                        'backbone': 'resnet50',
                        'agnostic_nms': False,
                        'stride': 32,
                        'max_detections': 15,
                        'target_width': 640,
                        'target_height': 640,
                        'score_threshold': 0.05,
                        'iou_threshold': 0.4
                    }
                },
            }]
        },

        '胎儿NT与头臀长切面': {
            'sub_types': ['胎儿NT切面', '胎儿头臀长测量切面'],
            'model_list': [{
                'model_class': 'early.NtCrlModel',
                'detect_with_roi': True,  # detect with roi,
                'capture_threshold': 70,
                'conf_threshold': 65,
                'params': {
                    'model_file_name': 'ntcrl_240426_improveXHG.pt',
                    # 'model_file_name': 'ntcrl_up3_1129.pt',
                    'class_mapping_file': 'nt_crl_classmapping.csv',
                    'config': {
                        'backbone': 'YOLOV8',
                        # 'backbone': 'resnet50',
                        'agnostic_nms': False,
                        'stride': 32,
                        'max_detections': 15,
                        'target_width': 640,
                        'target_height': 640,
                        'score_threshold': 0.2,
                        'iou_threshold': 0.25
                    }
                },
            }],
        },
        '肾、膈肌矢状切面': {
            'sub_types': ['左侧膈肌矢状切面', '右侧膈肌矢状切面'],
            'model_list': [{
                'model_class': 'early.KidneyLungEarlyModel',
                'detect_with_roi': False,  # detect with roi,
                'capture_threshold': 80,
                'conf_threshold': 80,
                'params': {
                    'model_file_name': 'diaphragm_early_v15.pt',
                    'class_mapping_file': 'chest_kidney_classmapping.csv',
                    'config': {
                        'backbone': 'YOLOV8',
                        'agnostic_nms': False,
                        'stride': 32,
                        'max_detections': 13,
                        'target_width': 640,
                        'target_height': 640,
                        'score_threshold': 0.3,
                        'iou_threshold': 0.25
                    }
                },
            }],
        },
        "心脏部分切面": {
            'sub_types': ["四腔心切面彩色多普勒", "三血管气管切面彩色多普勒"],
            'model_list': [{
                'model_class': 'early.HeartEarlyModel',
                'capture_threshold': 45,
                'detect_with_roi': False,  # detect with roi,
                'params': {
                    'model_file_name': 'HeartEarlymodel.pt',
                    'class_mapping_file': 'Heartearly.csv',
                    'config': {
                        'backbone': 'resnet50',
                        'agnostic_nms': False,
                        'stride': 64,
                        'max_detections': 10,
                        'target_width': 640,
                        'target_height': 640,
                        'score_threshold': 0.1,
                        'iou_threshold': 0.5
                    }
                }
            }]
        },
    },
    'measure': {
        '胎儿NT切面': {
            'model_class': 'NtMeasureModel',
            'detect_with_roi': True,
            'params': {
                'model_file_name': 'NT_240522_improve_acc.ckpt',  # f_score: 0.892
                'class_mapping_file': '',
                'config': {
                    'backbone': 'FastNT',
                    'padding_value': 0,
                    'target_width': 480,
                    'target_height': 160
                }
            }
        },
        '胎儿头臀长测量切面': {
            'model_class': 'CRLMeasureModel',
            'detect_with_roi': False,
            'params': {
                'model_file_name': 'CRL_upspeed_upacc_240926.ckpt',
                'class_mapping_file': 'CRL_measure_keypoint_mapping.csv',
                'config': {
                    'backbone': 'FastCRL',
                    'padding_value': 0,
                    'target_width': 640,
                    'target_height': 480
                }
            }
        },
    }
}
