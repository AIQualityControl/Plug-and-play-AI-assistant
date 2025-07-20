model_params = {
    'classification': {

    },
    'detection': {
        '甲状腺切面': [{
            'model_class': 'thyroid.ThyroidModel',
            'conf_threshold': 60,
            'capture_threshold': 60,
            'detect_with_roi': True,  # detect with roi,
            'params': {
                #thyroid_detection_115.pt
                'model_file_name': 'thyroid_detection_115.pt',
                'class_mapping_file': 'thyroid_classmapping.csv',
                'config': {
                    'backbone': 'YOLOV8',
                    'agnostic_nms': False,
                    'stride': 32,
                    'max_detections': 25,
                    'target_width': 640,
                    'target_height': 640,
                    'score_threshold': 0.4,
                    'iou_threshold': 0.45,
                }
            }
        }],
    },
    'measure': {
        '甲状腺切面': {
            'model_class': 'ThyroidClsModel',
            'detect_with_roi': True,
            'params': {
                'model_file_name': 'thyroid_cls1128_best1.pth.tar',
                'class_mapping_file': '',
                'config': {
                    'backbone': 'resnet50',
                    'padding_value': 128,
                    'target_width': 224,
                    'target_height': 224
                }
            }
        }
    }
}
