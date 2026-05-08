from .QcModel import QcModel
import cv2
import numpy as np


class YoloModel(QcModel):
    """
    docstring
    """

    def __init__(self, model_file_name, class_mapping_file, config, load_model,
                 gpu_id=0, model_dir=r'/data/QC_python/model/'):
        """
        docstring
        """
        super().__init__(model_file_name, class_mapping_file, config, load_model, gpu_id, model_dir)

    def load_model(self, model_path, gpu_id, backbone_name):
        """
        docstring
        """

        from yolo.detector import YOLOv5, YOLOv8

        if backbone_name.upper() == 'YOLOV8':
            print("[DEBUG YoloModel] Creating YOLOv8 model...")
            model = YOLOv8(model_path, self.config["target_height"], str(gpu_id),
                           self.config["iou_threshold"], self.config["score_threshold"])
        else:
            print("[DEBUG YoloModel] Creating YOLOv5 model...")
            model = YOLOv5(model_path, (self.config["target_width"], self.config["target_height"]), str(gpu_id),
                           self.config["iou_threshold"], self.config["score_threshold"],
                           self.config["stride"], self.config["agnostic_nms"])
        return model

    def preprocess(self, raw_image, target_size):
        # from yolov5.detector import preprocess_yolo

        # return preprocess_yolo(raw_image, target_size=target_size)
        return raw_image

    def detect(self, image_list):
        """
        docstring
        """
        if not isinstance(image_list, (list, tuple)):
            image_list = [image_list.image]


        results = self.model.predict(image_list)
        boxes, scores, labels = results[:3]
        boxes, scores, labels = self.postprocess(boxes, scores, labels,
                                                 score_threshold=self.config["score_threshold"],
                                                 max_detections=self.config["max_detections"],
                                                 )

        return boxes, scores, labels

    def postprocess(self, boxes_batch, scores_batch, labels_batch, score_threshold=0.2, max_detections=10):

        boxes_result = []
        scores_result = []
        labels_result = []

        # print(score_threshold)

        for boxes, scores, labels in zip(boxes_batch, scores_batch, labels_batch):
            if score_threshold > 0:
                # select indices which have a score above the threshold
                indices = np.where(scores > score_threshold)[0]

                # select those scores
                scores = scores[indices]

                # find the order with which to sort the scores
                scores_sort = np.argsort(-scores)[:max_detections]

                # select detections
                image_boxes = boxes[indices[scores_sort], :]
                image_scores = scores[scores_sort]
                image_labels = labels[indices[scores_sort]]
            else:
                # find the order with which to sort the scores
                scores_sort = np.argsort(-scores)[:max_detections]

                # select detections
                image_boxes = boxes[scores_sort, :]
                image_scores = scores[scores_sort]
                image_labels = labels[scores_sort]

            # unique except for BM has two pieces, if has Xiaonao, BM has one pieces at most
            if max_detections > 1:
                labels, scores, boxes = self.postprocess_special(image_labels, image_scores, image_boxes)

            else:
                # for label in labels:
                #     label_indices = np.where(image_labels == label)[0]
                #     image_labels, indices = np.unique(image_labels, return_index=True)
                #
                #     for j in range(1, len(label_indices)):
                #         indices = np.append(indices, label_indices[j])
                #         image_labels = np.append(image_labels, [BM_label, T_label,
                #                                                 E_label, L_label, SP_label, K_label, RP_label])

                labels = image_labels
                scores = image_scores
                boxes = image_boxes

            labels_result.append(labels)
            scores_result.append(scores)
            boxes_result.append(boxes)

        return boxes_result, scores_result, labels_result

    def postprocess_special(self, image_labels, image_scores, image_boxes):
        # get one detection at most for each label
        image_labels, indices = np.unique(image_labels, return_index=True)
        # print("unique:", image_labels, indices)
        image_scores = image_scores[indices]
        image_boxes = image_boxes[indices]

        return image_labels, image_scores, image_boxes
