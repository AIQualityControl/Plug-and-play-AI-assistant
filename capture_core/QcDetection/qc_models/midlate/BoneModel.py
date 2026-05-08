from ..YoloModel import YoloModel
import numpy as np
import math
import cv2
import os
import random
import json
from ...utility import math_util


class BoneModel(YoloModel):
    """
    docstring
    """
    PrePic = []  # planetype score frame_id
    CapedPic = [0, 0, 0, 0] 
    last_frame = -1  

    def __init__(self, model_file_name, class_mapping_file, config, load_model,
                 gpu_id=0, model_dir=r'/data/QC_python/model/'):
        """
        docstring
        """

        config['score_threshold'] = 0.25
        super().__init__(model_file_name, class_mapping_file, config, load_model, gpu_id, model_dir)

        parts_weight_mapping = {
            'femur': 35, "PEF": 25, 'DEF': 25, 'femur contour': 15, 'GUIN': 0, 'GUOUT': 0, 'humerus': 35, 'PEH': 25,
            'DEH': 25,
            'humerus contour': 15, 'GOIN': 0, 'GOOUT': 0}
        self.init_weight(parts_weight_mapping)
        self.bone_orientation = -1
        self.image_history = None

        self.check_num = 20
        self.minus_base_score = 5.0
        self.add_base_score = 5.0
        self.same_plane_type_add_score = 0.0

        self.enlarge_extra_score = 0.0

    def init_name_id_map(self, name_to_db_id_map, db_id_to_name_map):
        self.name_to_db_id_map = name_to_db_id_map
        self.db_id_to_name_map = db_id_to_name_map

        self.gu_type_id = self.name_to_db_id_map['femur long-axis plane']
        self.gu1_type_id = self.name_to_db_id_map['femur long-axis plane(1)']
        self.gong_type_id = self.name_to_db_id_map['humerus long-axis plane']
        self.gong1_type_id = self.name_to_db_id_map['humerus long-axis plane(1)']
        self.cr_type_id = self.name_to_db_id_map['尺桡骨冠状切面']
        self.cr1_type_id = self.name_to_db_id_map['尺桡骨冠状切面(1)']
        self.jf_type_id = self.name_to_db_id_map['胫腓骨冠状切面']
        self.jf1_type_id = self.name_to_db_id_map['胫腓骨冠状切面(1)']

        self.s_type_id = self.name_to_db_id_map['手掌冠状切面']
        self.qbcz_type_id = self.name_to_db_id_map['前臂长轴切面']
        self.xtcz_type_id = self.name_to_db_id_map['小腿长轴切面']
        self.xtsz_type_id = self.name_to_db_id_map['小腿矢状切面']

        self.minus_plane_type_ids = [self.cr_type_id, self.cr1_type_id, self.jf_type_id,
                                     self.jf1_type_id, self.s_type_id, self.qbcz_type_id,
                                     self.xtcz_type_id, self.xtsz_type_id]
        self.hl_fl_plane_type_ids = [self.gu_type_id, self.gu1_type_id, self.gong_type_id, self.gong1_type_id]

    def get_plane_classes(self, plane_type):
        if plane_type == "femur long-axis plane":
            return [0, 1, 2, 3, 4, 5]
        else:
            return [6, 7, 8, 9, 10, 11]

    def calculate_qc_score(self, std_info, std_score, parts_found, boxes_list, score_list, image, plane_type):
        """
        docstring
        """
        std_score = 0
        full_bone, gh1, gh2 = 0, 0, 0

        if 'DEF' not in parts_found and "PEF" not in parts_found and \
                'PEH' not in parts_found and 'DEH' not in parts_found:
            std_score = 0

        else:

            if ('humerus' in parts_found) or ('femur' in parts_found):
                if 'femur' in parts_found:
                    std_score = 15 + 35 * parts_found['femur']
                if 'humerus' in parts_found:
                    std_score = 15 + 35 * parts_found['humerus']
                if "PEF" in parts_found:
                    full_bone += 1
                    gh1 = parts_found["PEF"]
                    std_score += 10 + parts_found["PEF"] * 15
                if 'DEF' in parts_found:
                    full_bone += 1
                    gh2 = parts_found['DEF']
                    std_score += 10 + parts_found['DEF'] * 15
                if 'femur contour' in parts_found:
                    std_score += parts_found['femur contour'] * 10
                if 'GUIN' in parts_found:
                    std_score += parts_found['GUIN'] * 5
                if 'GUOUT' in parts_found:
                    std_score += parts_found['GUOUT'] * 5
                if 'PEH' in parts_found:
                    full_bone += 1
                    gh1 = parts_found['PEH']
                    std_score += 10 + parts_found['PEH'] * 15
                if 'DEH' in parts_found:
                    full_bone += 1
                    gh2 = parts_found['DEH']
                    std_score += 10 + parts_found['DEH'] * 15
                if 'humerus contour' in parts_found:
                    std_score += parts_found['humerus contour'] * 10
                if 'GOIN' in parts_found:
                    std_score += parts_found['GOIN'] * 5
                if 'GOOUT' in parts_found:
                    std_score += parts_found['GOOUT'] * 5
            else:
                std_score = 0
        # print(f'{self.frame_idx}:std_score:{std_score}')

        if std_score > 80:
            std_score = 80 + (std_score - 80) / 6

        if std_score >= 70 and full_bone < 2:
            std_score = 60 + (std_score - 70) / 6

        if full_bone == 2:
            diff = abs(gh1-gh2) * 100
            if diff > 10:
                std_score -= 0.15 * diff

        if std_score >= 80:
            label = tuple(parts_found.keys())
            if len(label) == 0:
                return std_score
            i_bone = -1
            for i in range(len(label)):
                if label[i] == 'humerus' or label[i] == 'femur':
                    i_bone = i
                    break
            if i_bone == -1:
                return std_score
            yt, yd, xl, xr = boxes_list[i_bone][1], boxes_list[i_bone][3], boxes_list[i_bone][0], boxes_list[i_bone][2]
            crop = image[int(yt):int(yd), int(xl):int(xr)]
            GrayImage = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            mea_thres = min(np.mean(GrayImage) * 1.65, 199)  # 二值化阈值
            ret, image = cv2.threshold(GrayImage, mea_thres, 255, cv2.THRESH_BINARY)

            contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) == 0:
                std_score = max(std_score - 8, 80)
                return std_score
            max_i = 0  
            max_len = -1
            for i in range(len(contours)):
                min_x = 100000
                max_x = -1
                for j in range(len(contours[i])):
                    if contours[i][j][0][0] < min_x:
                        min_x = contours[i][j][0][0]
                    if contours[i][j][0][0] > max_x:
                        max_x = contours[i][j][0][0]
                if (max_x - min_x) > max_len:
                    max_len = max_x - min_x
                    max_i = i

            rect = cv2.minAreaRect(contours[max_i])  # return (x,y),w,h,angle
            if rect[0][0] / (xr - xl) > 0.4 and rect[0][0] / (xr - xl) < 0.6:  
                std_score += 1
            else:
                std_score = max(std_score - 1, 80)
            if max(rect[1][0], rect[1][1]) / (xr - xl) > 0.7:  
                std_score += 1
            else:
                std_score = max(std_score - 1, 80)
            # 长宽比
            if rect[1][0] > 0 and rect[1][1] > 0 and max(rect[1][0], rect[1][1]) / min(rect[1][0], rect[1][1]) > 4:
                std_score += 1
            else:
                std_score = max(std_score - 1, 80)
            if rect[0][1] / (yd - yt) > 0.35 and rect[0][1] / (yd - yt) < 0.65:  
                std_score += 1
            else:
                std_score = max(std_score - 1, 80)
            # if x / (xr - xl) <= 0.25 and (x + w) / (xr - xl) >= 0.75:
            # std_score = std_score + 6
            # std_score += mea_thres % 100 / 20
            # brcnt = np.array([[[x, y]], [[x + w, y]], [[x + w, y + h]], [[x, y + h]]])
            # image = cv2.drawContours(crop, [brcnt], -1, (0, 0, 255), 2)
            # path = '/home/ultrasonic/zhoujun/mywork/quality_control_system/jizhutest/test_result/'
            # image_name = 1
            # while 1:
            #     if os.path.exists(path+str(image_name)+'.jpg'):
            #         image_name += 1
            #     else:
            #         break
            # cv2.imwrite(path+str(image_name)+'.jpg', image)
            # else:
            # std_score = max(std_score - 3, 80)
        if std_score > 99:
            std_score = 99

        if ('femur' in parts_found or 'humerus' in parts_found) and full_bone == 2 and std_score < 80:
            if 'femur' in parts_found:
                std_score = 80 + (gh1 + gh2) + parts_found['femur']
            if 'humerus' in parts_found:
                std_score = 80 + (gh1 + gh2) + parts_found['humerus']

        return std_score

    def description(self, scores, labels, boxes, image, plane_type, wrong_type=0):
        """
        docstring
        """

        auto_description = []
        auto_reason = []

        parts_found = {}
        label_list = []
        score_list = []

        std_info = ''
        std_score = 0
        for label, score, box in zip(labels, scores, boxes):
            # if label  in [4, 5, 10, 11]:
            #     continue

            part = self.get_annotation_name(label)

            score_list.append(score)
            label_list.append(part)

            if part.endswith('standard') or part.endswith('STD') or part.endswith('std'):
                part, std_info = self.get_std_info(part)

            # temp2 = '{' + '"name": "{5}", "vertex": "{0}, {1}, {2}, {3}", "score": {4}'.format(
            #     box[0], box[1], box[2], box[3], score, part) + '}'
            vertex_list = [str(x) for x in box]
            auto_description.append({
                'name': part,
                'vertex': ','.join(vertex_list),  # in order to be compatible with app
                'score': score
            })

            if part in parts_found:
                if score > parts_found[part]:

                    if part in self.parts_weight_mapping:
                        std_score += (score - parts_found[part]) * self.parts_weight_mapping[part]

                    parts_found[part] = score
            else:
                parts_found[part] = score
                # score
                if part in self.parts_weight_mapping:
                    std_score += score * self.parts_weight_mapping[part]

        # description process
        if plane_type == 'femur long-axis plane':
            if 'femur' not in parts_found:
                auto_reason.append('The femur is not clearly visible.')
            if "PEF" not in parts_found:
                auto_reason.append('The epiphysis at the proximal end of the femur is not clearly visible.')
            if 'DEF' not in parts_found:
                auto_reason.append('The epiphysis at the distal end of the femur is not clearly visible.')
        if plane_type == 'humerus long-axis plane':
            if 'humerus' not in parts_found:
                auto_reason.append('The humerus is not clearly visible.')
            if 'PEH' not in parts_found:
                auto_reason.append('The epiphysis at the proximal end of the humerus is not clearly visible.')
            if 'DEH' not in parts_found:
                auto_reason.append('The epiphysis at the distal end of the humerus is not clearly visible.')

        return auto_description, auto_reason, std_info, std_score, parts_found

    def get_coords_distance(self, boxes):
        """
        docstring
        """

        minDist = 1000000
        for i in range(len(boxes)):
            x1, y1 = (boxes[i][0] + boxes[i][2]) / 2, (boxes[i][1] + boxes[i][3]) / 2
            for j in range(i + 1, len(boxes)):
                x2, y2 = (boxes[j][0] + boxes[j][2]) / 2, (boxes[j][1] + boxes[j][3]) / 2
                dictance = math.sqrt(math.pow(x1 - x2, 2) + math.pow(y1 - y2, 2))
                if dictance < minDist:
                    minDist = dictance
                    xi, yi = i, j

        return minDist, xi, yi


    def confidence_score(self, scores_list, labels_list, boxes_list, std_info, std_score, parts_found, label_name_list):
        # def get_confidence(self, image_labels, image_scores, image_boxes, wrong_type=0):
        """
        Additional auxiliary treatment
        """

        parts_weight_mapping = {'femur': 35, "PEF": 25, 'DEF': 25, 'femur contour': 15, 'GUIN': 0, 'GUOUT': 0,
                                'humerus': 35, 'PEH': 25, 'DEH': 25, 'humerus contour': 15, 'GOIN': 0, 'GOOUT': 0,
                                'HAND': 0, 'FOOT': 0}

        conf_score = 0.40

        for i in range(len(labels_list)):
            conf_score += scores_list[i] * parts_weight_mapping.get(self.class_to_name[labels_list[i]], 0)

        if conf_score > 100:
            conf_score = 100
        elif conf_score < 0:
            conf_score = 0

        conf_score = self.normalize_score(conf_score, 15)
        # print("conf_score:", conf_score)

        return conf_score

    def compute_avg_confidence(self, boxes_list, scores_list, labels_list):
        average_confidence = np.mean(scores_list) if len(scores_list) > 0 else 0
        return average_confidence

    def updete_by_plane(self, boxes_list, scores_list, labels_list, plane_type):
        boxes_list = list(boxes_list)
        scores_list = list(scores_list)
        labels_list = list(labels_list)
        length = len(labels_list)
        if plane_type == 'femur long-axis plane':
            # print('type:gugu')
            i = 0
            while i < length:
                if labels_list[i] != 0 and labels_list[i] != 1 and labels_list[i] != 2 and labels_list[i] != 3 and \
                        labels_list[i] != 4 and labels_list[i] != 5:
                    del labels_list[i]
                    del boxes_list[i]
                    del scores_list[i]
                    i -= 1
                    length = len(labels_list)
                i += 1

        if plane_type == 'humerus long-axis plane':
            # print('type:gonggu')
            i = 0
            while i < length:
                if labels_list[i] != 6 and labels_list[i] != 7 and labels_list[i] != 8 and labels_list[i] != 9 and \
                        labels_list[i] != 10 and labels_list[i] != 11:
                    # print('label:',labels_list)
                    del labels_list[i]
                    del boxes_list[i]
                    del scores_list[i]
                    i -= 1
                    length = len(labels_list)
                i += 1
        boxes_list = np.array(boxes_list)
        scores_list = np.array(scores_list)
        labels_list = np.array(labels_list)
        return boxes_list, scores_list, labels_list

    def pre_deal(self, boxes_list, scores_list, labels_list):
        boxes_list = list(boxes_list)
        scores_list = list(scores_list)
        labels_list = list(labels_list)

        i_bone = -1
        for j in range(len(labels_list)):
            if labels_list[j] == 0 or labels_list[j] == 6:
                i_bone = j
                break
        if i_bone == -1:
            boxes_list = np.array(boxes_list)
            scores_list = np.array(scores_list)
            labels_list = np.array(labels_list)
            return boxes_list, scores_list, labels_list
        i = 0
        len_ = len(labels_list)
        while i < len_:
            if not math_util.is_box_intersect(boxes_list[i_bone], boxes_list[i]):
                del boxes_list[i]
                del scores_list[i]
                del labels_list[i]
                i -= 1
                len_ = len(labels_list)
                i_bone = -1
                for j in range(len_):
                    if labels_list[j] == 0 or labels_list[j] == 6:
                        i_bone = j
                        break
                if i_bone == -1:
                    boxes_list = np.array(boxes_list)
                    scores_list = np.array(scores_list)
                    labels_list = np.array(labels_list)
                    return boxes_list, scores_list, labels_list
            i += 1

        boxes_list = np.array(boxes_list)
        scores_list = np.array(scores_list)
        labels_list = np.array(labels_list)

        return boxes_list, scores_list, labels_list

    @classmethod
    def get_vertex_list(cls, vertex):
        if isinstance(vertex, list):
            vertex = [x for v in vertex for x in v]
        else:
            vertex = [float(i) for i in vertex.split(',')]
        return vertex


    def get_center_vertex(self, plane_result, names):
        center_vertex = -1
        if not plane_result:
            return center_vertex

        for one_result in plane_result:
            if isinstance(one_result, str):
                one_result = json.loads(one_result)
            if one_result['name'] in names:
                vertexs_bone = self.get_vertex_list(one_result['vertex'])
                center_vertex = [vertexs_bone[0] + (vertexs_bone[2] - vertexs_bone[0]) / 2,
                                 vertexs_bone[1] + (vertexs_bone[3] - vertexs_bone[1]) / 2]
                break
        return center_vertex


    def get_box_width(self, plane_result, names):
        center_vertex = -1
        box_width = -1
        if not plane_result:
            return center_vertex
        for one_result in plane_result:
            if isinstance(one_result, str):
                one_result = json.loads(one_result)
            if one_result['name'] in names:
                vertexs_bone = self.get_vertex_list(one_result['vertex'])
                box_width = vertexs_bone[2] - vertexs_bone[0]
                break

        return box_width


    def get_bone_contour_area(self, plane_result):
        bone_contour_area = -1
        if not plane_result:
            return bone_contour_area

        for one_result in plane_result:
            # one_result = json.loads(one_result)
            if one_result['name'] == 'humerus contour' or one_result['name'] == 'femur contour':
                vertexs_bone = self.get_vertex_list(one_result['vertex'])
                bone_contour_area = (vertexs_bone[2] - vertexs_bone[0]) * (vertexs_bone[3] - vertexs_bone[1])
                break
        return bone_contour_area


    def get_bone_area(self, plane_result):
        bone_area = -1
        vertexs_bone = -1
        vertexs_yuan = -1
        vertexs_jin = -1
        for one_result in plane_result:
            one_result = json.loads(one_result)
            if one_result['name'] == 'humerus' or one_result['name'] == 'femur':
                vertexs_bone = self.get_vertex_list(one_result['vertex'])
                bone_area = (vertexs_bone[2] - vertexs_bone[0]) * (vertexs_bone[3] - vertexs_bone[1])
            elif one_result['name'] == "PEF" or one_result['name'] == 'PEH':
                vertexs_jin = self.get_vertex_list(one_result['vertex'])
            elif one_result['name'] == 'DEF' or one_result['name'] == 'DEH':
                vertexs_yuan = self.get_vertex_list(one_result['vertex'])
        if vertexs_bone != -1 and vertexs_jin != -1 and vertexs_yuan != -1:
            intersection1 = (min(vertexs_bone[2], vertexs_jin[2]) - max(vertexs_bone[0], vertexs_jin[0])) \
                * (min(vertexs_bone[3], vertexs_jin[3]) - max(vertexs_bone[1], vertexs_jin[1]))
            proportion1 = intersection1 / ((vertexs_jin[2] - vertexs_jin[0]) * (vertexs_jin[3] - vertexs_jin[1]))
            # print(f'jin & bone proportion:{proportion1}')
            intersection2 = (min(vertexs_bone[2], vertexs_yuan[2]) - max(vertexs_bone[0], vertexs_yuan[0])) \
                * (min(vertexs_bone[3], vertexs_yuan[3]) - max(vertexs_bone[1], vertexs_yuan[1]))
            proportion2 = intersection2 / ((vertexs_yuan[2] - vertexs_yuan[0]) * (vertexs_yuan[3] - vertexs_yuan[1]))
            # print(f'yuan & bone proportion:{proportion2}')
            if proportion1 > 0.9 or proportion2 > 0.9:
                return -1
            elif proportion1 > 0.1 or proportion2 > 0.1:
                return bone_area
        else:
            return -1

    def get_pre_std_plane_info(self, history_queue):
        result = {}

        pregu, pregong, pregu1, pregong1 = -1, -1, -1, -1
        std_info_list = history_queue.std_info_list
        length = len(std_info_list)
        if length < 1:
            return result

        index = length - 1
        while index >= 0:
            if len(result) == 4:
                break
            type_id = std_info_list[index].auto_type
            if type_id == self.gu_type_id and pregu == -1:
                pregu = 1
                result['pregu'] = std_info_list[index]
            elif type_id == self.gu1_type_id and pregu1 == -1:
                pregu1 = 1
                result['pregu1'] = std_info_list[index]
            elif type_id == self.gong_type_id and pregong == -1:
                pregong = 1
                result['pregong'] = std_info_list[index]
            elif type_id == self.gong1_type_id and pregong1 == -1:
                pregong1 = 1
                result['pregong1'] = std_info_list[index]
            index -= 1
        return result

    def change_plane_type(self, plane_type, description):
        new_plane_type = 'femur long-axis plane' if plane_type == 'humerus long-axis plane' else 'humerus long-axis plane'
        # print(f'change plane type from {plane_type} to {new_plane_type}')
        if plane_type == 'femur long-axis plane':
            for info in description:
                if isinstance(info, str):
                    info = json.loads(info)
                if info['name'] == 'femur':
                    info['name'] = 'humerus'
                elif info['name'] == "PEF":
                    info['name'] = 'PEH'
                elif info['name'] == 'DEF':
                    info['name'] = 'DEH'
                elif info['name'] == 'femur contour':
                    info['name'] = 'humerus contour'
        elif plane_type == 'humerus long-axis plane':
            for info in description:
                if isinstance(info, str):
                    info = json.loads(info)
                if info['name'] == 'humerus':
                    info['name'] = 'femur'
                elif info['name'] == 'PEH':
                    info['name'] = "PEF"
                elif info['name'] == 'DEH':
                    info['name'] = 'DEF'
                elif info['name'] == 'humerus contour':
                    info['name'] = 'femur contour'
        return new_plane_type, description

    def change_plane_type_by_history_info(self, cur_plane_type, cur_description):
        length = len(self.history_queue)
        if length < 2:
            return cur_plane_type, cur_description

        pre_std_fl_hl_dict = self.get_pre_std_plane_info(self.history_queue)
        if not pre_std_fl_hl_dict:
            return cur_plane_type, cur_description

        # pre_std_fl_hl_dict_keys = pre_std_fl_hl_dict.keys()
        frame_idx_distance_cond = 100
        center_distance_cond = 80
        pregong1_score, pregong_score = 0, 0
        pregong1_count, pregong_count = 0, 0
        pregu1_score, pregu_score = 0, 0
        pregu_count, pregu1_count = 0, 0
        if cur_plane_type == 'femur long-axis plane':
            if 'pregong' in pre_std_fl_hl_dict or 'pregong1' in pre_std_fl_hl_dict:  
                pregong_frame_idx_distance = 9999999
                pregong1_frame_idx_distance = 9999999
                if 'pregong' in pre_std_fl_hl_dict:
                    pregong_score = pre_std_fl_hl_dict['pregong'].auto_score
                    pregong_count = self.history_queue.plane_type_count(plane_type=101)
                    pregong_frame_idx_distance = abs(pre_std_fl_hl_dict['pregong'].frame_idx - self.frame_idx)
                if 'pregong1' in pre_std_fl_hl_dict:
                    pregong1_score = pre_std_fl_hl_dict['pregong1'].auto_score
                    pregong1_count = self.history_queue.plane_type_count(plane_type=-101)
                    pregong1_frame_idx_distance = abs(pre_std_fl_hl_dict['pregong1'].frame_idx - self.frame_idx)
                min_gong_frame_idx_distance = min(pregong_frame_idx_distance, pregong1_frame_idx_distance)
                if min_gong_frame_idx_distance == pregong1_frame_idx_distance:
                    pre_score = pregong1_score
                    pre_count = pregong1_count
                else:
                    pre_score = pregong_score
                    pre_count = pregong_count
                if min_gong_frame_idx_distance < 60 and pre_score >= 70 and pre_count > 15:  
                    cur_plane_type, cur_description = self.change_plane_type(cur_plane_type, cur_description)
            elif self.bone_orientation == 1 and 'pregu1' in pre_std_fl_hl_dict:  
                cur_center_vertex = self.get_center_vertex(cur_description, ['humerus contour', 'femur contour'])
                pre_center_vertex = self.get_center_vertex(
                    pre_std_fl_hl_dict['pregu1'].detection_results['annotations'], ['humerus contour', 'femur contour'])
                if cur_center_vertex != -1 and pre_center_vertex != -1:
                    distance = math.sqrt((cur_center_vertex[0] - pre_center_vertex[0]) ** 2 + (
                        cur_center_vertex[1] - pre_center_vertex[1]) ** 2)
                    frame_idx_distance = abs(pre_std_fl_hl_dict['pregu1'].frame_idx - self.frame_idx)
                    if distance < center_distance_cond and frame_idx_distance < frame_idx_distance_cond:
                        self.bone_orientation = 2

        elif cur_plane_type == 'humerus long-axis plane':
            if 'pregu' in pre_std_fl_hl_dict or 'pregu1' in pre_std_fl_hl_dict:  
                pregu_frame_idx_distance = 9999999
                pregu1_frame_idx_distance = 9999999
                if 'pregu' in pre_std_fl_hl_dict:
                    pregu_score = pre_std_fl_hl_dict['pregu'].auto_score
                    pregu_count = self.history_queue.plane_type_count(plane_type=100)
                    pregu_frame_idx_distance = abs(pre_std_fl_hl_dict['pregu'].frame_idx - self.frame_idx)
                if 'pregu1' in pre_std_fl_hl_dict:
                    pregu1_score = pre_std_fl_hl_dict['pregu1'].auto_score
                    pregu1_count = self.history_queue.plane_type_count(plane_type=-100)
                    pregu1_frame_idx_distance = abs(pre_std_fl_hl_dict['pregu1'].frame_idx - self.frame_idx)
                min_gu_frame_idx_distance = min(pregu_frame_idx_distance, pregu1_frame_idx_distance)
                if min_gu_frame_idx_distance == pregu1_frame_idx_distance:
                    pre_score = pregu1_score
                    pre_count = pregu1_count
                else:
                    pre_score = pregu_score
                    pre_count = pregu_count
                if min_gu_frame_idx_distance < 60 and pre_score >= 70 and pre_count > 15:
                    cur_plane_type, cur_description = self.change_plane_type(cur_plane_type, cur_description)
            elif self.bone_orientation == 1 and 'pregong1' in pre_std_fl_hl_dict:  
                cur_center_vertex = self.get_center_vertex(cur_description, ['humerus contour', 'femur contour'])
                pre_center_vertex = self.get_center_vertex(
                    pre_std_fl_hl_dict['pregong1'].detection_results['annotations'], ['humerus contour', 'femur contour'])
                if cur_center_vertex != -1 and pre_center_vertex != -1:
                    distance = math.sqrt((cur_center_vertex[0] - pre_center_vertex[0]) ** 2 + (
                        cur_center_vertex[1] - pre_center_vertex[1]) ** 2)
                    frame_idx_distance = abs(pre_std_fl_hl_dict['pregong1'].frame_idx - self.frame_idx)
                    if distance < center_distance_cond and frame_idx_distance < frame_idx_distance_cond:
                        self.bone_orientation = 2
            elif self.bone_orientation == 2 and 'pregong' in pre_std_fl_hl_dict: 
                cur_center_vertex = self.get_center_vertex(cur_description, ['humerus contour', 'femur contour'])
                pre_center_vertex = self.get_center_vertex(
                    pre_std_fl_hl_dict['pregong'].detection_results['annotations'], ['humerus contour', 'femur contour'])
                if cur_center_vertex != -1 and pre_center_vertex != -1:
                    distance = math.sqrt((cur_center_vertex[0] - pre_center_vertex[0]) ** 2 + (
                        cur_center_vertex[1] - pre_center_vertex[1]) ** 2)
                    frame_idx_distance = abs(pre_std_fl_hl_dict['pregong'].frame_idx - self.frame_idx)
                    if distance < center_distance_cond and frame_idx_distance < frame_idx_distance_cond:
                        self.bone_orientation = 1

        return cur_plane_type, cur_description

    def description_and_score(self, boxes_list, scores_list, labels_list, roi_list, image=None, wrong_type=0):
        # classify gugu or gonggu
        guguNum, gongguNum = 0, 0
        if wrong_type == 0:
            boxes_list, scores_list, labels_list = self.pre_deal(boxes_list, scores_list, labels_list)
            is_out = 0
            is_other = 0
            for ind in labels_list:
                if ind < 6:               # in [0, 1, 2, 3, 4, 5]:
                    guguNum += 1
                    if ind == 5:
                        is_out = 2
                    elif ind == 4:
                        is_out = 1
                elif ind < 12:            # in [6, 7, 8, 9, 10, 11]:
                    gongguNum += 1
                    if ind == 11:
                        is_out = 2
                    elif ind == 10:
                        is_out = 1
                elif ind == 12 or ind == 13:
                    is_other = 1

            if is_other == 1:
                plane_type = "others"
            elif guguNum >= gongguNum:
                plane_type = "femur long-axis plane"
            else:
                plane_type = "humerus long-axis plane"

            if is_out == 2:
                self.bone_orientation = 2
            else:
                self.bone_orientation = 1
        else:
            plane_type = wrong_type

        extra_score = 10 * min(gongguNum, guguNum) / max(1, max(gongguNum, guguNum))
        # Based on the section correction frame
        boxes_list, scores_list, labels_list = self.updete_by_plane(boxes_list, scores_list, labels_list, plane_type)
        # Correct the extremely inaccurate position of the frame
        # boxes_list, scores_list, labels_list = self.get_location(boxes_list, scores_list, labels_list, plane_type)
        label_name_list = []
        for label in labels_list:
            label_name_list.append(self.get_annotation_name(label))

        description, reason, std_info, std_score, parts_found = \
            self.description(scores_list, labels_list, boxes_list, image, plane_type, wrong_type)

        if self.history_queue is not None:
            plane_type, description = self.change_plane_type_by_history_info(plane_type, description)

        auto_score = self.calculate_qc_score(std_info, std_score, parts_found, boxes_list, scores_list, image,
                                             plane_type)

        confidence = self.confidence_score(scores_list, labels_list, boxes_list, std_info, std_score, parts_found,
                                           label_name_list)
        auto_score -= extra_score
        if plane_type == 'others':
            auto_score = 0
        elif auto_score < 0:
            auto_score = 10.1
        # print('pre_pic:', BoneModel.PrePic)

        # if self.is_doubleside():
        #     plane_type += '(1)'
        for desc in description:
            if desc['name'] in ['GUIN', 'GUOUT', 'GOIN', 'GOOUT']:
                description.remove(desc)
        return plane_type, auto_score, description, reason, confidence

    def is_doubleside(self, auto_type=0):
        return self.bone_orientation == 2

    def get_location(self, boxes_list, scores_list, labels_list, plane_type):
        if plane_type == 'femur long-axis plane':
            gugu = -1
            gugjd = -1
            gugyd = -1
            for i in range(len(labels_list)):
                if labels_list[i] == 0:
                    gugu = i
                if labels_list[i] == 1:
                    gugjd = i
                if labels_list[i] == 2:
                    gugyd = i
            if gugu == -1:
                return boxes_list, scores_list, labels_list
            if gugjd != -1:
                if not math_util.is_box_intersect(boxes_list[gugu], boxes_list[gugjd]):
                    labels_list = np.delete(labels_list, gugjd, axis=0)
                    boxes_list = np.delete(boxes_list, gugjd, axis=0)
                    scores_list = np.delete(scores_list, gugjd, axis=0)

            for i in range(len(labels_list)):
                if labels_list[i] == 0:
                    gugu = i
                if labels_list[i] == 1:
                    gugjd = i
                if labels_list[i] == 2:
                    gugyd = i
            if gugyd != -1:
                if not math_util.is_box_intersect(boxes_list[gugu], boxes_list[gugyd]):
                    labels_list = np.delete(labels_list, gugyd, axis=0)
                    boxes_list = np.delete(boxes_list, gugyd, axis=0)
                    scores_list = np.delete(scores_list, gugyd, axis=0)

            gugu = -1
            gugjd = -1
            gugyd = -1
            for i in range(len(labels_list)):
                if labels_list[i] == 0:
                    gugu = i
                if labels_list[i] == 1:
                    gugjd = i
                if labels_list[i] == 2:
                    gugyd = i
            if gugyd != -1 and gugjd != -1:
                if math_util.is_box_intersect(boxes_list[gugyd], boxes_list[gugjd]):
                    if scores_list[gugjd] > scores_list[gugyd]:
                        labels_list = np.delete(labels_list, gugyd, axis=0)
                        boxes_list = np.delete(boxes_list, gugyd, axis=0)
                        scores_list = np.delete(scores_list, gugyd, axis=0)
                    else:
                        labels_list = np.delete(labels_list, gugjd, axis=0)
                        boxes_list = np.delete(boxes_list, gugjd, axis=0)
                        scores_list = np.delete(scores_list, gugjd, axis=0)
            return boxes_list, scores_list, labels_list
        if plane_type == 'humerus long-axis plane':
            gogu = -1
            gogjd = -1
            gogyd = -1
            for i in range(len(labels_list)):
                if labels_list[i] == 4:
                    gogu = i
                if labels_list[i] == 5:
                    gogjd = i
                if labels_list[i] == 6:
                    gogyd = i
            if gogu == -1:
                return boxes_list, scores_list, labels_list
            if gogjd != -1:
                if not math_util.is_box_intersect(boxes_list[gogu], boxes_list[gogjd]):
                    labels_list = np.delete(labels_list, gogjd, axis=0)
                    boxes_list = np.delete(boxes_list, gogjd, axis=0)
                    scores_list = np.delete(scores_list, gogjd, axis=0)

            for i in range(len(labels_list)):
                if labels_list[i] == 0:
                    gogu = i
                if labels_list[i] == 1:
                    gogjd = i
                if labels_list[i] == 2:
                    gogyd = i
            if gogyd != -1:
                if not math_util.is_box_intersect(boxes_list[gogu], boxes_list[gogyd]):
                    labels_list = np.delete(labels_list, gogyd, axis=0)
                    boxes_list = np.delete(boxes_list, gogyd, axis=0)
                    scores_list = np.delete(scores_list, gogyd, axis=0)

            gogu = -1
            gogjd = -1
            gogyd = -1
            for i in range(len(labels_list)):
                if labels_list[i] == 0:
                    gogu = i
                if labels_list[i] == 1:
                    gogjd = i
                if labels_list[i] == 2:
                    gogyd = i
            if gogyd != -1 and gogjd != -1:
                if math_util.is_box_intersect(boxes_list[gogyd], boxes_list[gogjd]):
                    if scores_list[gogjd] > scores_list[gogyd]:
                        labels_list = np.delete(labels_list, gogyd, axis=0)
                        boxes_list = np.delete(boxes_list, gogyd, axis=0)
                        scores_list = np.delete(scores_list, gogyd, axis=0)
                    else:
                        labels_list = np.delete(labels_list, gogjd, axis=0)
                        boxes_list = np.delete(boxes_list, gogjd, axis=0)
                        scores_list = np.delete(scores_list, gogjd, axis=0)
            return boxes_list, scores_list, labels_list

    def deal_seqpic(self, plane_type, score, is_out):
        re_score = score
        flag1 = 0
        tempPic = [plane_type, score, self.frame_idx]
        lenPreQue = len(BoneModel.PrePic)
        if lenPreQue != 0 and self.frame_idx < BoneModel.PrePic[lenPreQue - 1][2]:
            BoneModel.PrePic = []
        if len(BoneModel.PrePic) < 5:
            BoneModel.PrePic.append(tempPic)
        else:
            del BoneModel.PrePic[0]
            BoneModel.PrePic.append(tempPic)

        lenPreQue = len(BoneModel.PrePic)

        for i in range(lenPreQue):
            if "femur long-axis plane" in BoneModel.PrePic[i]:
                flag1 += 1
            if "humerus long-axis plane" in BoneModel.PrePic[i]:
                flag1 -= 1
        if flag1 != -5 and flag1 != 5:
            re_score = min(re_score, 82) 
        if lenPreQue < 5 or BoneModel.PrePic[lenPreQue - 1][2] - BoneModel.PrePic[0][2] > 50:
            re_score = min(re_score, 82)

        return re_score

    def add_score_by_history(self):
        min_score = 60.1

        length = len(self.history_queue)
        if length < 2:
            return 0

        cur_info = self.history_queue[-1]
        cur_description = cur_info.annotations

        cur_bone_area = self.get_bone_contour_area(cur_description)
        cur_plane_type = cur_info.auto_type
        cur_auto_score = cur_info.auto_score

        if cur_auto_score < min_score:
            return 0

        pre_info = self.history_queue[-2]
        pre_video_score = pre_info.video_score

        cur_center_vertex = self.get_center_vertex(cur_description, ['humerus contour', 'femur contour'])
        pre_plane_type = pre_info.auto_type

        check_num_20 = 20
        check_num_60 = 60
        history_bone_area_sum = 0.
        history_bone_sum = 0.
        history_bone_change = []

        change_score = 0.

        fl_hl_index_distance = 0
        index = length - 2

        cur_box_width = self.get_box_width(cur_description, ['humerus contour', 'femur contour'])
        pre_box = self.get_pre_std_plane_info(self.history_queue)
        for index_name in pre_box:
            if pre_box[index_name].auto_type == cur_info.auto_type:
                pre_box_width = self.get_box_width(pre_box[index_name].annotations, ['humerus contour', 'femur contour'])
                pre_score = pre_box[index_name].auto_score
                cur_score = cur_info.auto_score
                if abs(pre_score - cur_score) <= 3 and (cur_box_width / pre_box_width) >= 1.1:
                    lagersize_add_score = 2 * (1-(abs(pre_score - cur_score))/3)
                    change_score += lagersize_add_score

        while index > 0 and (check_num_20 > 0 or check_num_60 > 0):

            cur_info = self.history_queue[index]
            plane_type = cur_info.auto_type
            plane_score = max(cur_info.auto_score, 50)

            plane_description = cur_info.annotations
            plane_center_vertex = self.get_center_vertex(plane_description, ['humerus contour', 'femur contour'])

            distance = -1  

            if plane_type not in self.hl_fl_plane_type_ids:
                fl_hl_index_distance += 1
            elif plane_type in self.hl_fl_plane_type_ids and fl_hl_index_distance < 4:
                fl_hl_index_distance = 0


            if check_num_20 > 0:
                if plane_type in self.minus_plane_type_ids and check_num_20 > 0:
                    minus_score = self.minus_base_score * (index / length) * (plane_score / 100)
                    change_score -= minus_score
                if cur_plane_type != plane_type:
                    if cur_center_vertex != -1 and plane_center_vertex != -1:
                        distance = math.sqrt((cur_center_vertex[0] - plane_center_vertex[0]) ** 2 + (
                            cur_center_vertex[1] - plane_center_vertex[1]) ** 2)
                    if cur_plane_type in [self.gu_type_id, self.gu1_type_id]:
                        if plane_type in [self.gong_type_id, self.gong1_type_id] or \
                                (plane_type in [self.gu_type_id, self.gu1_type_id] and (distance < 100 or distance == -1)):
                            minus_score = 5 * (index / length) * (plane_score / 100)
                            change_score -= minus_score
                    elif cur_plane_type in [self.gong_type_id, self.gong1_type_id]:
                        if plane_type in [self.gu_type_id, self.gu1_type_id] or \
                                (plane_type in [self.gong_type_id, self.gong1_type_id] and (distance < 100 or distance == -1)):
                            minus_score = 5 * (index / length) * (plane_score / 100)
                            change_score -= minus_score
            if check_num_60 > 0:
                if plane_score > 80 and plane_type == cur_plane_type and plane_type in self.hl_fl_plane_type_ids:
                    if cur_bone_area != -1 and fl_hl_index_distance < 4:
                        one_plane_result = plane_description
                        if one_plane_result is not None and len(one_plane_result) != 0:
                            history_bone_area = self.get_bone_contour_area(one_plane_result)
                            if history_bone_area != -1:
                                if len(history_bone_change) == 0:
                                    history_bone_change.append(history_bone_area)
                                else:
                                    last_history_bone_change = history_bone_change[0]
                                    if abs(history_bone_area / last_history_bone_change - 1) > 0.2:
                                        history_bone_sum = 0.
                                        history_bone_area_sum = 0.
                                        # print(f'before append:{history_bone_change}')
                                        history_bone_change.insert(0, history_bone_area)
                                        # print(f'after append:{history_bone_change}')
                                    else:
                                        history_bone_sum += 1
                                        history_bone_area_sum += history_bone_area
                                        # print(f'before change last:{history_bone_change}')
                                        history_bone_change[0] = history_bone_area_sum / history_bone_sum
                                        # print(f'after change last:{history_bone_change}')

            index -= 1
            check_num_20 -= 1
            check_num_60 -= 1
        if cur_bone_area != -1 and len(history_bone_change) != 0:
            # print(f'history_bone_change:{history_bone_change}')
            bone_proportion = cur_bone_area / history_bone_change[0]
            # print(f'bone_proportion1 :{bone_proportion}')
            if bone_proportion > 1.5:
                extra_score = math.log2(bone_proportion * 16)
                change_score += extra_score
                if extra_score > self.enlarge_extra_score:
                    self.enlarge_extra_score = extra_score
                # print(f'{self.frame_idx} enlarge extra_score:{extra_score}')
            elif self.enlarge_extra_score > 0:
                if bone_proportion < 0.7:  # 缩小
                    self.enlarge_extra_score = 0
                else:
                    std_info = self.history_queue.last_std_info
                    if std_info:
                        pre_capture_type = std_info.auto_type
                        if (cur_plane_type in [self.gu_type_id, self.gu1_type_id] and
                            pre_capture_type in [self.gu_type_id, self.gu1_type_id]) or \
                                (cur_plane_type in [self.gong_type_id, self.gong1_type_id] and
                                    pre_capture_type in [self.gong_type_id, self.gong1_type_id]):
                            change_score += self.enlarge_extra_score
                        else:
                            self.enlarge_extra_score = 0

        max_score = 98.1
        if pre_plane_type is not None and pre_plane_type == cur_plane_type:
            max_score = max(max_score, pre_video_score + 0.05)

        if change_score > 0:
            change_score = min(change_score, max_score - cur_auto_score)
        elif change_score < 0:
            change_score = max(change_score, min_score - cur_auto_score)

        # print(f'{self.frame_idx} pre_plane_type:{pre_plane_type}, cur_plane_type:{cur_plane_type}')
        # print(f'{self.frame_idx} pre_video_score:{pre_video_score}, cur_video_score:{cur_auto_score + change_score}')

        if pre_plane_type is not None and pre_plane_type == cur_plane_type:
            if abs(cur_auto_score + change_score - pre_video_score) < 2:
                # print(f'add 0.01，same_plane_type_add_score:{self.same_plane_type_add_score}')
                self.same_plane_type_add_score += 0.01
                change_score += self.same_plane_type_add_score
        else:
            self.same_plane_type_add_score = 0.0

        return change_score
