import json
import os
import numpy as np
from .aes_cipher import AesCipher
from loguru import logger
import cv2


def cv2_imread(image_path):
    return cv2.imdecode(np.fromfile(image_path), cv2.IMREAD_COLOR)


def get_annotation_path(path, prefer_gt=False):
    if os.path.isdir(path):
        if prefer_gt:
            anno_path = os.path.join(path, 'gt_annotations.cjson')
            if os.path.exists(anno_path):
                return anno_path

            anno_path = os.path.join(path, 'gt_annotations.json')
            if os.path.exists(anno_path):
                return anno_path

        anno_path = os.path.join(path, 'annotations.cjson')
        if os.path.exists(anno_path):
            return anno_path

        anno_path = os.path.join(path, 'annotations.json')
        if os.path.exists(anno_path):
            return anno_path

        anno_path = os.path.join(path, 'annotations.csv')
        if os.path.exists(anno_path):
            return anno_path


def decipher_cjson(cjson_path):
    with open(cjson_path, 'r', encoding='utf-8') as fs:
        contents = fs.read()
        # decipher
        anno_json = AesCipher.decipher_text(contents)

        return anno_json


def load_annotations(annotation_path: str, polygon_to_box=False):
    """
    return: (img2anno, config) or None
    """
    if not os.path.exists(annotation_path):
        logger.error(f'annotation path does not exist: {annotation_path}')
        return

    if os.path.isdir(annotation_path):
        path = annotation_path
        annotation_path = get_annotation_path(annotation_path)
        if not annotation_path:
            logger.error(f'no annotation file exists: {path}')
            return

    # parse the annotation file
    _, ext = os.path.splitext(annotation_path)
    ext = ext.lower()
    if ext not in ('.cjson', '.json', '.csv'):
        logger.error(f'invalid annotation format: {ext}, only json or csv file is supported')
        return

    img2anno = None
    with open(annotation_path, 'r', encoding='utf-8') as fs:
        try:
            if ext == '.cjson':
                contents = fs.read()
                # decipher
                anno_json = AesCipher.decipher_text(contents)
                if not anno_json.startswith('{'):
                    logger.error(anno_json)
                else:
                    anno_json = json.loads(anno_json)
                    img2anno = parse_json(anno_json, polygon_to_box)
            elif ext == '.json':
                anno_json = json.load(fs)
                img2anno = parse_json(anno_json, polygon_to_box)
            else:
                img2anno = parse_csv(fs, polygon_to_box), {}
        except Exception:
            logger.error('illeagal annotation format')

    return img2anno


def parse_json(anno_json, polygon_to_box=False):
    """
    return (annotations, config)
    """
    config = anno_json['config'] if 'config' in anno_json else {}

    if 'annotations' not in anno_json:
        logger.error('invalid annotation file: does not contain annotations key')
        return

    image_annotations = anno_json['annotations']

    # normalize
    for image_name, image_anno in image_annotations.items():
        if 'annosets' in image_anno:
            for anno_set in image_anno['annosets']:
                normalize(anno_set, polygon_to_box)
        else:
            # convert to new format
            image_annotations[image_name] = {}
            new_image_anno = image_annotations[image_name]

            if 'id' in image_anno:
                new_image_anno['id'] = image_anno['id']
                del image_anno['id']

            if 'GA' in image_anno:
                new_image_anno['GA'] = image_anno['GA']
                del image_anno['GA']

            if 'ruler' in image_anno:
                new_image_anno['ruler'] = image_anno['ruler']
                del image_anno['ruler']

            if 'malformations' in image_anno:
                new_image_anno['malformations'] = image_anno['malformations']
                del image_anno['malformations']

            normalize(image_anno, polygon_to_box)
            new_image_anno['annosets'] = [image_anno]

    return image_annotations, config


def normalize(annoset, polygon_to_box):
    # category
    if 'image_type' not in annoset:
        if 'bodyPart' in annoset:
            annoset['image_type'] = annoset['bodyPart']
            del annoset['bodyPart']
        elif 'plane_type' in annoset:
            annoset['image_type'] = annoset['plane_type']
            del annoset['plane_type']
        else:
            annoset['image_type'] = ''

    if 'standard' not in annoset and 'std_type' in annoset:
        annoset['standard'] = annoset['std_type']
        del annoset['std_type']

    if 'subclass' in annoset and annoset['subclass'] == 'empty':
        annoset['subclass'] = ''

    if 'annotations' not in annoset:
        annoset['annotations'] = []

    for anno in annoset['annotations']:
        if 'vertex' in anno:
            # polygon
            if isinstance(anno['vertex'], str):
                # convert to vertex list
                vertex_list = anno['vertex'].split(';')
                vertex_list = [vertex.split(',') for vertex in vertex_list if vertex]
                anno['vertex'] = [(int(float(x) + 0.5), int(float(y) + 0.5)) for x, y in vertex_list]
            elif not polygon_to_box:
                break

            # bounding box
            if polygon_to_box and len(anno['vertex']) > 2:
                min_x, min_y = np.min(anno['vertex'], axis=0)
                max_x, max_y = np.max(anno['vertex'], axis=0)
                anno['vertex'] = [[min_x, min_y], [max_x, max_y]]
                # change type to be box
                anno['type'] = 2

        else:
            vertex_list = [anno['start'].split(','), anno['end'].split(',')]
            del anno['start']
            del anno['end']
            anno['vertex'] = [(int(float(x) + 0.5), int(float(y) + 0.5)) for x, y in vertex_list]


def parse_csv(fs, polygon_bo_box):
    image_annotations = {}
    for line in fs.readlines():
        line = line.strip()
        if len(line) == 0 or line.startswith('#'):
            continue

        items = line.split(',')
        if len(items) < 7:
            raise Exception('Invalid csv format: ' + line)

        if items[0] not in image_annotations:
            image_annotations[items[0]] = {
                'annosets': []
            }

        anno_set = image_annotations[items[0]]['annosets']
        if len(anno_set) == 0:
            anno_set.append({
                'image_type': items[5],
                'standard': items[6],
                'annotations': []
            })

        anno_set[0]['annotations'].append({
            'type': 2,
            'name': items[5] if len(items) < 8 or items[7] == 'Box' else items[7],
            'vertex': [[int(float(items[1]) + 0.5), int(float(items[2]) + 0.5)],
                       [int(float(items[3]) + 0.5), int(float(items[4]) + 0.5)]]
        })

        # normalize(anno_set, polygon_bo_box)
    return image_annotations


def save_annotations(output_path: str, image2anno):
    """
    output_path: if end_with cjson, encrypt json
    """
    if not output_path or not image2anno:
        return

    result_list = []
    for image_name, anno_set in image2anno.items():
        if not anno_set:
            continue

        # AnnotationSet
        if not isinstance(anno_set, (list, dict)):
            anno_set = anno_set.to_json_object()
        str_anno = json.dumps(anno_set, ensure_ascii=False)

        str_json = f'"{image_name}":{str_anno}'

        result_list.append(str_json)

    str_anno = '{"config": {}, "annotations": {\n' + ',\n'.join(result_list) + '\n}}'
    if output_path.endswith('.cjson'):
        str_anno = AesCipher.cipher_text(str_anno)

    with open(output_path, 'w', encoding='utf-8') as fs:
        fs.write(str_anno)


if __name__ == '__main__':
    # JSON的绝对路径
    path_root = r'C:\Users\guang\Desktop\measure-data\AFI_measure'
    path_root = r'H:\online data\wechat\WeChat Files\wxid_eg4uwsyd5ggs21\FileStorage\File\2024-08'
    # path = path_root + '/' + 'annotations.cjson'
    # path = path.replace('\\', '/')

    results = load_annotations(path_root)
    if not results:
        logger.error('failed to load annotations for: ' + path_root)
        exit(1)

    image2anno, config = results
    # 将cjson/csv中的数据保存为json

    # print(anno_set)
    anno = {
        "config": config,
        "annotations": image2anno
    }

    new_path = os.path.join(path_root, 'annotations.json')
    new_json = json.dumps(anno, ensure_ascii=False, indent=4)
    with open(new_path, 'w', encoding='utf-8') as f:
        f.write(new_json)
