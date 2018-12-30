# -*- coding: utf-8 -* -
'''
解析加载COCO格式标注信息或Pascal格式标注信息
'''
from pycocotools import mask
from utils import label_map_util
from utils import dataset_util
from lxml import etree
import json
import os
import pprint

def load_coco(annotations_file, image_dir):
    coco_ann_dict = dict()

    with open(annotations_file,"r") as fid:
        groundtruth_data = json.load(fid)
        images = groundtruth_data['images']
        image_dict = dict()
        for image in images:
            image_dict[image["id"]] = image

        category_index = label_map_util.create_category_index(
            groundtruth_data['categories'])
        if 'annotations' in groundtruth_data:
            for annotation in groundtruth_data['annotations']:
                image_id = annotation['image_id']
                if image_id not in coco_ann_dict:
                    coco_ann_dict[image_id] = []
                annotation["image_path"] = os.path.join(image_dir, image_dict[image_id]["file_name"])
                coco_ann_dict[image_id].append(annotation)
        missing_annotation_count = 0
        for image in images:
            image_id = image['id']
            if image_id not in coco_ann_dict:
                missing_annotation_count += 1
                coco_ann_dict[image_id] = []

    return coco_ann_dict

def load_pascal(data_dir,set,is_detect=True,is_seg=False):
    assert is_detect != is_seg

    annotations_dir = os.path.join(data_dir,"Annotations")
    image_dir = os.path.join(data_dir,"JPEGImages")

    pascal_ann_dict = dict()
    if is_detect:
        examples_path = os.path.join(data_dir, 'ImageSets', 'Main', set + '.txt')
    if is_seg:
        examples_path = os.path.join(data_dir, 'ImageSets', 'Segmentation', set + '.txt')
    examples_list = dataset_util.read_examples_list(examples_path)

    for example in examples_list:
        path = os.path.join(annotations_dir, example + '.xml')
        with open(path, 'r') as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']
        data["img_path"] = os.path.join(image_dir, data['filename'])
        pascal_ann_dict[data['filename']] = data

    return pascal_ann_dict

def load_pascal_single(anno_path,image_dir):
    with open(anno_path, 'r') as fid:
        xml_str = fid.read()
    xml = etree.fromstring(xml_str)
    data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']
    data["img_path"] = os.path.join(image_dir, data['filename'])
    return data

if __name__ == "__main__":
    coco_ann = load_coco("../dataset/coco/annotations/instances_val.json","../dataset/coco/annotations/val")
    pprint.pprint(coco_ann)

    # pascal_ann = load_pascal("../dataset/voc/VOC2012","train",True,False)
    # pprint.pprint(pascal_ann)


