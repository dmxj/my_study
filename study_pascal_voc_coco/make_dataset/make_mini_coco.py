# -*- coding: utf-8 -* -
'''
制作一个小的COCO数据集，目录结构参考：dataset/coco
'''
import os
import json
import shutil
import random

'''COCO数据集原始的位置'''
sourcce_coco_data_path = {
    "train": {
        "image_path": "/Users/rensike/Files/data/coco/coco2017/images/val2017",
        "anno_path": "/Users/rensike/Files/data/coco/coco2017/annotations/instances_val2017.json",
        "min_count": 10,
    },
    "val": {
        "image_path": "/Users/rensike/Files/data/coco/coco2017/images/val2017",
        "anno_path": "/Users/rensike/Files/data/coco/coco2017/annotations/instances_val2017.json",
        "min_count": 5,
    },
    "test": {
        "image_path": "/Users/rensike/Files/data/coco/coco2017/images/val2017",
        "anno_path": "/Users/rensike/Files/data/coco/coco2017/annotations/instances_val2017.json",
        "min_count": 8,
    },
}

'''COCO生成的迷你数据集的位置'''
dist_mini_coco_data_path = "/Users/rensike/Files/temp/coco_mini"
# 训练|验证|测试
set_list = list(sourcce_coco_data_path.keys())
# 构建迷你数据集的缩放比例
ratio = 1 / 50
# 是否随机选择图片
is_shuffle = True


def mkdir(dir_path):
    '''
    创建不存在的目录
    :param dir_path:
    :return:
    '''
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def make_coco_data_dir(coco_data_path):
    '''
    创建COCO数据集的目录结构
    :param coco_data_path:
    :return:
    '''
    mkdir(os.path.join(coco_data_path, "annotations"))
    for set_type in set_list:
        mkdir(os.path.join(coco_data_path, set_type))


def copy_files(set_type, image_name_list):
    '''
    拷贝愿图片到目标目录
    :param set_type:
    :param image_name_list:
    :return:
    '''
    source_image_path = sourcce_coco_data_path[set_type]["image_path"]
    dist_image_path = os.path.join(dist_mini_coco_data_path, set_type)

    for image_name in image_name_list:
        shutil.copy(os.path.join(source_image_path, image_name), os.path.join(dist_image_path, image_name))


def load_anno(anno_path):
    '''
    加载标注文件中的标注信息、图片信息、类别信息等
    :param anno_path:
    :return:
    '''
    with open(anno_path, "r") as fid:
        groundtruth_data = json.load(fid)

    anno_dict = dict()
    if 'annotations' in groundtruth_data:
        for annotation in groundtruth_data['annotations']:
            image_id = annotation["image_id"]
            anno_dict[image_id] = annotation

    image_dict = dict()
    if "images" in groundtruth_data:
        for image in groundtruth_data["images"]:
            image_dict[image["id"]] = image

    return anno_dict, image_dict, groundtruth_data["categories"]


def get_cls_info_from_anno(anno_dict):
    '''
    从标注信息中构建类别相关信息
    :param anno_dict:
    :return:
    '''
    cls_info = dict()
    for image_id in anno_dict:
        annotation = anno_dict[image_id]
        cat_id = annotation["category_id"]
        if cat_id not in cls_info:
            cls_info[cat_id] = {}
        image_id = annotation["image_id"]
        if image_id not in cls_info[cat_id]:
            cls_info[cat_id][image_id] = []
        cls_info[cat_id][image_id].append(annotation["id"])
    return cls_info


def make_mini_dataset():
    '''
    构建迷你数据集
    :return:
    '''
    # 标注文件 ==> 标注信息
    anno_map = {}
    cls_info_map = {}
    image_id_used = []
    for set_type in sourcce_coco_data_path:
        min_count = sourcce_coco_data_path[set_type]["min_count"]
        anno_path = sourcce_coco_data_path[set_type]["anno_path"]

        if anno_path not in anno_map:
            anno_map[anno_path] = load_anno(anno_path)

        anno_dict, image_dict, categories = anno_map[anno_path]

        if anno_path not in cls_info_map:
            cls_info_map[anno_path] = get_cls_info_from_anno(anno_dict)

        cls_info = cls_info_map[anno_path]
        image_id_select = []
        for cat_id in cls_info:
            image_id_list = list(cls_info[cat_id].keys())
            image_count = len(image_id_list)
            select_count = round(image_count * ratio)
            if select_count < min_count:
                select_count = min_count
                if min_count > image_count:
                    select_count = image_count

            image_id_list_diff = list(set(image_id_list).difference(set(image_id_used)))
            if len(image_id_list_diff) == 0:
                image_id_list_diff = image_id_list
            select_count = min(select_count, len(image_id_list_diff))
            if is_shuffle:
                image_id_select += random.sample(image_id_list_diff, select_count)
            else:
                image_id_select += image_id_list_diff[:select_count]

            print("类型：{} - 类别：{} - 数量：{}".format(set_type, cat_id, select_count))
        print("类型：{} - 总量：{}".format(set_type, len(image_id_select)))

        image_id_used += image_id_select

        res_anno_file_name = "instances_{}.json".format(set_type)
        res_anno_json = {
            "categories": categories,
            "images": [],
            "annotations": []
        }

        image_name_list = []
        for image_id in image_id_select:
            res_anno_json["images"].append(image_dict[image_id])
            res_anno_json["annotations"].append(anno_dict[image_id])
            image_name_list.append(image_dict[image_id]["file_name"])
        json.dump(res_anno_json, open(os.path.join(dist_mini_coco_data_path, "annotations", res_anno_file_name), 'w'))
        copy_files(set_type, image_name_list)


def main():
    '''
    主函数
    :return:
    '''
    make_coco_data_dir(dist_mini_coco_data_path)
    make_mini_dataset()


if __name__ == '__main__':
    main()
