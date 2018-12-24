# -*- coding: utf-8 -* -
'''
制作一个小的VOC数据集
'''
import os
import shutil
import mmcv
import random
from lxml import etree
from utils import dataset_util

'''VOC数据集原始地址'''
pascal_voc_source_data_path = "/Users/rensike/Files/data/voc/VOCdevkit2/VOC2012"
'''VOC生成的迷你数据集的位置'''
dist_mini_voc_data_path = "/Users/rensike/Files/temp/voc_mini"
# 相对于原来的数据削减的比例
det_ratio = 1 / 50
# 每一类的图片的最少值
min_det_train_count, min_det_val_count = 10, 8
# 是否包含分割图片
is_contain_seg = True
# 分割数据相对原来的数据消减的比例
seg_ratio = 1 / 30
# 分割每一类的图片的最少值
min_seg_train_count, min_seg_val_count = 6, 4
# 只构建train和val数据集
set_list = ["train", "val"]
# 是否随机选择数据
is_shuffle = True


def mkdir(dir_path):
    '''
    创建不存在的目录
    :param dir_path:
    :return:
    '''
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def make_voc_data_dir(voc_data_path):
    '''
    创建pascal voc的目录结构
    :param voc_data_path:
    :return:
    '''
    mkdir(voc_data_path)
    mkdir(os.path.join(voc_data_path, "Annotations"))
    mkdir(os.path.join(voc_data_path, "ImageSets"))
    mkdir(os.path.join(voc_data_path, "ImageSets/Main"))
    mkdir(os.path.join(voc_data_path, "JPEGImages"))

    if is_contain_seg:
        mkdir(os.path.join(voc_data_path, "ImageSets/Segmentation"))
        mkdir(os.path.join(voc_data_path, "SegmentationClass"))
        mkdir(os.path.join(voc_data_path, "SegmentationObject"))


def load_sample_list(set, isDet=True, isSeg=False):
    '''
    加载样本列表
    :param set: train|val|trainval|test
    :param isDet: 是否加载检测数据
    :param isSeg: 是否加载分割数据
    :return:
    '''
    assert isDet != isSeg
    sample_file = os.path.join(pascal_voc_source_data_path, "ImageSets/Main", "{}.txt".format(set))
    if isSeg:
        sample_file = os.path.join(pascal_voc_source_data_path, "ImageSets/Segmentation", "{}.txt".format(set))
    sample_list = mmcv.list_from_file(sample_file)
    return sample_list


def load_anno_from_sample_list(sample_list):
    '''
    根据样本列表加载对应的标注信息
    :param sample_list:
    :return:
    '''
    annotations_dir = os.path.join(pascal_voc_source_data_path, "Annotations")

    def load_anno_sample(image_id):
        anno_file = os.path.join(annotations_dir, image_id + '.xml')
        with open(anno_file, 'r') as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        anno_data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']
        return anno_data

    anno_dict = dict()
    for image_id in sample_list:
        anno_dict[image_id] = load_anno_sample(image_id)
    return anno_dict


def copy_files(sample_list, is_seg=False):
    '''
    根据样本列表拷贝图片、标注文件到目标目录
    :param sample_list:
    :return:
    '''
    source_image_path = os.path.join(pascal_voc_source_data_path, "JPEGImages")
    source_anno_path = os.path.join(pascal_voc_source_data_path, "Annotations")
    dist_image_path = os.path.join(dist_mini_voc_data_path, "JPEGImages")
    dist_anno_path = os.path.join(dist_mini_voc_data_path, "Annotations")

    mkdir(dist_image_path)
    mkdir(dist_anno_path)

    if is_seg:
        source_seg_cls_path = os.path.join(pascal_voc_source_data_path, "SegmentationClass")
        source_seg_obj_path = os.path.join(pascal_voc_source_data_path, "SegmentationObject")

        dist_seg_cls_path = os.path.join(dist_mini_voc_data_path, "SegmentationClass")
        dist_seg_obj_path = os.path.join(dist_mini_voc_data_path, "SegmentationObject")

        mkdir(dist_seg_cls_path)
        mkdir(dist_seg_obj_path)

    for image_id in sample_list:
        shutil.copy(os.path.join(source_image_path, "{}.jpg".format(image_id)),
                    os.path.join(dist_image_path, "{}.jpg".format(image_id)))
        shutil.copy(os.path.join(source_anno_path, "{}.xml".format(image_id)),
                    os.path.join(dist_anno_path, "{}.xml".format(image_id)))

        if is_seg:
            shutil.copy(os.path.join(source_seg_cls_path, "{}.png".format(image_id)),
                        os.path.join(dist_seg_cls_path, "{}.png".format(image_id)))
            shutil.copy(os.path.join(source_seg_obj_path, "{}.png".format(image_id)),
                        os.path.join(dist_seg_obj_path, "{}.png".format(image_id)))


def get_cls_info():
    def get_anno_info(anno, image_id, cls_info):
        '''
        {"类名":{"图片id1":包含该类的框数量,"图片id2":包含该类的框数量,"total":总框数量}}
        :param anno:
        :param image_id:
        :param cls_info:
        :return:
        '''
        if "object" in anno:
            for object in anno["object"]:
                cls_name = object["name"]
                if cls_name not in cls_info:
                    cls_info[cls_name] = {}
                if image_id not in cls_info[cls_name]:
                    cls_info[cls_name][image_id] = 0
                if "total" not in cls_info[cls_name]:
                    cls_info[cls_name]["total"] = 0
                cls_info[cls_name][image_id] += 1
                cls_info[cls_name]["total"] += 1

    det_cls_info = {set_type: dict() for set_type in set_list}
    for set_type in set_list:
        sample_list = load_sample_list(set_type)
        anno_dict = load_anno_from_sample_list(sample_list)

        for image_id in anno_dict:
            get_anno_info(anno_dict[image_id], image_id, det_cls_info[set_type])

    if not is_contain_seg:
        return det_cls_info, None

    seg_cls_info = {set_type: dict() for set_type in set_list}
    for set_type in set_list:
        sample_list = load_sample_list(set_type, isDet=False, isSeg=True)
        anno_dict = load_anno_from_sample_list(sample_list)

        for image_id in anno_dict:
            get_anno_info(anno_dict[image_id], image_id, seg_cls_info[set_type])

    return det_cls_info, seg_cls_info


def make_mini_dataset(cls_info, isSeg=False):
    '''
    构建检测或分割的数据
    :param isSeg:
    :return:
    '''
    print("构建分割：") if isSeg else print("构建检测：")
    ImageSetsDir = "ImageSets/Segmentation" if isSeg else "ImageSets/Main"
    ratio = seg_ratio if isSeg else det_ratio
    min_train_count = min_seg_train_count if isSeg else min_det_train_count
    min_val_count = min_seg_val_count if isSeg else min_det_val_count
    trainval_sample_list = []
    for set_type in set_list:
        candidate_sample_list = []
        for cls_name in cls_info[set_type]:
            the_cls_info = cls_info[set_type][cls_name]
            select_count = round(((len(the_cls_info) - 1) * ratio))
            img_id_list = list(the_cls_info.keys())
            img_id_list.remove("total")
            if set_type == "train":
                if select_count <= min_train_count:
                    select_count = min(min_train_count,len(the_cls_info) - 1)
            if set_type == "val":
                if select_count <= min_val_count:
                    select_count = min(min_val_count,len(the_cls_info) - 1)

            if is_shuffle:
                candidate_sample_list += random.sample(img_id_list, select_count)
            else:
                candidate_sample_list += img_id_list[:select_count]

        candidate_sample_list = list(set(candidate_sample_list))
        print("获取{}总量:{}", set_type, len(candidate_sample_list))
        with open(os.path.join(dist_mini_voc_data_path, ImageSetsDir, "{}.txt".format(set_type)), "w") as fw:
            fw.write("\n".join(candidate_sample_list) + "\n")
        trainval_sample_list += candidate_sample_list

    trainval_sample_list = list(set(trainval_sample_list))
    print("样本总量:{}", len(trainval_sample_list))
    with open(os.path.join(dist_mini_voc_data_path, ImageSetsDir, "trainval.txt"), "w") as fw:
        fw.write("\n".join(trainval_sample_list) + "\n")
    copy_files(trainval_sample_list, isSeg)


def main():
    '''
    创建迷你VOC数据集的主函数
    :return:
    '''
    make_voc_data_dir(dist_mini_voc_data_path)
    det_cls_info, seg_cls_info = get_cls_info()
    make_mini_dataset(det_cls_info)
    make_mini_dataset(seg_cls_info, True)

if __name__ == '__main__':
    main()