# -*- coding: utf-8 -* -
'''
统计pascal voc不同类的数量、框的数量等
'''
import mmcv
import os
from utils import dataset_util
from lxml import etree
import prettytable as pt

# pascal_voc_data_path = "/Users/rensike/Files/data/voc/VOCdevkit2/VOC2012"
pascal_voc_data_path = "/Users/rensike/Files/temp/voc_mini"

def load_sample_list(set, isDet=True, isSeg=False):
    '''
    加载样本列表
    :param set: train|val|trainval|test
    :param isDet: 是否加载检测数据
    :param isSeg: 是否加载分割数据
    :return:
    '''
    assert isDet != isSeg
    sample_file = os.path.join(pascal_voc_data_path, "ImageSets/Main", "{}.txt".format(set))
    if isSeg:
        sample_file = os.path.join(pascal_voc_data_path, "ImageSets/Segmentation", "{}.txt".format(set))
    sample_list = mmcv.list_from_file(sample_file)
    return sample_list


def load_anno_from_sample_list(sample_list):
    '''
    根据样本列表加载对应的标注信息
    :param sample_list:
    :return:
    '''
    annotations_dir = os.path.join(pascal_voc_data_path, "Annotations")

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


def static_pascal_voc_total():
    '''
    统计Pascal Voc数据集的总体详情
    :return:
    '''
    set_list = ["train", "val", "trainval"]
    total_info_tb = pt.PrettyTable()
    total_info_tb.field_names = [" # ", "数据集类型", "检测样本数量", "分割样本数量", "交集数量"]
    for ix, set_type in enumerate(set_list):
        det_sample_list = load_sample_list(set_type)
        seg_sample_list = load_sample_list(set_type, isDet=False, isSeg=True)

        total_info_tb.add_row([ix, set_type, len(det_sample_list), len(seg_sample_list),
                               len(list(set(det_sample_list).intersection(set(seg_sample_list))))])

    return total_info_tb


def static_pascal_voc_class(isDet=True, isSeg=False):
    '''
    对检测或分割数据的不同类数据进行统计
    :param isDet:
    :param isSeg:
    :return:
    '''
    assert isDet != isSeg

    def parse_cls_info(anno, cls_info):
        if "object" in anno:
            already_mark_this_img = dict()
            for object in anno["object"]:
                cls_name = object["name"]
                if cls_name not in cls_info:
                    cls_info[cls_name] = {"image_count": 0, "box_count": 0}
                if cls_name not in already_mark_this_img:
                    cls_info[cls_name]["image_count"] += 1
                    already_mark_this_img[cls_name] = True
                cls_info[cls_name]["box_count"] += 1

    set_list = ["train", "val", "trainval"]
    cls_info_total = dict()
    for set_type in set_list:
        sample_list = load_sample_list(set_type, isDet, isSeg)
        anno_dict = load_anno_from_sample_list(sample_list)

        cls_info = dict()
        [parse_cls_info(anno_dict[img_id], cls_info) for img_id in anno_dict]

        cls_info_total[set_type] = cls_info

    task_type = "检测" if isDet else "分割"
    cls_info_tb = pt.PrettyTable()
    cls_info_tb.field_names = [" # ", "类别", "{}train图片数".format(task_type), "{}train bbox数".format(task_type),
                               "{}val图片数".format(task_type), "{}val bbox数".format(task_type),
                               "{}trainval图片数".format(task_type),
                               "{}trainval bbox数".format(task_type)]

    cls_info_tb.sortby = "{}trainval bbox数".format(task_type)

    cls_info_tb_dict = dict()
    for set_type in cls_info_total:
        for cls_name in cls_info_total[set_type]:
            if cls_name not in cls_info_tb_dict:
                cls_info_tb_dict[cls_name] = dict()
            if set_type not in cls_info_tb_dict[cls_name]:
                cls_info_tb_dict[cls_name][set_type] = dict()
            cls_info_tb_dict[cls_name][set_type]["image_count"] = cls_info_total[set_type][cls_name][
                "image_count"]
            cls_info_tb_dict[cls_name][set_type]["box_count"] = cls_info_total[set_type][cls_name][
                "box_count"]

    sum_count_arr = [0 for i in range(6)]
    for ix, cls_name in enumerate(cls_info_tb_dict):
        count_arr = [
            cls_info_tb_dict[cls_name]["train"]["image_count"],
            cls_info_tb_dict[cls_name]["train"]["box_count"],
            cls_info_tb_dict[cls_name]["val"]["image_count"],
            cls_info_tb_dict[cls_name]["val"]["box_count"],
            cls_info_tb_dict[cls_name]["trainval"]["image_count"],
            cls_info_tb_dict[cls_name]["trainval"]["box_count"]
        ]
        for i,cnt in enumerate(count_arr):
            sum_count_arr[i] += cnt
        cls_info_tb.add_row([ix,cls_name] + count_arr)
    cls_info_tb.add_row(["","总数"] + sum_count_arr)
    return cls_info_tb


if __name__ == "__main__":
    total_info_tb = static_pascal_voc_total()
    det_cls_info_tb = static_pascal_voc_class()
    seg_cls_info_tb = static_pascal_voc_class(isDet=False, isSeg=True)

    print(total_info_tb)
    print(det_cls_info_tb)
    print(seg_cls_info_tb)
