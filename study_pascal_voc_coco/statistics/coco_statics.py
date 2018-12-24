# -*- coding: utf-8 -* -
'''
统计coco数据集不同类的数量、框的数量等
'''
import prettytable as pt
import json
from utils import label_map_util

# coco_anno_path = "/Users/rensike/Files/data/coco/coco2017/annotations/instances_val2017.json"
coco_anno_path = "/Users/rensike/Files/temp/coco_mini/annotations/instances_train.json"

def static_coco(isDet=True, isSeg=False):
    assert isDet != isSeg
    task_type = "检测" if isDet else "分割"

    static_info_tb = pt.PrettyTable()
    static_info_tb.field_names = [" # ", "类别", "{}-图片数量".format(task_type), "{}-检测框数量".format(task_type)]

    static_info_tb.sortby = "{}-检测框数量".format(task_type)

    cls_info = dict()
    with open(coco_anno_path, "r") as fid:
        groundtruth_data = json.load(fid)
        images = groundtruth_data['images']
        category_index = label_map_util.create_category_index(
            groundtruth_data['categories'])

        already_mark_this_img = dict()
        if 'annotations' in groundtruth_data:
            for annotation in groundtruth_data['annotations']:
                if isSeg and "segmentation" not in annotation:
                    continue
                if len(annotation["segmentation"]) == 0:
                    print("分割的坐标为空！")
                cat = category_index[annotation["category_id"]]["name"]
                image_id = annotation['image_id']
                if cat not in cls_info:
                    cls_info[cat] = {"image_count": 0, "box_count": 0}
                cls_info[cat]["box_count"] += 1
                if image_id not in already_mark_this_img:
                    already_mark_this_img[image_id] = dict()
                if cat not in already_mark_this_img[image_id]:
                    cls_info[cat]["image_count"] += 1
                    already_mark_this_img[image_id][cat] = True

    total_image_count = 0
    total_box_count = 0
    for ix, cat in enumerate(cls_info):
        static_info_tb.add_row([ix, cat, cls_info[cat]["image_count"], cls_info[cat]["box_count"]])
        total_image_count += cls_info[cat]["image_count"]
        total_box_count += cls_info[cat]["box_count"]
    static_info_tb.add_row(["", "总量", total_image_count, total_box_count])
    return static_info_tb


if __name__ == "__main__":
    det_static_info_tb = static_coco()
    seg_static_info_tb = static_coco(isDet=False,isSeg=True)

    print(det_static_info_tb)
    print(seg_static_info_tb)
