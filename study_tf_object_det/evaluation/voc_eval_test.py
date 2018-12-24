# -*- coding: utf-8 -* -
'''
使用官方的voc_eval进行模型在Pascal Voc数据集上的评估。
（1）将模型在测试集上进行推理，得到的结果保存到文件中；
（2）将推理结果路径、测试集标注文件、测试集等传给voc_eval/voc_eval方法进行推理，得到结果
'''
import object_detection
from evaluation import voc_eval
from utils import label_map_util
import mmcv
import os

# 数据相关路径
VOC_TEST_EVALSET_FILE = "/Users/rensike/Files/temp/voc_mini/ImageSets/Main/val.txt"
VOC_TEST_IMG_PATH = "/Users/rensike/Files/temp/voc_mini/JPEGImages/"
VOC_TEST_ANNO_PATH = "/Users/rensike/Files/temp/voc_mini/Annotations/"
VOC_LABELS_PATH = "../data/pascal_label_map.pbtxt"

# 模型路径
FROZEN_MODEL_PATH = "/Users/rensike/Resources/models/tensorflow/object_detection/ssd_mobilenet_v1_coco_2017_11_17/frozen_inference_graph.pb"

# 结果文件路径
TEST_RESULT_FILE = "/Users/rensike/Files/temp/voc_mini_eval_det_result/det_{}.txt"
EVAL_CACHE_DIR = "/Users/rensike/Files/temp/voc_mini_eval_cache"


def infer_on_eval_dataset():
    eval_list = mmcv.list_from_file(VOC_TEST_EVALSET_FILE)
    eval_list = filter(lambda x:x != "",eval_list)
    eval_img_list = [os.path.join(VOC_TEST_IMG_PATH,img_id + ".jpg") for img_id in eval_list]
    tf_obj_det = object_detection.ObjectDetectionModel(FROZEN_MODEL_PATH, VOC_LABELS_PATH)
    output_dict_map = tf_obj_det.run_inference_batch(eval_img_list)

    category_index = label_map_util.create_category_index_from_labelmap(VOC_LABELS_PATH)

    det_result_dict = {}

    for image_id in output_dict_map:
        output_dict = output_dict_map["image_id"]
        for i in range(output_dict["num_detections"]):
            confidence = output_dict['detection_scores'][i]
            bbox = output_dict['detection_boxes'][i]
            class_name = category_index[output_dict['detection_classes'][i]]["name"]

            if class_name not in det_result_dict:
                det_result_dict[class_name] = []

            det_item = [image_id] + [confidence] + bbox
            det_item = map(str,det_item)
            det_result_dict[class_name].append(" ".join(det_item))

    for class_name in det_result_dict:
        det_list = det_result_dict[class_name]
        with open(TEST_RESULT_FILE.format(class_name),"a+") as det_f:
            det_f.write("\n".join(det_list) + "\n")
        print("class:{} det result record done.".format(class_name))

def run_voc_val():
    category_index = label_map_util.create_category_index_from_labelmap(VOC_LABELS_PATH)
    for cat_id in category_index:
        class_name = category_index[cat_id]["name"]
        rec, prec, ap = voc_eval.voc_eval(TEST_RESULT_FILE,
                          os.path.join(VOC_TEST_ANNO_PATH,"{}.xml",
                          VOC_TEST_EVALSET_FILE,
                          class_name,
                          EVAL_CACHE_DIR))

        print("class:{}, recall = {}, precision = {}, ap = {}".format(class_name,rec,prec,ap))

if __name__ == '__main__':
    infer_on_eval_dataset()
    run_voc_val()


