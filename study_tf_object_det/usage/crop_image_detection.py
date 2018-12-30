# -*- coding: utf-8 -* -
'''
对大图进行切图检测，大图切成若干小图，使用小图进行检测，小图检测的结果再合并到最终的结果中
'''
import os
import time
import numpy as np
from PIL import Image
from utils import image_util
from utils import visualization_utils as vis_util
from object_detection import ObjectDetectionModel

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = '/Users/rensike/Resources/models/tensorflow/object_detection/ssd_mobilenet_v1_voc/frozen_inference_graph.pb'
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('../data', 'pascal_label_map.pbtxt')

# 测试大图片预测的效果和时间
def test_big_image_infer(image_path):
    tf_obj_det = ObjectDetectionModel(PATH_TO_FROZEN_GRAPH, PATH_TO_LABELS)
    t0 = time.time()
    outdict = tf_obj_det.run_inference(image_path,save_path="../result/big_image_det_result2.jpg")
    print(outdict)
    print("detect cost time: ",time.time() - t0)

# 大图裁剪成小图
crop_width = 200
crop_height = 200
overlap = 30
def crop_image_to_small(image_path):
    image = Image.open(image_path)
    image_np = image_util.load_image_into_numpy_array(image)
    print("image shape:",image_np.shape)
    origin_height,origin_width,_ = image_np.shape
    max_ix_num = int((origin_width - crop_width) / (crop_width-overlap)) + 2
    max_iy_num = int((origin_height - crop_height) / (crop_height-overlap)) + 2
    print("max_ix_num:",max_ix_num)
    print("max_iy_num:",max_iy_num)
    sub_image_xy_list = []
    sub_image_np_list = []
    for ix in range(max_ix_num):
        start_ix = ix*(crop_width-overlap)
        if start_ix >= origin_width:
            break
        if start_ix + crop_width > origin_width:
            start_ix = origin_width - crop_width
        for iy in range(max_iy_num):
            start_iy = iy * (crop_height - overlap)
            if start_iy >= origin_height:
                break
            if start_iy + crop_height > origin_height:
                start_iy = origin_height - crop_height

            crop_img = image_np[start_iy:start_iy+crop_height,start_ix:start_ix+crop_width,:]
            sub_image_xy_list.append((start_ix,start_iy))
            sub_image_np_list.append(crop_img[np.newaxis,::])
    return image_np,sub_image_xy_list,sub_image_np_list

def crop_image_detect(image_path,save_path):
    tf_obj_det = ObjectDetectionModel(PATH_TO_FROZEN_GRAPH, PATH_TO_LABELS)
    print("start run")
    t0 = time.time()
    image_np, sub_image_xy_list, sub_image_np_list = crop_image_to_small(image_path)
    print("sub_image_np_list length:",len(sub_image_np_list))
    print("sub_image_xy_list length:",len(sub_image_xy_list))
    print(np.concatenate(sub_image_np_list).shape)
    output_dict_list = tf_obj_det.inference_image_np_batch(np.concatenate(sub_image_np_list))

    print("finish inference.")

    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        np.vstack([output_dict['detection_boxes'] for output_dict in output_dict_list]),
        np.vstack([output_dict['detection_classes'] for output_dict in output_dict_list]),
        np.vstack([output_dict['detection_scores'] for output_dict in output_dict_list]),
        tf_obj_det.category_index,
        instance_masks=np.vstack([output_dict['detection_masks'] for output_dict in output_dict_list]),
        use_normalized_coordinates=False,
        line_thickness=8)
    image_drawed = image_util.load_numpy_array_into_image(image_np)
    image_drawed.save(save_path)
    print("process done,cost time: ",time.time() - t0)

if __name__ == '__main__':
    image_path = "/Users/rensike/Files/work/tf_object_detect/big_image.jpeg"
    # image_path = "/Users/rensike/Files/temp/voc_mini/JPEGImages/2010_002537.jpg"
    # test_big_image_infer(image_path)
    crop_image_detect(image_path,"../result/crop_image_det_result.jpg")

    # image = Image.open(image_path)
    # image_np = image_util.load_image_into_numpy_array(image)
    # print(image_np[0:200,0:200,:].shape)