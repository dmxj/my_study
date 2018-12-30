# -*- coding: utf-8 -* -
import sys

sys.path.append("/Users/rensike/Workspace/models/research")
import numpy as np
import os
import tensorflow as tf

from distutils.version import StrictVersion
from PIL import Image
import glob
import time
import collections

from utils import label_map_util
from utils import visualization_utils as vis_util
from utils import ops as utils_ops
from utils import image_util

if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
    raise ImportError('Please upgrade your TensorFlow installation to v1.9.* or later!')


class ObjectDetectionModel(object):
    def __init__(self, frozen_graph_path, label_map_path):
        self.frozen_graph_path = frozen_graph_path
        self.label_map_path = label_map_path
        self.load_model()
        self.load_sess()
        self.load_label_map()

    def load_model(self):
        '''
        加载模型
        :return:
        '''
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.frozen_graph_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        self.graph = detection_graph

    def load_sess(self):
        '''
        初始化session
        :return:
        '''
        with self.graph.as_default():
            self.sess = tf.Session()
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            self.tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    self.tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                        tensor_name)

            self.image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

    def load_label_map(self):
        '''
        加载label，生成 {类id : 类名} 的映射字典
        :return:
        '''
        category_index = label_map_util.create_category_index_from_labelmap(self.label_map_path, use_display_name=True)
        self.category_index = category_index
        self.category_name_index = {category_index[cat_id]["name"]: category_index[cat_id] for cat_id in category_index}
        self.categories = label_map_util.create_categories_from_labelmap(self.label_map_path)

    def reframe_detection_mask(self, image):
        '''
        对图片的分割掩码进行转换
        :param image:
        :return:
        '''
        if 'detection_masks' in self.tensor_dict:
            # The following processing is only for single image
            detection_boxes = tf.squeeze(self.tensor_dict['detection_boxes'], [0])
            detection_masks = tf.squeeze(self.tensor_dict['detection_masks'], [0])
            # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
            real_num_detection = tf.cast(self.tensor_dict['num_detections'][0], tf.int32)
            detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
            detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
            detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                detection_masks, detection_boxes, image.shape[0], image.shape[1])
            detection_masks_reframed = tf.cast(
                tf.greater(detection_masks_reframed, 0.5), tf.uint8)
            # Follow the convention by adding back the batch dimension
            self.tensor_dict['detection_masks'] = tf.expand_dims(
                detection_masks_reframed, 0)

    def transfrom_input(self, image_path, resize_width=0, resize_height=0):
        '''
        根据图片路径转换输入图片
        :param image_path:
        :return:
        '''
        image = Image.open(image_path)
        if resize_height > 0 and resize_width > 0:
            image = image.resize((resize_width, resize_height))
        image_np = image_util.load_image_into_numpy_array(image)
        return image_np

    def show_detection_result(self, image_np, output_dict):
        '''
        显示检测的结果
        :param image_np:
        :param output_dict:
        :return:
        '''
        assert "detection_boxes" in output_dict, "there must have detection_boxes in output_dict"
        assert "detection_classes" in output_dict, "there must have detection_classes in output_dict"
        assert "detection_scores" in output_dict, "there must have detection_scores in output_dict"
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            self.category_index,
            instance_masks=output_dict.get('detection_masks'),
            use_normalized_coordinates=True,
            line_thickness=8)
        image_drawed = image_util.load_numpy_array_into_image(image_np)
        image_drawed.show()

    def save_detection_result(self, image_np, output_dict, save_path):
        '''
        保存检测的结果
        :param image_np:
        :param output_dict:
        :param save_path:
        :return:
        '''
        assert "detection_boxes" in output_dict, "there must have detection_boxes in output_dict"
        assert "detection_classes" in output_dict, "there must have detection_classes in output_dict"
        assert "detection_scores" in output_dict, "there must have detection_scores in output_dict"
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            self.category_index,
            instance_masks=output_dict.get('detection_masks'),
            use_normalized_coordinates=True,
            line_thickness=8)
        image_drawed = image_util.load_numpy_array_into_image(image_np)
        image_drawed.save(save_path)

    def inference_image_np(self, image_np, is_filter=False, is_show=False, save_path=None):
        self.reframe_detection_mask(image_np)
        # Run inference
        output_dict = self.sess.run(self.tensor_dict,
                                    feed_dict={self.image_tensor: np.expand_dims(image_np, 0)})

        # all outputs are float32 numpy arrays, so convert types as appropriate
        output_dict['num_detections'] = int(output_dict['num_detections'][0])
        output_dict['detection_classes'] = output_dict[
            'detection_classes'][0].astype(np.uint8)
        # ymin,xmin,ymax,xmax
        output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
        output_dict['detection_scores'] = output_dict['detection_scores'][0]
        if 'detection_masks' in output_dict:
            output_dict['detection_masks'] = output_dict['detection_masks'][0]

        if is_show:
            self.show_detection_result(image_np, output_dict)

        if save_path is not None:
            self.save_detection_result(image_np, output_dict, save_path)

        return self.filter_output_dict(output_dict) if is_filter else output_dict

    def run_inference(self, image_path, is_filter=False, is_show=False, save_path=None):
        '''
        运行检测任务
        :param image_path: 图片路径
        :param is_show: 是否显示
        :param save_path: 保存的路径
        :return:
        '''
        image = self.transfrom_input(image_path)
        return self.inference_image_np(image, is_filter, is_show, save_path)

    def inference_image_np_batch(self, image_np_batch, is_filter=False, is_show=False, save_path_list=None):
        self.reframe_detection_mask(image_np_batch[0])
        # Run inference
        output_dict = self.sess.run(self.tensor_dict,
                                    feed_dict={self.image_tensor: image_np_batch})

        output_dict_list = []
        for ix, image in enumerate(image_np_batch):
            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict_item = {}
            output_dict_item['num_detections'] = int(output_dict['num_detections'][ix])
            output_dict_item['detection_classes'] = output_dict[
                'detection_classes'][ix].astype(np.uint8)
            output_dict_item['detection_boxes'] = output_dict['detection_boxes'][ix]
            output_dict_item['detection_scores'] = output_dict['detection_scores'][ix]
            if 'detection_masks' in output_dict:
                output_dict_item['detection_masks'] = output_dict['detection_masks'][ix]
            output_dict_list.append(self.filter_output_dict(output_dict_item) if is_filter else output_dict_item)

            if is_show:
                self.show_detection_result(image, output_dict_item)

            if save_path_list is not None:
                self.save_detection_result(image, output_dict_item, save_path_list[ix])

        return output_dict_list

    def run_inference_batch(self, image_path_list, image_size=224, is_filter=False, is_show=False, save_path_list=None):
        '''
        batch运行检测任务
        :param image_path: 图片路径
        :param is_show: 是否显示
        :param save_path: 保存的路径
        :return:
        '''
        assert save_path_list is None or len(image_path_list) == len(
            save_path_list), "save_path_list length must equals to image_path_list"
        image_lists = np.array([self.transfrom_input(img_path, image_size, image_size) for img_path in image_path_list])
        output_dict_list = self.run_inference_batch(image_lists, is_filter, is_show, save_path_list)

        output_dict_map = {}
        for ix, output_dict in enumerate(output_dict_list):
            image_id = os.path.split(image_path_list[ix])[-1].rsplit(".", 1)[0]
            output_dict_map[image_id] = output_dict
        return output_dict_map

    def filter_output_dict(self, output_dict, min_score_thresh=0.5):
        '''
        过滤检测结果，只保留分值在阈值（默认0.5）以上的检测框
        :param output_dict:
        :param min_score_thresh:
        :return:
        '''
        filtered_output_dict = output_dict.copy()
        boxes = output_dict['detection_boxes']
        classes = output_dict['detection_classes']
        scores = output_dict['detection_scores']
        instance_masks = output_dict['detection_masks'] if "detection_masks" in output_dict else None

        filtered_boxes = []
        filtered_classes = []
        filtered_scores = []
        filtered_instance_masks = []

        max_boxes_to_draw = boxes.shape[0]
        for i in range(max_boxes_to_draw):
            if scores[i] > min_score_thresh:
                box = tuple(boxes[i].tolist())
                filtered_boxes.append(box)
                filtered_scores.append(scores[i])
                if instance_masks is not None:
                    filtered_instance_masks.append(instance_masks[i])
                else:
                    if classes[i] in self.category_index.keys():
                        filtered_classes.append(classes[i])
                    else:
                        filtered_classes.append(-1)

        filtered_output_dict["detection_boxes"] = filtered_boxes
        filtered_output_dict["detection_classes"] = filtered_classes
        filtered_output_dict["detection_scores"] = filtered_scores
        filtered_output_dict["detection_masks"] = filtered_instance_masks
        filtered_output_dict["num_detections"] = len(filtered_boxes)
        return filtered_output_dict


if __name__ == "__main__":
    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    PATH_TO_FROZEN_GRAPH = '/Users/rensike/Resources/models/tensorflow/object_detection/ssd_mobilenet_v1_voc/frozen_inference_graph.pb'
    # List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

    tf_obj_det = ObjectDetectionModel(PATH_TO_FROZEN_GRAPH, PATH_TO_LABELS)

    # inference single image
    TEST_IMAGE_PATH = "/Users/rensike/Files/temp/voc_mini/JPEGImages/2010_002537.jpg"
    det_output = tf_obj_det.run_inference(TEST_IMAGE_PATH, is_filter=True, is_show=True)
    # width,height = Image.open(TEST_IMAGE_PATH).size
    # from draw import draw_bbox
    # bbox_list = []
    # label_list = []
    # for i in range(det_output["num_detections"]):
    #     bbox = det_output["detection_boxes"][i]
    #     bbox_list.append([bbox[1]*width,bbox[0]*height,bbox[3]*width, bbox[2]*height])
    #     label_list.append("-")
    # draw_bbox.draw_bbox_use_pil(TEST_IMAGE_PATH,bbox_list,label_list,isShow=True)
    print(det_output)
    # tf_obj_det.run_inference(TEST_IMAGE_PATH)
    # print("inference single image time consume is:", time.time() - t0)

    # inference batch image
    # TEST_IMAGE_LIST = glob.glob("images/*.jpg")
    # TEST_IMAGE_SAVE_LIST = [os.path.join("./result/images",os.path.split(img_path)[-1].rsplit(".",1)[0] + "_det_result.jpg") for img_path in TEST_IMAGE_LIST]
    # t0 = time.time()
    # tf_obj_det.run_inference_batch(TEST_IMAGE_LIST, is_show=False,save_path_list=TEST_IMAGE_SAVE_LIST)
    # print("inference 16 images time consume is:", time.time() - t0)

    # det_res = tf_obj_det.run_inference_batch(["image1.jpg", "image2.jpg"],is_filter=True, is_show=False)
    # print(det_res)
