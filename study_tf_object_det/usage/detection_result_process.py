# -*- coding: utf-8 -* -
'''
对检测结果进行处理
'''
import os
import shutil
import unicodedata
import object_detection
import numpy as np
from lxml import etree
from prettytable import PrettyTable
from PIL import Image
from utils import image_util
from utils import dataset_util
from utils import visualization_utils as vis_util
from utils import np_box_ops as box_ops
from utils import object_detection_evaluation


class DetectionResultProcess(object):
    def __init__(self, frozen_graph_path, label_map_path):
        self.tf_obj_det = object_detection.ObjectDetectionModel(frozen_graph_path, label_map_path)

    def _load_anno_sample(self, anno_path):
        '''
        加载一个标注信息
        :param anno_path:
        :return:
        '''
        with open(anno_path, 'r') as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        anno_data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']
        return anno_data

    def _draw_and_save_ground_truth(self, image_path, anno_path, save_path):
        '''
        将groung truth的框进行绘制和保存
        :param image_path:
        :param anno_path:
        :param save_path:
        :return:
        '''
        anno_data = self._load_anno_sample(anno_path)
        image_np = self.tf_obj_det.transfrom_input(image_path)
        output_dict = dict(detection_boxes=[], detection_classes=[], detection_scores=[])

        for obj in anno_data["object"]:
            output_dict['detection_boxes'].append(
                [int(obj["bndbox"]["ymin"]), int(obj["bndbox"]["xmin"]), int(obj["bndbox"]["ymax"]),
                 int(obj["bndbox"]["xmax"])])
            output_dict['detection_classes'].append(self.tf_obj_det.category_name_index[obj["name"]]["id"])
            output_dict['detection_scores'].append(1.0)
        output_dict['detection_boxes'] = np.array(output_dict['detection_boxes'])
        output_dict['detection_classes'] = np.array(output_dict['detection_classes'])
        output_dict['detection_scores'] = np.array(output_dict['detection_scores'])
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            self.tf_obj_det.category_index,
            instance_masks=output_dict.get('detection_masks'),
            use_normalized_coordinates=False,
            line_thickness=8)
        image_drawed = image_util.load_numpy_array_into_image(image_np)
        image_drawed.save(save_path)

    def _calc_single_sample_det_info(self, ground_bbox_list, det_bbox_list, iou_threshold=0.5):
        '''
        返回某张图片的检测结果信息（虚警个数、漏警个数）
        :param ground_bbox_list: ground truth检测框列表，ymin, xmin, ymax, xmax
        :param det_bbox_list: 检测结果框列表，ymin, xmin, ymax, xmax
        :param iou_threshold:
        :return:
        '''
        if len(ground_bbox_list) == 0:
            return 0, len(det_bbox_list), 0

        if len(det_bbox_list) == 0:
            return 0, 0, len(ground_bbox_list)

        iou_mat = box_ops.iou(np.array(ground_bbox_list), np.array(det_bbox_list))

        # iou过阈值的为正确检测的个数TP
        tp = 0
        gt_over_map = {}
        det_over_map = {}
        for i in range(len(ground_bbox_list)):
            if i in gt_over_map:
                continue
            max_iou = 0
            max_j = -1
            for j in range(len(det_bbox_list)):
                if j in det_over_map:
                    continue
                if iou_mat[i, j] >= iou_threshold and iou_mat[i,j] > max_iou:
                    max_iou = iou_mat[i,j]
                    max_j = j

            if max_j >= 0:
                tp += 1
                gt_over_map[i] = True
                det_over_map[max_j] = True

        # 检测的框没有ground truth的框跟其对应的，即过检（虚警）的框，FP
        fp = len(det_bbox_list) - tp

        # ground truth的框，没检测到的，即过杀（漏警）的框，FN
        fn = len(ground_bbox_list) - tp

        return tp, fp, fn

    def calc_det_info(self, image_path_list, anno_path_list, save_path=None):
        '''
        计算检测的结果，虚警个数、漏警个数，虚警率、漏警率，等等
        :param image_path_list:
        :param anno_path_list:
        :param save_path:
        :return:
        '''
        assert len(image_path_list) == len(
            anno_path_list), "anno path list length should be equal to image path list length"

        tp_total, fp_total, fn_total = 0, 0, 0
        gt_total, det_total = 0, 0
        det_info_list = []
        for (image_path, anno_path) in list(zip(image_path_list, anno_path_list)):
            image_name = os.path.basename(image_path)
            width, height = Image.open(image_path).size
            det_output_dict = self.tf_obj_det.run_inference(image_path, is_filter=True)
            det_bbox_list = det_output_dict["detection_boxes"]
            det_bbox_list = [[bbox[0] * height, bbox[1] * width, bbox[2] * height, bbox[3] * width] for bbox in
                             det_bbox_list]
            anno_data = self._load_anno_sample(anno_path)
            ground_bbox_list = [
                [int(obj["bndbox"]["ymin"]),
                 int(obj["bndbox"]["xmin"]),
                 int(obj["bndbox"]["ymax"]),
                 int(obj["bndbox"]["xmax"])] for obj in anno_data["object"]]
            tp, fp, fn = self._calc_single_sample_det_info(ground_bbox_list, det_bbox_list)
            tp_total += tp
            fp_total += fp
            fn_total += fn
            gt_total += len(ground_bbox_list)
            det_total += len(det_bbox_list)
            det_info_list.append([
                image_name,
                tp, fp, fn,
                len(ground_bbox_list), len(det_bbox_list),
                round(float(fp) / len(det_bbox_list), 2) if len(det_bbox_list) > 0 else 0,
                round(float(fn) / len(ground_bbox_list), 2) if len(ground_bbox_list) > 0 else 0,
            ])

        print("there are {} images detected.".format(len(det_info_list)))

        if save_path is not None:
            if os.path.exists(save_path):
                os.remove(save_path)
            with open(save_path, "a+") as det_info_fw:
                for det_info in det_info_list:
                    det_info_fw.write("\t".join([str(i) for i in det_info]) + "\n")
                det_info_fw.write("\t".join(["total",
                                             str(tp_total), str(fp_total), str(fn_total),
                                             str(gt_total), str(det_total),
                                             str(round(float(fp_total) / det_total, 2)),
                                             str(round(float(fn_total) / gt_total, 2))
                                             ]))

        table = PrettyTable(["总量", "TP", "FP", "FN", "ground truth", "detections", "虚警率", "漏警率", "精确度", "召回率"])
        table.add_row(["-", tp_total, fp_total, fn_total, gt_total, det_total,
                       round(float(fp_total) / det_total, 2),round(float(fn_total) / gt_total, 2),
                       round(1-(float(fp_total) / det_total),2),round(1-(float(fn_total) / gt_total),2)])

        print(table)

    def make_evaluation(self,image_path_list, anno_path_list, save_path=None):
        '''
        进行评估，使用pascal voc评估指标
        :param image_path_list:
        :param anno_path_list:
        :param save_path:
        :return:
        '''
        evaluator_ode = object_detection_evaluation.ObjectDetectionEvaluation(len(self.tf_obj_det.category_index))
        evaluator_pde = object_detection_evaluation.PascalDetectionEvaluator(categories=self.tf_obj_det.categories)
        for (image_path,anno_path) in list(zip(image_path_list,anno_path_list)):
            image_name = os.path.split(image_path)[-1].rsplit(".",1)[0]
            width, height = Image.open(image_path).size
            anno_data = self._load_anno_sample(anno_path)
            groundtruth_boxes = [
                        [obj["bndbox"]["ymin"],
                         obj["bndbox"]["xmin"],
                         obj["bndbox"]["ymax"],
                         obj["bndbox"]["xmax"]] for obj in anno_data["object"]]
            groundtruth_classes = [self.tf_obj_det.category_name_index[obj["name"]]["id"] for obj in anno_data["object"]]
            evaluator_ode.add_single_ground_truth_image_info(image_name,
                                                         np.array(groundtruth_boxes,dtype=np.float),
                                                         np.array(groundtruth_classes) - 1)
            evaluator_pde.add_single_ground_truth_image_info(image_name,
                                                             {
                                                                 "groundtruth_classes": np.array(groundtruth_classes) - 1,
                                                                 "groundtruth_boxes": np.array(groundtruth_boxes,dtype=np.float)
                                                             })

            det_output_dict = self.tf_obj_det.run_inference(image_path, is_filter=True)
            det_bbox_list = det_output_dict["detection_boxes"]
            det_bbox_list = [[bbox[0] * height, bbox[1] * width, bbox[2] * height, bbox[3] * width] for bbox in
                             det_bbox_list]

            evaluator_ode.add_single_detected_image_info(image_name,
                                                     np.array(det_bbox_list,dtype=np.float),
                                                     np.array(det_output_dict["detection_scores"]),
                                                     np.array(det_output_dict["detection_classes"]) - 1
                                                     )
            evaluator_pde.add_single_detected_image_info(image_name,
                                                         {
                                                             "detection_classes":np.array(det_output_dict["detection_classes"]) - 1,
                                                             "detection_scores": np.array(det_output_dict["detection_scores"]),
                                                             "detection_boxes": np.array(det_bbox_list,dtype=np.float),
                                                         })

        metrics_ode = evaluator_ode.evaluate()
        metrics_pde = evaluator_pde.evaluate()
        mAP = metrics_ode.mean_ap

        class_precision_map = {self.tf_obj_det.category_index[ix+1]["name"]:pre_arr[-1] for ix,pre_arr in enumerate(metrics_ode.precisions)}
        class_recall_map = {self.tf_obj_det.category_index[ix+1]["name"]:recall_arr[-1] for ix,recall_arr in enumerate(metrics_ode.recalls)}

        print("precisions length:",len(metrics_ode.precisions))
        print("recalls length:",len(metrics_ode.recalls))

        print(class_precision_map)
        print(class_recall_map)

        table = PrettyTable(["#", "类别", "AP", "精确率", "召回率"])
        eval_res_list = []
        for ix,category in enumerate(self.tf_obj_det.categories):
            category_name = category["name"]
            display_name = (
                    evaluator_pde._metric_prefix + 'PerformanceByCategory/AP@{}IOU/{}'.format(
                                    evaluator_pde._matching_iou_threshold,
                                    unicodedata.normalize('NFKD', category_name).encode('ascii', 'ignore')
                                )
            )
            table.add_row([ix,category_name,metrics_pde[display_name],class_precision_map[category_name],class_recall_map[category_name]])
            eval_res_list.append([category_name,str(metrics_pde[display_name]),str(class_precision_map[category_name]),str(class_recall_map[category_name])])

        mP = float(np.mean(np.array(list(class_precision_map.values()))))
        mR = float(np.mean(np.array(list(class_recall_map.values()))))
        table.add_row([len(self.tf_obj_det.categories),"平均值",mAP,round(mP,4),round(mR,4)])
        eval_res_list.append(["平均值",str(mAP),str(round(mP,4)),str(round(mR,4))])

        if save_path is not None:
            if os.path.exists(save_path):
                os.remove(save_path)
            with open(save_path,"a+") as fw:
                fw.write("\n".join(["\t".join(eval_item) for eval_item in eval_res_list]))
        print(table)


    def make_vis_result(self, image_path_list, anno_path_list, save_path):
        '''
        保存检测结果，以及ground truth和检测结果的拼接对比图片
        :param image_path_list:
        :param save_path:
        :return:
        '''
        assert len(image_path_list) == len(
            anno_path_list), "anno path list length should be equal to image path list length"
        ground_truth_path = os.path.join(save_path, "gt_result")
        det_result_path = os.path.join(save_path, "det_result")
        compare_result_path = os.path.join(save_path, "compare_result")

        if os.path.exists(ground_truth_path):
            shutil.rmtree(ground_truth_path)

        if os.path.exists(det_result_path):
            shutil.rmtree(det_result_path)

        if os.path.exists(compare_result_path):
            shutil.rmtree(compare_result_path)

        os.makedirs(ground_truth_path)
        os.makedirs(det_result_path)
        os.makedirs(compare_result_path)

        image_gt_path_list = []
        for ix, image_path in enumerate(image_path_list):
            image_filename = os.path.split(image_path)[-1]
            image_name, image_suffix = image_filename.rsplit(".", 1)
            gt_save_path = os.path.join(ground_truth_path, image_name + "_gt" + "." + image_suffix)
            self._draw_and_save_ground_truth(image_path, anno_path_list[ix], gt_save_path)
            image_gt_path_list.append(gt_save_path)

        print("draw and save ground result finish.")

        image_det_path_list = []
        for image_path in image_path_list:
            image_filename = os.path.split(image_path)[-1]
            image_name, image_suffix = image_filename.rsplit(".", 1)
            det_save_path = os.path.join(det_result_path, image_name + "_det" + "." + image_suffix)
            self.tf_obj_det.run_inference(image_path, save_path=det_save_path)
            image_det_path_list.append(det_save_path)

        print("make detect vis result finish.")

        for (image_gt_path, image_det_path) in list(zip(image_gt_path_list, image_det_path_list)):
            image_filename = os.path.split(image_gt_path)[-1]
            image_name, image_suffix = image_filename.rsplit(".", 1)
            image_compare_filepath = os.path.join(compare_result_path, image_name + "_gt_and_det" + "." + image_suffix)
            image_util.concat_images([image_gt_path, image_det_path], is_show=False, save_path=image_compare_filepath)

        print("make ground truth and detect result compare vis finish.")

    def test(self,image_path,anno_path):
        anno_data = self._load_anno_sample(anno_path)
        width,height = Image.open(image_path).size
        det_output_dict = self.tf_obj_det.run_inference(image_path, is_filter=True)
        det_bbox_list = det_output_dict["detection_boxes"]
        det_bbox_list = [[bbox[0] * height, bbox[1] * width, bbox[2] * height, bbox[3] * width] for bbox in
                         det_bbox_list]
        anno_data = self._load_anno_sample(anno_path)
        ground_bbox_list = [
            [int(obj["bndbox"]["ymin"]),
             int(obj["bndbox"]["xmin"]),
             int(obj["bndbox"]["ymax"]),
             int(obj["bndbox"]["xmax"])] for obj in anno_data["object"]]
        tp, fp, fn = self._calc_single_sample_det_info(ground_bbox_list, det_bbox_list)
        print(tp)
        print(fp)
        print(fn)


if __name__ == '__main__':
    # iou_mat = box_ops.iou(np.array([[0, 29.41, 100.0, 44.45]]), np.array([[0, 29.41, 270.43, 52.4]]))
    # print(iou_mat)
    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    PATH_TO_FROZEN_GRAPH = '/Users/rensike/Resources/models/tensorflow/object_detection/ssd_mobilenet_v1_voc/frozen_inference_graph.pb'
    # List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS = os.path.join('../data', 'pascal_label_map.pbtxt')
    det_result_process = DetectionResultProcess(PATH_TO_FROZEN_GRAPH, PATH_TO_LABELS)

    # image_path = "/Users/rensike/Files/temp/voc_mini/JPEGImages/2008_001928.jpg"
    # anno_path = "/Users/rensike/Files/temp/voc_mini/Annotations/2008_001928.xml"
    #
    # det_result_process.test(image_path,anno_path)

    import mmcv

    image_list = mmcv.list_from_file("/Users/rensike/Files/temp/voc_mini/ImageSets/Main/val.txt")
    image_path_list = [os.path.join("/Users/rensike/Files/temp/voc_mini/JPEGImages", image + ".jpg") for image in
                       image_list]
    anno_path_list = [os.path.join("/Users/rensike/Files/temp/voc_mini/Annotations", image + ".xml") for image in
                      image_list]
    # det_result_process.make_vis_result(image_path_list, anno_path_list, "/Users/rensike/Files/temp/det_process_result")

    # det_result_process.calc_det_info(image_path_list, anno_path_list,
    #                                  "/Users/rensike/Files/temp/det_process_result/det_info.txt")

    det_result_process.make_evaluation(image_path_list,anno_path_list,"/Users/rensike/Files/temp/det_process_result/evaluation.txt")
