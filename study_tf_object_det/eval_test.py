# -*- coding: utf-8 -* -
'''
测试模型评估
'''
import os
from utils import label_map_util
from utils import object_detection_evaluation
from metrics import coco_evaluation
import numpy as np
from prettytable import PrettyTable

image_info = {
    "01.jpg":{
        "groundtruth_classes":np.array([1,2,2]),
        "groundtruth_boxes":np.array([
            [29.42, 62.6,269.55, 180.58],
            [7.9, 17.31,102.49, 118.47],
            [0, 29.41, 100.0, 44.45]
        ]),
    },
    "02.jpg":{
        "groundtruth_classes": np.array([3, 4]),
        "groundtruth_boxes": np.array([
            [3.88, 0, 287.98, 251.13],
            [98.61, 60.75,263.24, 172.56]
        ]),
    }
}

image_detection = {
    "01.jpg":{
        "detection_classes":np.array([1,2,2]),
        "detection_masks":np.array([
            [282.07, 241.93, 276.44, 218.14, 285.2, 211.88, 284.57, 210, 272.68, 199.99, 269.55, 193.73, 270.18, 186.21, 275.18, 181.83, 277.69, 180.58, 285.83, 180.58, 288.33, 182.46, 292.09, 184.96, 293.96, 188.72, 296.47, 191.22, 296.47, 193.1, 298.35, 198.73, 298.35, 199.36, 298.97, 201.86, 295.84, 206.25, 292.71, 207.5, 291.46, 209.38, 291.46, 210, 298.97, 217.51, 295.22, 243.18],
            [110.39, 135.78, 110.39, 127.62, 110.01, 119.6, 106.87, 118.47, 104.37, 120.1, 102.49, 122.73, 103.74,
             125.49, 105.24, 128.88, 106.87, 132.39, 107.38, 135.78, 110.39, 135.65],
            [308.51, 16.72, 300.7, 29.41, 281.82, 28.44, 278.57, 14.12, 287.03, 7.61, 279.22, 5.98, 277.59, 4.03,
             270.43, 0.45, 318.92, 0, 322.83, 7.93, 314.69, 12.82, 312.09, 13.79, 308.51, 15.42]
                                    ]),
        "detection_boxes":np.array([
            [29.42, 62.6, 269.55, 180.58],
            [7.9, 17.31, 102.49, 118.47],
            [0, 29.41, 270.43, 52.4]
        ]),
        "detection_scores":np.array([86.7,99.5,49.2])
    },
    "02.jpg":{
        "detection_classes": np.array([3, 4]),
        "detection_masks":np.array([
            [4.46, 19.19, 12.67, 95.55, 16.77, 133.31, 33.94, 244.34, 45.57, 251.13, 170.65, 231.74, 291.86, 197.8,
             275.37, 1.94, 85.33, 0, 3.88, 4.85, 7.76, 39.75],
            [267.65, 198.97, 269.41, 172.56, 361.85, 178.72, 358.33, 231.55, 292.3, 233.31, 263.24, 233.31, 267.65,
             194.57]
        ]),
        "detection_boxes":np.array([
            [3.88, 0, 287.98, 251.13],
            [98.61, 60.75, 263.24, 172.56]
        ]),
        "detection_scores": np.array([78.7, 94.5])
    }
}

# 共有4个类别
evaluator = object_detection_evaluation.ObjectDetectionEvaluation(4)

for image_name in image_info:
    evaluator.add_single_ground_truth_image_info(image_name,
                                                 image_info[image_name]["groundtruth_boxes"],
                                                 image_info[image_name]["groundtruth_classes"])

for image_name in image_detection:
    evaluator.add_single_detected_image_info(image_name,
                                             image_detection[image_name]["detection_boxes"],
                                             image_detection[image_name]["detection_scores"],
                                             image_detection[image_name]["detection_classes"]
                                             )
metrics = evaluator.evaluate()
print(metrics)







