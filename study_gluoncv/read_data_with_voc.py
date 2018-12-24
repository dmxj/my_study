#encoding:utf-8

"""
参考：https://gluon-cv.mxnet.io/build/examples_datasets/pascal_voc.html
"""
from gluoncv import data, utils
from matplotlib import pyplot as plt
import os
import random

# 数据根路径默认为：～/.mxnet/datasets/voc，可以设定root参数来改变
# train_dataset = data.VOCDetection(splits=[(2007, 'trainval'), (2012, 'trainval')])
# val_dataset = data.VOCDetection(splits=[(2007, 'test')])
train_dataset = data.VOCDetection(root=os.path.join("/Users/rensike/Files/data/", "voc/VOCdevkit"), splits=[(2007, 'trainval')])
val_dataset = data.VOCDetection(root=os.path.join("/Users/rensike/Files/data/", "voc/VOCdevkit"), splits=[(2007, 'val')])
print("Number of training images:", len(train_dataset))
print("Number of validation images:", len(val_dataset))

# 选择一个示例进行可视化
train_image, train_label = random.choice(train_dataset)
print("Image size (height, width, RGB):", train_image.shape)

# 获取图片的bounding box
bounding_boxes = train_label[:, :4]
print("Num if objects:", bounding_boxes.shape[0])
print('Bounding boxes (num_boxes, x_min, y_min, x_max, y_max):\n', bounding_boxes)

# 获取图片的各个框的class id
class_ids = train_label[:, 4:5]
print("Class IDs (num_boxes, ):\n", class_ids)

# 可视化实例
utils.viz.plot_bbox(train_image.asnumpy(), bounding_boxes, scores=None, labels=class_ids, class_names=train_dataset.classes)
plt.show()