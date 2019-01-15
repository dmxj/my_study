#encoding:utf-8

"""
参考：https://gluon-cv.mxnet.io/build/examples_detection/demo_faster_rcnn.html
"""

from matplotlib import pyplot as plt
import gluoncv
from gluoncv import model_zoo, data, utils
import os
import numpy as np
import random

# 加载faster rcnn预训练模型(默认保存到：~/.mxnet/models)
net = model_zoo.get_model("faster_rcnn_resnet50_v1b_voc", pretrained=True)

# 准备图片，并预处理
# im_fname = utils.download('https://github.com/dmlc/web-data/blob/master/gluoncv/detection/biking.jpg?raw=true',path='biking.jpg')
img_path = "./images"
im_fname = [os.path.join(img_path,"img1.jpg"),os.path.join(img_path,"img2.jpg"),os.path.join(img_path,"img3.jpg")]
x, orig_img = data.transforms.presets.rcnn.load_test(im_fname)

# 预测推理
# the shape of box_ids is (batch_size, num_bboxes, 1)
# the shape of score is (batch_size, num_bboxes, 1)
# the shape of bboxes is (batch_size, num_bboxes, 4)
# box_ids, scores, bboxes = net(x[0])

# 显示图片
for ix in range(len(im_fname)):
    box_ids, scores, bboxes = net(x[ix])
    ax = utils.viz.plot_bbox(orig_img[ix], bboxes[0], scores[0], box_ids[0], class_names=net.classes)
    plt.show()