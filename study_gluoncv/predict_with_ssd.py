#encoding:utf-8

"""
参考：https://gluon-cv.mxnet.io/build/examples_detection/demo_ssd.html
"""

from gluoncv import model_zoo, data, utils
from matplotlib import pyplot as plt
import numpy as np
import mxnet

# 加载SSD模型(默认保存到：~/.mxnet/models)
net = model_zoo.get_model('ssd_512_resnet50_v1_voc', pretrained = True)

# 加载图片， 并做预处理
im_fname = utils.download('https://github.com/dmlc/web-data/blob/master/' +
                          'gluoncv/detection/street_small.jpg?raw=true',
                          path='images/street_small.jpg')
x, img = data.transforms.presets.ssd.load_test(im_fname, short=512)
print('Shape of pre-processed image:', x.shape)
print("type of x is:", type(x))

# 推理并显示图片
# the shape of class_IDs is (batch_size, num_bboxes, 1)
# the shape of scores is (batch_size, num_bboxes, 1)
# the shape of bounding_boxs is (batch_size, num_bboxes, 4)
class_IDs, scores, bounding_boxs = net(x)
print("whill show result")
ax = utils.viz.plot_bbox(img, bounding_boxs[0], scores[0], class_IDs[0], class_names=net.classes)

plt.savefig("ssd_det_result.jpg")

plt.show()
