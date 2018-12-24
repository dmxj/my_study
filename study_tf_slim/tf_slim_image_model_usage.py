# -*- coding: utf-8 -* -
'''
https://github.com/tensorflow/models/blob/master/research/slim/README.md#installing-the-tf-slim-image-models-library
使用TF-Slim进行图像分类，还需要安装：TF-slim image models library，这个库不在TF的核心库中。
（1）git clone https://github.com/tensorflow/models
（2）cd models/research/slim
（3）测试是否可用：python -c "from nets import cifarnet; mynet = cifarnet.cifarnet"
注意：models/research/slim这个目录可以取出来放到单独放到一个地方直接使用。
可以查看源码目录中的各种*_test.py学习不同模型的使用
'''
import sys
# 我将slim文件夹放到了/Users/rensike/Files/temp/目录下
sys.path.append("/Users/rensike/Files/temp/slim")

import tensorflow as tf

slim = tf.contrib.slim

'''
获取pnasnet最后的输出层的变量名称
'''
from nets.nasnet import pnasnet

batch_size = 5
height, width = 331, 331
num_classes = 1000
inputs = tf.random_uniform((batch_size, height, width, 3))
tf.train.create_global_step()
with slim.arg_scope(pnasnet.pnasnet_large_arg_scope()):
  logits, end_points = pnasnet.build_pnasnet_large(inputs, num_classes)
auxlogits = end_points['AuxLogits']
predictions = end_points['Predictions']

print(predictions.op.name) # final_layer/predictions








