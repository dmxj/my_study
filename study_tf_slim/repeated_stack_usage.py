# -*- coding: utf-8 -* -
'''
Slim使用Repeated和Stack实现层的重复
'''
import tensorflow.contrib.slim as slim
import tensorflow as tf

# 使用slim.repeat 重复net卷积操作3次
input = slim.variable("input_var",[1,28,28,3])
net = slim.conv2d(input,128,[3,3],
                  weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                  weights_regularizer=slim.l2_regularizer(0.0005),
                  scope="conv1_1")

net = slim.repeat(net,3,slim.conv2d,256,[3,3],scope="conv2")
net = slim.max_pool2d(net, [2, 2], scope='pool2')
'''
相当于：
net = slim.conv2d(net, 256, [3, 3], scope='conv2/conv2_1')
net = slim.conv2d(net, 256, [3, 3], scope='conv2/conv2_2')
net = slim.conv2d(net, 256, [3, 3], scope='conv2/conv2_3')
net = slim.max_pool2d(net, [2, 2], scope='pool2')
'''

# 使用slim.stack 重复全连接/卷积操作，但是可以使用不同的参数，相当于扩展网络
# 冗长的方式——全连接网络：
x = slim.variable("input_x",[-1,10],initializer=tf.truncated_normal_initializer())
x = slim.fully_connected(x, 32, scope='fc/fc_1')
x = slim.fully_connected(x, 64, scope='fc/fc_2')
x = slim.fully_connected(x, 128, scope='fc/fc_3')
# 相当于以下的stack堆叠(隐藏层分别是32、64、128)：
x = slim.stack(x, slim.fully_connected,[32,64,128],scope="fc")

# 冗长的方式——卷积神经网络
conv = slim.variable("input_img",[1,28,28,3])
conv = slim.conv2d(conv,32,[3,3],scope="core/core_1")
conv = slim.conv2d(conv,32,[3,3],scope="core/core_1")
conv = slim.conv2d(conv,32,[3,3],scope="core/core_1")
conv = slim.conv2d(conv,32,[3,3],scope="core/core_1")
# 相当于使用以下的stack堆叠的方式：
conv = slim.stack(conv,slim.conv2d,[(32, [3, 3]), (32, [1, 1]), (64, [3, 3]), (64, [1, 1])],scope="core")

