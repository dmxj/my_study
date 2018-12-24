# -*- coding: utf-8 -* -
'''
使用arg scope在不同的层之间共享参数值
'''
import tensorflow.contrib.slim as slim
import tensorflow as tf

input = slim.variable("input", [1, 28, 28, 3], weights_initializer=tf.truncated_normal_initializer(stddev=0.01))

# 以下三个卷积层共用很多超参数，读起来比较晦涩
net = slim.conv2d(input, 64, [11, 11], 4, padding='SAME',
                  weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                  weights_regularizer=slim.l2_regularizer(0.0005), scope='conv1')
net = slim.conv2d(net, 128, [11, 11], padding='VALID',
                  weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                  weights_regularizer=slim.l2_regularizer(0.0005), scope='conv2')
net = slim.conv2d(net, 256, [11, 11], padding='SAME',
                  weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                  weights_regularizer=slim.l2_regularizer(0.0005), scope='conv3')

# 可以将相同的参数提取出来公用，但是代码仍然不够清晰
padding = "SAME"
initializer = tf.truncated_normal_initializer(stddev=0.01)
regularizer = slim.l2_regularizer(0.0005)
net2 = slim.conv2d(input, 64, [11, 11], 4,
                   padding=padding,
                   weights_initializer=initializer,
                   weights_regularizer=regularizer,
                   scope='conv1')
net2 = slim.conv2d(net2, 128, [11, 11],
                   padding='VALID',
                   weights_initializer=initializer,
                   weights_regularizer=regularizer,
                   scope='conv2')
net2 = slim.conv2d(net2, 256, [11, 11],
                   padding=padding,
                   weights_initializer=initializer,
                   weights_regularizer=regularizer,
                   scope='conv3')

# 通过使用一个 arg_scope，我们能够在保证每一层使用相同参数值的同时，简化代码
with slim.arg_scope([slim.conv2d],
                    padding="SAME",
                    weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                    weights_regularizer=slim.l2_regularizer(0.0005)):
    conv = slim.conv2d(input, 64, [11, 11], scope="conv1")
    conv = slim.conv2d(conv, 128, [11, 11], padding='VALID', scope="conv2")
    conv = slim.conv2d(conv, 256, [11, 11], scope="conv3")

# 也可以嵌套地使用 arg_scope，并且在同一个 scope 中可以使用多个 op
# 第一个 arg_scope 中对 conv2d、fully_connected 层使用相同的 weights_initializer
# 在第二个 arg_scope 中，给 conv2d 的其它默认参数进行了指定
with slim.arg_scope([slim.conv2d, slim.fully_connected],
                    activation_fn=tf.nn.relu,
                    weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                    weights_regularizer=slim.l2_regularizer(0.0005)):
    with slim.arg_scope([slim.conv2d], stride=1, padding='SAME'):
        conv2 = slim.conv2d(input, 64, [11, 11], 4, padding='VALID', scope='conv1')
        conv2 = slim.conv2d(conv2, 256, [5, 5],
                            weights_initializer=tf.truncated_normal_initializer(stddev=0.03),
                            scope='conv2')
        conv2 = slim.fully_connected(conv2, 1000, activation_fn=None, scope='fc')
