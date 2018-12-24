# -*- coding: utf-8 -* -
'''
tensorflow slim基本使用方法
'''
import tensorflow as tf
import tensorflow.contrib.slim as slim

# 创建一个权重变量，名称为"weights"，用一个截断的正态分布初始化它，用 l2_loss 进行正则，并将它放在 CPU 上
weights_var = slim.variable('weights',
                        shape=[10, 10, 3 , 3],
                        initializer=tf.truncated_normal_initializer(stddev=0.1),
                        regularizer=slim.l2_regularizer(0.05),
                        device='/CPU:0')

# 通过model_variable来定义一个代表模型参数的变量，non-model变量指训练、评估过程中需要但推理过程不需要的变量（例如global step）
weights_model_var = slim.model_variable('weights',
                              shape=[10, 10, 3 , 3],
                              initializer=tf.truncated_normal_initializer(stddev=0.1),
                              regularizer=slim.l2_regularizer(0.05),
                              device='/CPU:0')
model_variables = slim.get_model_variables()

# 定义并获取一个常规的变量
my_var = slim.variable("my_var",
                       shape=[20,1],
                       initializer=tf.zeros_initializer())
regular_variables_and_model_variables = slim.get_variables()

# slim.model_variable将变量添加到了tf.GrapghKeys.MODEL_VARIABLES容器中，也可以手动将自定义的layer或variables添加到对应的容器中
my_custom_var = tf.Variable(1,name="my_custom_var",dtype=tf.int8)
slim.add_model_variable(my_custom_var)

# 定义卷积层
input = slim.variable("input_var",[1,28,28,3])
net = slim.conv2d(input,128,[3,3],
                  weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                  weights_regularizer=slim.l2_regularizer(0.0005),
                  scope="conv1_1")



