# -*- coding: utf-8 -* -
'''
使用Slim从训练好的额模型ckpt中选择部分变量进行恢复
'''
import tensorflow as tf
import tensorflow.contrib.slim as slim

# Create some variables.
v1 = slim.variable(name="v1",shape=[1,2,2],dtype=tf.float16)
v2 = slim.variable(name="nested/v2",dtype=tf.int8)
# ... ...

# 获取需要恢复的变量的列表
# 获取名称是v2的变量
variables_to_restore_0 = slim.get_variables_by_name("v2")

# 获取名称以2为结尾的变量
variables_to_restore_1 = slim.get_variables_by_suffix("2")

# 获取名称的命名空间在"nested"下的变量
variables_to_restore_2 = slim.get_variables(scope="nested")

# 获取名称包含"nested"的变量
variables_to_restore_3 = slim.get_variables_to_restore(include=["nested"])

# 获取名称排除"v1"的变量
variables_to_restore_4 = slim.get_variables_to_restore(exclude=["v1"])

# Create the saver which will be used to restore the variables.
restorer = tf.train.Saver(variables_to_restore_0)

with tf.Session() as sess:
    # Restore variables from disk.
    restorer.restore(sess,"/tmp/model.ckpt")
    print("Model restored.")
    # Do some work with the model
