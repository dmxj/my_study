# -*- coding: utf-8 -* -
'''
部分恢复模型文件中的变量
'''
import tensorflow as tf

tf.reset_default_graph()
# Create some variables.
v1 = tf.get_variable("v1", [3], initializer = tf.zeros_initializer)
v2 = tf.get_variable("v2", [5], initializer = tf.zeros_initializer)

# Add ops to save and restore only "v2" using the name "v2"
# 使用字典，只保存或者恢复部分变量，键代表保存的变量名称，可以自定义
saver = tf.train.Saver({"v2":v2})

# Use the saver object normally after that
with tf.Session() as sess:
    # Initialize v1 since the saver will not
    # 因为V1没有被保存|恢复，所以v1需要初始化
    v1.initializer.run()
    saver.restore(sess,"./models/1_save_vars/model.ckpt")

    print("v1 : %s" % v1.eval())
    print("v2 : %s" % v2.eval())
