# -*- coding: utf-8 -* -
'''
从Tensorflow Slim预训练模型的检查点中恢复模型，并进行预测，最后生成pbtxt
'''
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import inception
import numpy as np

inception_v2, inception_v2_arg_scope = inception.inception_v2, inception.inception_v2_arg_scope

height = 224
width = 224
channels = 3

# Create graph
X = tf.placeholder(tf.float32, shape=[None, height, width, channels])
with slim.arg_scope(inception_v2_arg_scope()):
    logits, end_points = inception_v2(X, num_classes=1001, is_training=False)
predictions = end_points["Predictions"]
saver = tf.train.Saver()

X_test = np.ones((1, height, width, channels))  # a fake image, you can use your own image

# Execute graph
with tf.Session() as sess:
    saver.restore(sess, "./inception_v2.ckpt")
    predictions_val = predictions.eval(feed_dict={X: X_test})
    tf.train.write_graph(sess.graph_def, './', 'inception_v2.pbtxt')
