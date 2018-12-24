# -*- coding: utf-8 -* -
'''
训练工作流程。随机重排输入数据。
'''
import tensorflow as tf

'''
Dataset.shuffle() 转换会使用类似于 tf.RandomShuffleQueue 的算法随机重排输入数据集：它会维持一个固定大小的缓冲区，并从该缓冲区统一地随机选择下一个元素。
'''
# 随机重排输入数据
dataset = tf.data.Dataset.range(2000)
dataset = dataset.shuffle(buffer_size=1000)
dataset = dataset.batch(10)
iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()

with tf.Session() as sess:
    print(sess.run(next_element))