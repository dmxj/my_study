# -*- coding: utf-8 -* -
'''
批处理数据集元素。使用填充批处理张量。
'''
import tensorflow as tf

'''
使用dataset_batch_simple.py的方法处理适合具有相同大小的张量。
不过，很多模型（例如序列模型）处理的输入数据可能具有不同的大小（例如序列的长度不同）。
为了解决这种情况，可以通过 Dataset.padded_batch() 转换来指定一个或多个会被填充的维度，从而批处理不同形状的张量。
'''

dataset = tf.data.Dataset.range(100)
dataset = dataset.map(lambda x:tf.fill([tf.cast(x,tf.int32)],x))
dataset = dataset.padded_batch(4,padded_shapes=[None])

iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()

with tf.Session() as sess:
    print(sess.run(next_element))  # ==> [[0, 0, 0], [1, 0, 0], [2, 2, 0], [3, 3, 3]]
    print(sess.run(next_element))  # ==> [[4, 4, 4, 4, 0, 0, 0],
                                   #      [5, 5, 5, 5, 5, 0, 0],
                                   #      [6, 6, 6, 6, 6, 6, 0],
                                   #      [7, 7, 7, 7, 7, 7, 7]]









