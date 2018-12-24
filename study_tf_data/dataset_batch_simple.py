# -*- coding: utf-8 -* -
'''
批处理数据集元素。简单的批处理
'''
import tensorflow as tf

'''
最简单的批处理形式是将数据集中的 n 个连续元素堆叠为一个元素。
Dataset.batch() 转换正是这么做的，它与 tf.stack() 运算符具有相同的限制（被应用于元素的每个组件）：即对于每个组件 i，所有元素的张量形状都必须完全相同。
'''

inc_dataset = tf.data.Dataset.range(100)
dec_dataset = tf.data.Dataset.range(0,-100,-1)
dataset = tf.data.Dataset.zip((inc_dataset,dec_dataset))
batches_dataset = dataset.batch(4)

iterator = batches_dataset.make_one_shot_iterator()
next_element = iterator.get_next()

with tf.Session() as sess:
    print(sess.run(next_element))   # ==> ([0, 1, 2,   3],   [ 0, -1,  -2,  -3])
    print(sess.run(next_element))   # ==> ([4, 5, 6,   7],   [-4, -5,  -6,  -7])
    print(sess.run(next_element))   # ==> ([8, 9, 10, 11],   [-8, -9, -10, -11])








