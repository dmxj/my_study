# -*- coding: utf-8 -* -
'''
训练工作流程。处理多个周期。
'''
import tensorflow as tf

# 处理多个周期
dataset = tf.data.Dataset.range(5)
dataset = dataset.repeat(10)    # 将数据重复10个周期，如果repeat不填参数，将无限次地重复输入。
dataset = dataset.batch(20)     # 将数据按照batch size=20进行分割

'''
使用dataset.repeat()重复数据，无法获取一个周期结束的信号，如果想在每个周期结束时收到信号，可以使用下面的写法：
'''
dataset2 = tf.data.Dataset.range(5)
iterator = dataset2.make_one_shot_iterator()
next_element = iterator.get_next()
# Compute for 10 epochs
for _ in range(10):
    with tf.Session() as sess:
        while True:
            try:
                print("next_element:",sess.run(next_element))
            except tf.errors.OutOfRangeError:
                break

