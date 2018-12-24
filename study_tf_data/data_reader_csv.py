# -*- coding: utf-8 -* -
'''
读取CSV输入数据到dataset中
'''
import tensorflow as tf

# =================？？执行错误？？===============

filenames = ["./my_csv.csv"]
'''
my_csv总共五列（姓名、年龄、身高、分数、是否婚配）；
record_defaults定义其类型或者默认值（当该列的值缺失时）
'''
record_defaults = [tf.string,tf.int8,tf.float16,0.0,tf.bool]
dataset = tf.contrib.data.CsvDataset(filenames,record_defaults,header=True,select_cols=[0,1,2,3,4])
iterator = dataset.make_one_shot_iterator()
line_element = iterator.get_next()

with tf.Session() as sess:
    while True:
        try:
            print(sess.run(line_element))
        except tf.errors.OutOfRangeError:
            break














