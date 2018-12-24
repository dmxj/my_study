# -*- coding: utf-8 -* -
'''
使用 Dataset.map() 预处理数据。
Dataset.map(f) 转换通过将指定函数 f 应用于输入数据集的每个元素来生成新数据集。
此转换基于 map() 函数（通常应用于函数式编程语言中的列表和其他结构）。
函数 f 会接受表示输入中单个元素的 tf.Tensor 对象，并返回表示新数据集中单个元素的 tf.Tensor 对象。
此函数的实现使用标准的 TensorFlow 指令将一个元素转换为另一个元素。
'''
import tensorflow as tf
'''
解析 tf.Example 协议缓冲区消息。
许多输入管道都从 TFRecord 格式的文件中提取 tf.train.Example 协议缓冲区消息（例如这种文件使用 tf.python_io.TFRecordWriter 编写而成）。
每个 tf.train.Example 记录都包含一个或多个“特征”，输入管道通常会将这些特征转换为张量。
'''

def _parse_function(example_proto):
    features = {
        "x":tf.FixedLenSequenceFeature((1,2),tf.int64,allow_missing=True),
        "y":tf.FixedLenSequenceFeature((1,2),tf.float32,allow_missing=True)
    }
    parsed_features = tf.parse_single_example(example_proto,features)
    return parsed_features["x"],parsed_features["y"]

filenames = ["./train.tfrecords"]
dataset = tf.data.TFRecordDataset(filenames)
dataset = dataset.map(_parse_function)
iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()

with tf.Session() as sess:
    while True:
        try:
            x,y = sess.run(next_element)
            print("x:",x)
            print("y:",y)
        except tf.errors.OutOfRangeError:
            break


