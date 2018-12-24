# -*- coding: utf-8 -* -
'''
制作简单的tfrecord
'''
import tensorflow as tf
import numpy as np

train_x = np.arange(0,10).reshape((5,2))
val_x = np.arange(10,20).reshape((5,2))

train_y = train_x*2+np.random.random()
val_y = val_x*2+np.random.random()

print(train_x)
'''
[[0 1]
 [2 3]
 [4 5]
 [6 7]
 [8 9]]
'''
print(val_x)
'''
[[10 11]
 [12 13]
 [14 15]
 [16 17]
 [18 19]]
'''

train_writer = tf.python_io.TFRecordWriter("./train.tfrecords")
val_writer = tf.python_io.TFRecordWriter("./val.tfrecords")

'''训练集数据写入TfRecord'''
for i in range(train_x.shape[0]):
    example = tf.train.Example(features=tf.train.Features(feature={
        "x":tf.train.Feature(int64_list=tf.train.Int64List(value=train_x[i,:])),
        "y":tf.train.Feature(float_list=tf.train.FloatList(value=train_y[i]))
    }))
    train_writer.write(example.SerializeToString())
train_writer.close()

'''验证集数据写入TfRecord'''
for i in range(val_x.shape[0]):
    example = tf.train.Example(features=tf.train.Features(feature={
        "x":tf.train.Feature(int64_list=tf.train.Int64List(value=val_x[i,:])),
        "y":tf.train.Feature(float_list=tf.train.FloatList(value=val_y[i]))
    }))
    val_writer.write(example.SerializeToString())
val_writer.close()


