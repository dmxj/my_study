# -*- coding: utf-8 -* -
'''
从tfrecord中读取数据到dataset中
'''
import tensorflow as tf

'''
TFRecordDataset 初始化程序的 filenames 参数可以是字符串、字符串列表，也可以是字符串 tf.Tensor。
因此，如果您有两组分别用于训练和验证的文件，则可以使用 tf.placeholder(tf.string) 来表示文件名，并使用适当的文件名初始化迭代器：
'''

filenames = tf.placeholder(tf.string,shape=[None])
dataset = tf.data.TFRecordDataset(filenames=filenames)
dataset = dataset.repeat()
dataset = dataset.batch(2)
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()

# You can feed the initializer with the appropriate filenames for the current
# phase of execution, e.g. training vs. validation.

# Initialize `iterator` with training data.
with tf.Session() as sess:
    training_filenames = ["train.tfrecords"]
    sess.run(iterator.initializer,feed_dict={filenames:training_filenames})
    next_train_data = sess.run(next_element)
    print(next_train_data)

# Initialize `iterator` with val data.
with tf.Session() as sess:
    val_filenames = ["val.tfrecords"]
    sess.run(iterator.initializer,feed_dict={filenames:val_filenames})
    next_val_data = sess.run(next_element)
    print(next_val_data)


