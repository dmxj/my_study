# -*- coding: utf-8 -* -
'''
读取输入数据。
输入数据可以是：NumPy 数组、TFRecord 数据、文本数据、CSV 数据
'''
import numpy as np
import tensorflow as tf

'''读取Numpy数组'''
npy_path = "/Users/rensike/Resources/datasets/keras/boston_housing.npz"
# Load the training data into two NumPy arrays, for example using `np.load()`.
with np.load(npy_path) as data:
    x = data["x"]
    y = data["y"]

assert x.shape[0] == y.shape[0]

dataset = tf.data.Dataset.from_tensor_slices((x, y))
'''
请注意，上面的代码段会将 features 和 labels 数组作为 tf.constant() 指令嵌入在 TensorFlow 图中。
这样非常适合小型数据集，但会浪费内存，因为会多次复制数组的内容，并可能会达到 tf.GraphDef 协议缓冲区的 2GB 上限。
'''

# =====================替代方案======================

'''
作为替代方案，您可以根据 tf.placeholder() 张量定义 Dataset，并在对数据集初始化 Iterator 时馈送 NumPy 数组。
'''
# Load the training data into two NumPy arrays, for example using `np.load()`.
with np.load(npy_path) as data:
    x = data["x"]
    y = data["y"]

assert x.shape[0] == y.shape[0]

x_placeholder = tf.placeholder(x.dtype, x.shape)
y_placeholder = tf.placeholder(y.dtype, y.shape)

dataset2 = tf.data.Dataset.from_tensor_slices((x_placeholder, y_placeholder))
iterator = dataset2.make_initializable_iterator()
next_element = iterator.get_next()

with tf.Session() as sess:
    sess.run(iterator.initializer, feed_dict={
        x_placeholder: x,
        y_placeholder: y
    })
    print("get next element:", sess.run(next_element))
