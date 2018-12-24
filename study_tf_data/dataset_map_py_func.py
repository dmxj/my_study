# -*- coding: utf-8 -* -
'''
使用 Dataset.map() 预处理数据。使用 tf.py_func() 应用任意 Python 逻辑
'''
import tensorflow as tf
import cv2

'''
为了确保性能，我们建议您尽可能使用 TensorFlow 指令预处理数据。不过，在解析输入数据时，调用外部 Python 库有时很有用。
为此，请在 Dataset.map() 转换中调用 tf.py_func() 指令。
'''

# Use a custom OpenCV function to read the image, instead of the standard
# TensorFlow `tf.read_file()` operation.
def _read_by_function(filename,label):
    image_decoded = cv2.imread(filename.decode(),cv2.IMREAD_GRAYSCALE)
    return image_decoded,label

# Use standard TensorFlow operations to resize the image to a fixed shape.
def _resize_function(image_decoded,label):
    image_decoded.set_shape([None,None,None])
    image_resized = tf.image.resize_images(image_decoded,[28,28])
    return image_resized,label

# A vector of filenames.
filenames = tf.constant(["./imgs/img0.jpg","./imgs/img1.jpg","./imgs/img2.jpeg"])

# `labels[i]` is the label for the image in `filenames[i].
labels = tf.constant([0,1,2])

dataset = tf.data.Dataset.from_tensor_slices((filenames,labels))
dataset = dataset.map(
    lambda filename,label:tuple(
        tf.py_func(_read_by_function,[filename,label],[tf.uint8,label.dtype])
    )
)
dataset = dataset.map(_resize_function)









