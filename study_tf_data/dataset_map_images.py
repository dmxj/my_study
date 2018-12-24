# -*- coding: utf-8 -* -
'''
使用 Dataset.map() 预处理数据。解码图片数据并调整其大小。
'''
import tensorflow as tf

'''
在用真实的图片数据训练神经网络时，通常需要将不同大小的图片转换为通用大小，这样就可以将它们批处理为具有固定大小的数据。
'''

# Reads an image from a file, decodes it into a dense tensor, and resizes it
# to a fixed shape.
def _parse_function(filename,label):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string)
    image_resized = tf.image.resize_images(image_decoded,[28,28])
    return image_resized,label

# A vector of filenames.
filenames = tf.constant(["./imgs/img0.jpg","./imgs/img1.jpg","./imgs/img2.jpeg"])

# `labels[i]` is the label for the image in `filenames[i].
labels = tf.constant([0,1,2])

dataset = tf.data.Dataset.from_tensor_slices((filenames,labels))
dataset = dataset.map(_parse_function)
iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()

with tf.Session() as sess:
    while True:
        try:
            img,label = sess.run(next_element)
            print("img:",img)
            print("label:",label)
        except tf.errors.OutOfRangeError:
            break


