# -*- coding: utf-8 -* -
'''
读取文本输入数据到dataset中
'''
import tensorflow as tf

'''
很多数据集都是作为一个或多个文本文件分布的。tf.data.TextLineDataset 提供了一种从一个或多个文本文件中提取行的简单方法。
给定一个或多个文件名，TextLineDataset 会为这些文件的每行生成一个字符串值元素。
像 TFRecordDataset 一样，TextLineDataset 将接受 filenames（作为 tf.Tensor），因此您可以通过传递 tf.placeholder(tf.string) 进行参数化。
'''
filenames = ["file1.txt","file2.txt"]
dataset = tf.data.TextLineDataset(filenames)
iterator = dataset.make_one_shot_iterator()
line_element = iterator.get_next()

with tf.Session() as sess:
    while True:
        try:
            print(sess.run(line_element))
        except tf.errors.OutOfRangeError:
            break

'''
默认情况下，TextLineDataset 会生成每个文件的每一行，这可能是不可取的（例如，如果文件以标题行开头或包含注释）。
可以使用 Dataset.skip() 和 Dataset.filter() 转换来移除这些行。
为了将这些转换分别应用于每个文件，我们使用 Dataset.flat_map() 为每个文件创建一个嵌套的 Dataset。
'''
print("\n======跳过不必要的数据行======\n")
dataset2 = tf.data.Dataset.from_tensor_slices(filenames)
# Use `Dataset.flat_map()` to transform each file as a separate nested dataset,
# and then concatenate their contents sequentially into a single "flat" dataset.
# * Skip the first line (header row).
# * Filter out lines beginning with "#" (comments).
dataset2 = dataset2.flat_map(
    lambda filename:(
        tf.data.TextLineDataset(filename)
            .skip(1)    # 跳过文件第一行
            .filter(lambda line:tf.not_equal(tf.substr(line,0,1),"#"))      # 跳过以#开头的行
    )
)
iterator2 = dataset2.batch(5).make_one_shot_iterator()
line_element2 = iterator2.get_next()
with tf.Session() as sess:
    while True:
        try:
            print(sess.run(line_element2))
        except tf.errors.OutOfRangeError:
            break

