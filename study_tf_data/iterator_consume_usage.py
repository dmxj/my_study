# -*- coding: utf-8 -* -
'''
消耗迭代器中的值
'''
import tensorflow as tf

'''
1. Iterator.get_next() 方法返回一个或多个 tf.Tensor 对象，这些对象对应于迭代器有符号的下一个元素。
每次评估这些张量时，它们都会获取底层数据集中下一个元素的值。
2. 请注意，与 TensorFlow 中的其他有状态对象一样，调用 Iterator.get_next() 并不会立即使迭代器进入下个状态。
您必须在 TensorFlow 表达式中使用此函数返回的 tf.Tensor 对象，
并将该表达式的结果传递到 tf.Session.run()，以获取下一个元素并使迭代器进入下个状态。
3. 如果迭代器到达数据集的末尾，则执行 Iterator.get_next() 操作会产生 tf.errors.OutOfRangeError。
在此之后，迭代器将处于不可用状态；如果需要继续使用，则必须对其重新初始化。
'''

dataset = tf.data.Dataset.range(5)
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()

# Typically `result` will be the output of a model, or an optimizer's
# training operation.
result = tf.add(next_element, next_element)

# 版本1的写法
with tf.Session() as sess:
    sess.run(iterator.initializer)
    print(sess.run(result))  # ==> "0"
    print(sess.run(result))  # ==> "2"
    print(sess.run(result))  # ==> "4"
    print(sess.run(result))  # ==> "6"
    print(sess.run(result))  # ==> "8"
    try:
        sess.run(result)
    except tf.errors.OutOfRangeError:
        print("End of dataset")  # ==> "End of dataset"

# 版本2的写法
with tf.Session() as sess:
    sess.run(iterator.initializer)
    while True:
        try:
            sess.run(result)
        except tf.errors.OutOfRangeError:
            break

'''
如果数据集的每个元素都具有嵌套结构，则 Iterator.get_next() 的返回值将是一个或多个 tf.Tensor 对象，这些对象具有相同的嵌套结构。
'''
dataset1 = tf.data.Dataset.from_tensor_slices(tf.reshape(tf.range(40),[4,10]))
dataset2 = tf.data.Dataset.from_tensor_slices((tf.range(4), tf.reshape(tf.range(40),[4,10])))
dataset3 = tf.data.Dataset.zip((dataset1, dataset2))

iterator = dataset3.make_initializable_iterator()
next1, (next2, next3) = iterator.get_next()

'''
请注意，next1、next2 和 next3 是由同一个操作/节点（通过 Iterator.get_next() 创建）生成的张量。
因此，评估其中任何一个张量都会使所有组件的迭代器进入下个状态。
典型的迭代器消耗方会在一个表达式中包含所有组件。
'''

# 这种写法有问题
with tf.Session() as sess:
    sess.run(iterator.initializer)
    print("next1:",sess.run(next1))  # => [0 1 2 3 4 5 6 7 8 9]
    print("next2:",sess.run(next2))  # => 1
    print("next3:",sess.run(next3))  # => [20 21 22 23 24 25 26 27 28 29]

print("正确的写法：")
# 应该使用这种写法
with tf.Session() as sess:
    sess.run(iterator.initializer)
    val1,val2,val3 = sess.run([next1,next2,next3])
    print("next1:",val1)
    print("next2:",val2)
    print("next3:",val3)
