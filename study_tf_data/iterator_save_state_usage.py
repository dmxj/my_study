# -*- coding: utf-8 -* -
'''
保存迭代器的状态
'''
import tensorflow as tf

dataset = tf.data.Dataset.range(10)
iterator = dataset.make_one_shot_iterator()
next = iterator.get_next()

# Create saveable object from iterator.
saveable = tf.contrib.data.make_saveable_from_iterator(iterator)

# Save the iterator state by adding it to the saveable objects collections
tf.add_to_collection(tf.GraphKeys.SAVEABLE_OBJECTS,saveable)
saver = tf.train.Saver()

with tf.Session() as sess:
    print(sess.run(next))
    print(sess.run(next))

    saver.save(sess,"./ckpt/")
    print("save iterator to disk")

with tf.Session() as sess:
    print("restore iterator from disk")
    saver.restore(sess,"./ckpt/")
    print(sess.run(next))
    print(sess.run(next))






