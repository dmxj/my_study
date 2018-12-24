# -*- coding: utf-8 -* -
'''
从头开始训练inception v2，使用手写字体数据集
'''
import tensorflow as tf
from tensorflow.contrib.slim import nets
from tensorflow.keras import datasets
from tensorflow.contrib.slim.python.slim.learning import train_step
import os
import numpy as np

slim = tf.contrib.slim

'''加载手写字体数据集'''
datasets_prefix = "/Users/rensike/Resources/datasets/keras"
(mnist_x_train, mnist_y_train), (mnist_x_test, mnist_y_test) = datasets.mnist.load_data(
    os.path.join(datasets_prefix, "mnist.npz"))
mnist_x_train = np.expand_dims(mnist_x_train, 3)
mnist_x_test = np.expand_dims(mnist_x_test, 3)

mnist_x_train = mnist_x_train + tf.zeros(shape=mnist_x_train.shape)
mnist_y_train = mnist_y_train + tf.zeros(shape=mnist_y_train.shape)
mnist_x_test = mnist_x_test + tf.zeros(shape=mnist_x_test.shape)
mnist_y_test = mnist_y_test + tf.zeros(shape=mnist_y_test.shape)

print("手写字体数据集，训练集X shape:", mnist_x_train.shape)  # (60000, 28, 28, 1)
print("手写字体数据集，训练集Y shape:", mnist_y_train.shape)  # (60000,)
print("手写字体数据集，测试集X shape:", mnist_x_test.shape)  # (10000, 28, 28, 1)
print("手写字体数据集，测试集Y shape:", mnist_y_test.shape)  # (10000,)

'''定义训练和测试数据集'''
train_dataset = tf.data.Dataset.from_tensor_slices((mnist_x_train, mnist_y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((mnist_x_test, mnist_y_test))

train_iterator = train_dataset.batch(32).make_one_shot_iterator()
train_batch_element = train_iterator.get_next()

'''定义超参数'''
epochs = 5
learning_rate = 0.0001
train_log_dir = "./result/train_from_scratch_model_dir/"
if os.path.exists(train_log_dir):
    os.mkdir(train_log_dir)
validation_every_n_step = 200
test_every_n_step = 100

'''定义训练时执行的操作'''


def train_step_fn(session, *args, **kwargs):
    total_loss, should_stop = train_step(session, *args, **kwargs)

    if train_step_fn.step % validation_every_n_step == 0:
        accuracy = sess.run(train_step_fn.accuracy_validation)
        print("Validation Accuracy:", accuracy)
        if accuracy >= 0.993:
            print("准确率达到了99.3%，退出训练")

    if train_step_fn.step % test_every_n_step == 0:
        accuracy = session.run(train_step_fn.accuracy_test)
        print("Test Accuracy:", accuracy)

    train_step_fn.step += 1
    return [total_loss, should_stop]


train_step_fn.step = 0
logits, end_points = nets.inception.inception_v2(mnist_x_test, 10)
predictions_validation = tf.argmax(logits, 1)
labels_validation = tf.squeeze(mnist_y_test)
_, train_step_fn.accuracy_validation = tf.metrics.accuracy(predictions=predictions_validation, labels=labels_validation)
_, train_step_fn.accuracy_test = tf.metrics.accuracy(predictions=predictions_validation, labels=labels_validation)

'''训练epochs轮'''
for i in range(epochs):
    print("now start traing epoch ({}/{})".format(i + 1, epochs))
    with tf.Session() as sess:

        while True:
            try:
                train_batch_data, train_batch_target = sess.run(train_batch_element)
                # Create the model
                logits, end_points = nets.inception.inception_v2(train_batch_data, 10)

                # Define the loss functions and get the total loss.
                loss = tf.losses.softmax_cross_entropy(logits, train_batch_target)

                # Define optimizer
                optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

                # Define train op
                train_op = slim.learning.create_train_op(total_loss=loss, optimizer=optimizer)

                # train_op 主要进行两个操作：（a）计算loss；（b）进行梯度更新
                slim.learning.train(
                    train_op,
                    logdir=train_log_dir,
                    save_interval_secs=300,  # 每600秒保存一次model checkpoint
                    train_step_fn=train_step_fn  # 训练过程中执行
                )

            except tf.errors.OutOfRangeError:
                break
