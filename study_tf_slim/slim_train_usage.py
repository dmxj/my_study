# -*- coding: utf-8 -* -
'''
使用Slim提供的训练工具进行模型的训练。
Slim训练包含：重复地计算loss、计算梯度、将模型保存到disk、操作梯度等等。
'''
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets

# 使用VGG16进行训练
vgg = nets.vgg

# Params
learning_rate = 0.001
# where checkpoints are stored
train_log_dir = "./checkpoints/"
if not tf.gfile.Exists(train_log_dir):
    tf.gfile.MakeDirs(train_log_dir)

with tf.Graph().as_default():
    # Load the images and labels
    mnist = input_data.read_data_sets("/tmp/")
    images, labels = mnist.train.images, mnist.train.labels

    # Create the model
    predictions, _ = vgg.vgg_16(images)

    # Define the loss functions and get the total loss.
    loss = tf.losses.softmax_cross_entropy(predictions, labels)
    tf.summary.scalar("losses/total_loss", loss)

    # Define optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    # create_train_op ensures that each time we ask for the loss, the update_ops
    # are run and the gradients being computed are applied too.
    train_op = slim.learning.create_train_op(total_loss=loss, optimizer=optimizer)

    # train_op 主要进行两个操作：（a）计算loss；（b）进行梯度更新
    slim.learning.train(
        train_op,
        logdir=train_log_dir,
        number_of_steps=1000,  # 执行1000步梯度下降
        save_summaries_secs=300,  # 每300秒保存一次summaries,
        save_interval_secs=600  # 每600秒保存一次model checkpoint
    )




