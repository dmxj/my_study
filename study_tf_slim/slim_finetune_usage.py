# -*- coding: utf-8 -* -
'''
迁移学习。
假设有一个在ImageNet上训练好的1000类模型的checkpoint，但是想要应用到只有10类的mnist数据集中，需要迁移学习
'''
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets

# Params define
learning_rate = 0.0001
train_log_dir = "./log/train_model/"

if not tf.gfile.Exists(train_log_dir):
    tf.gfile.MakeDirs(train_log_dir)

# Load mnist Data
mnist = input_data.read_data_sets("/tmp/mnist_dataset")
images,labels = mnist.train.images,mnist.train.labels

# Create the model
vgg = nets.vgg
predictions = vgg.vgg_16(images)

# Define the loss functions and get the total loss.
loss = tf.losses.softmax_cross_entropy(predictions, labels)
tf.summary.scalar("losses/total_loss", loss)

# Define optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

# create_train_op ensures that each time we ask for the loss, the update_ops
# are run and the gradients being computed are applied too.
train_op = slim.learning.create_train_op(total_loss=loss, optimizer=optimizer)

# 只从checkpoint中恢复卷积网络的部分
model_path = "/path/to/pre_trained_on_imagenet.checkpoint" # 已经被保存到本地的pre trained 模型
variables_to_restore = slim.get_variables_to_restore(exclude=["fc6","fc7","fc8"])
init_fn = slim.assign_from_checkpoint_fn(model_path,variables_to_restore)

# 从checkpoint中恢复的卷积部分作为初始参数
slim.learning.train(
    train_op,
    logdir=train_log_dir,
    init_fn=init_fn
)










