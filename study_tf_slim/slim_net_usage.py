# -*- coding: utf-8 -* -
'''
使用Slim内置的网络
'''
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow.contrib.slim.nets as nets

vgg = nets.vgg
'''
alextnet = nets.alexnet
inception = nets.inception
resnet_v1 = nets.resnet_v1
resnet_v2 = nets.resnet_v2
overfeat = nets.overfeat
'''

# Load the images and labels
mnist = input_data.read_data_sets("/tmp/")
images,labels = mnist.train.images, mnist.train.labels

# Create the model
predictions, _ = vgg.vgg_16(images)

# Define the loss functions and get the total loss.
loss = tf.losses.softmax_cross_entropy(predictions, labels)


