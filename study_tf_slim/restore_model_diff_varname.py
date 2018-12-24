# -*- coding: utf-8 -* -
'''
使用Slim恢复变量，当checkpoint中的变量名称和当前代码的graph中的变量名称不一致时，需要使用一个字典将checkpoint中的变量名称映射到
'''
import tensorflow as tf
import tensorflow.contrib.slim as slim

# 假设在当前graph中的变量名字是'conv1/weights',而checkpoint文件中的变量名字是'vgg16/conv1/weights'
def name_in_checkpoint(var):
    return "vgg16/" + var.op.name

# 假设在当前graph中的变量名字是"conv1/weights"、"conv1/bias"的形式，而checkpoint文件中的变量明名字是"conv1/params1"、"conv1/params2"的格式
def name_in_checkpoint_v2(var):
    if "weights" in var.op.name:
        return var.op.name.replace("weights","params1")
    if "bias" in var.op.name:
        return var.op.name.replace("bias","params2")

variables_to_restore = slim.get_model_variables()
# checkpoint中的变量名称 => 当前图中的变量名称
variables_to_restore = {name_in_checkpoint(var):var for var in variables_to_restore}

restorer = tf.train.Saver(variables_to_restore)

with tf.Session() as sess:
    # Restore variables from disk
    restorer.restore(sess,"/tmp/model.ckpt")












