# -*- coding: utf-8 -* -
'''
创建TF-Record
'''
import tensorflow as tf
import os
from PIL import Image
import numpy as np

def int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_list_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def create_tf_records(data_dir,output_path):
    '''
    将图片的相关信息保存成tf-record，包括图片label、图像原生bytes、图片宽和高、图片的shape
    :param data_dir: 图片文件夹
    :param output_path: 保存的TF-Record路径
    :return:
    '''
    assert tf.gfile.Exists(data_dir),"data dir not exist!"

    with tf.python_io.TFRecordWriter(output_path) as writer:
        for img_name in os.listdir(data_dir):
            img_path = os.path.join(data_dir,img_name)
            if os.path.isfile(img_path):
                img_obj = Image.open(img_path)
                img_width,img_height = img_obj.size
                img_label = img_name.rsplit(".",1)[0]
                example = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            "label":int64_feature(int(img_label)),
                            "image_raw":bytes_feature(img_obj.tobytes()),
                            "width":int64_feature(img_width),
                            "height": int64_feature(img_height),
                            "shape":int64_list_feature(np.array(img_obj).shape)
                        }
                    )
                )
                writer.write(example.SerializeToString())



if __name__ == "__main__":
    data_dir = "./imgs/"
    output_path = "./results/imgs.tfrecords"
    create_tf_records(data_dir,output_path)


