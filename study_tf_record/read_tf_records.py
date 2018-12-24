# -*- coding: utf-8 -* -
'''
读取Tf-Record
'''
import tensorflow as tf
import numpy as np

def read_tf_records(tfrecord_path,flatten=True):
    images = []
    labels = []

    for serialized_example in tf.python_io.tf_record_iterator(tfrecord_path):
        example = tf.train.Example()
        example.ParseFromString(serialized_example)

        label = example.features.feature["label"].int64_list.value[0]
        img_width =  example.features.feature["width"].int64_list.value[0]
        img_height =  example.features.feature["height"].int64_list.value[0]
        img_shape =  example.features.feature["shape"].int64_list.value

        image_raw = example.features.feature["image_raw"].bytes_list.value[0]
        image = np.frombuffer(image_raw,dtype=np.uint8).reshape(-1 if flatten else img_shape)
        image = np.true_divide(image,255.0,dtype=np.float32)

        images.append(image)
        labels.append(label)

    return images,labels

if __name__ == "__main__":
    tfrecord_path = "./results/imgs.tfrecords"
    images,labels = read_tf_records(tfrecord_path)
    print(images[0])






