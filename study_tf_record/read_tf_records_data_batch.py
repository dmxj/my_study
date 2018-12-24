# -*- coding: utf-8 -* -
'''
使用TensorFlow Dataset的API读取tfrecords，并batch化
'''
import tensorflow as tf
import read_tf_records
import numpy as np

# 获取训练输入函数
def get_training_inputs(batch_size,tfrecords_file):
    '''
    构建训练时的输入函数（Estimator的用法）。
    :param batch_size:
    :param tfrecords_file:
    :return: 输出的训练数据需要是(data,label)的tuple，其中训练数据data是{"特征名1":"特征值1","特征名2":"特征值2"}的形式，训练数据label可以直接返回
    '''
    assert tf.gfile.Exists(tfrecords_file),"tf record file does not exist!"
    dataset_iterator_initializer_hook = DatasetIteratorInitializerRunHook()
    def training_input_fn():
        file_name = tf.placeholder(tf.string,shape=[None],name="file_name")
        iterator = get_batch_iterator(file_name,batch_size)

        # session被创建以后，传入tfrecord的路径，对迭代器进行初始化
        dataset_iterator_initializer_hook.set_iterator_initializer_func(
            lambda sess:sess.run(iterator.initializer,feed_dict={
                file_name:tfrecords_file
            })
        )

        datas,targets = iterator.get_next()

        return {"datas":datas},targets
    return training_input_fn,dataset_iterator_initializer_hook


# 获取评估输入函数
def get_test_inputs(batch_size,tfrecords_file):
    '''
    构建评估时的输入函数。读取测试数据集的tfrecord，在训练完成时只调用一次
    :param batch_size:
    :param tfrecords_file:
    :return:
    '''
    test_images, test_labels = read_tf_records.read_tf_records(tfrecords_file)
    test_images = np.array(test_images)
    test_labels = np.array(test_labels)

    return tf.estimator.inputs.numpy_input_fn(
        x={"datas":test_images},
        y=test_labels,
        num_epochs=1,
        batch_size=batch_size,
        shuffle=False
    )


def get_batch_iterator(tfrecords_file,batch_size):
    '''
    读取tfrecord，并进行相关转换，返回数据集的迭代器
    :param tfrecords_file:
    :param batch_size:
    :return:
    '''
    dataset = tf.data.TFRecordDataset(tfrecords_file)\
                .map(_example_proto_to_features_fn,num_parallel_calls=10)\
                .map(_scale_image_fn,num_parallel_calls=10)\
                .shuffle(buffer_size=100)\
                .repeat(10)\
                .batch(batch_size)\
                .prefetch(1)    # make sure there is always 1 batch ready to be served
    return dataset.make_initializable_iterator()

def _example_proto_to_features_fn(example_proto):
    '''
    解析tf-records中的一个example，并将图片数据reshape成一维，返回图像数据向量和图像label
    :param example_proto:
    :return:
    '''
    features = tf.parse_single_example(example_proto,features={
        "label":tf.FixedLenFeature([],tf.int64),
        "image_raw":tf.FixedLenFeature([],tf.string),
        "width":tf.FixedLenFeature([],tf.int64),
        "height": tf.FixedLenFeature([], tf.int64),
        "shape":tf.FixedLenSequenceFeature([1,3],tf.int64)
    })

    image = tf.decode_raw(features["image_raw"],tf.uint8)
    image_shape = features["shape"]
    image_1d = tf.reshape(image,[image_shape[0]*image_shape[1]*image_shape[2]])
    label = tf.cast(features["label"],tf.int32)

    return image_1d,label

def _scale_image_fn(image,label):
    '''
    将图片像素值归一化到0～1之间
    :param image:
    :param label:
    :return:
    '''
    scaled_image = tf.multiply(tf.cast(image,tf.float32),1.0/255/0)
    return scaled_image,label

class DatasetIteratorInitializerRunHook(tf.train.SessionRunHook):
    '''
    数据集迭代器初始化Hook，当session被创建以后，会执行iterator_initializer_func
    '''
    def __init__(self):
        super(DatasetIteratorInitializerRunHook, self).__init__()
        self.iterator_initializer_func = None

    def set_iterator_initializer_func(self,func):
        self.iterator_initializer_func = func

    def after_create_session(self, session, coord):
        '''
        保证session被创建后执行
        :param session:
        :param coord:
        :return:
        '''
        self.iterator_initializer_func(session)

if __name__ == '__main__':
    pass





