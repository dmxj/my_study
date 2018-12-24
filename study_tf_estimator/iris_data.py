# -*- coding: utf-8 -* -
'''
从data/ 目录下加载训练集和测试集
'''
import pandas as pd
import tensorflow as tf

# CSV列名
CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth','PetalLength', 'PetalWidth', 'Species']
# 鸢尾花种类
SPECIES = ['Setosa', 'Versicolor', 'Virginica']

# 训练集文件路径
TRAIN_PATH = "./data/iris_training.csv"
# 测试集文件路径
TEST_PATH = "./data/iris_test.csv"

def load_data(y_name="Species"):
    '''
    解析训练集、测试集文件，返回训练数据、测试数据
    :param y_name:
    :return:
    '''
    train = pd.read_csv(TRAIN_PATH,names=CSV_COLUMN_NAMES,header=0)
    train_x,train_y= train,train.pop(y_name)

    test = pd.read_csv(TEST_PATH,names=CSV_COLUMN_NAMES,header=0)
    test_x,test_y = test,test.pop(y_name)

    return (train_x,train_y),(test_x,test_y)

def train_input_fn(features,labels,batch_size):
    '''
    用于Estimator训练的输入函数
    :param features:
    :param labels:
    :param batch_size:
    :return:
    '''
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features),labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    return dataset


def eval_input_fn(features,labels,batch_size):
    '''
    用于Estimator评估或预测的输入函数
    :param features:
    :param labels:
    :param batch_size:
    :return:
    '''
    features = dict(features)
    if labels is None:  # labels为空时，进行模型预测
        inputs = features
    else:   # labels不为空时，进行模型评估
        inputs = (features,labels)

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None."
    dataset = dataset.batch(batch_size)

    return dataset

