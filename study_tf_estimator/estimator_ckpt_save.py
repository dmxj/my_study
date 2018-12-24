# -*- coding: utf-8 -* -
'''
训练时保存模型检查点
'''
import tensorflow as tf
import iris_data
import time

(train_x, train_y), (test_x, test_y) = iris_data.load_data()

BATCH_SIZE = 10
TRAIN_STEPS = 10000

'''
默认情况下：
1、Estimator 会将检查点文件写入由 Python 的 tempfile.mkdtemp 函数选择的临时目录中；
2、每 10 分钟（600 秒）写入一个检查点；
3、在 train 方法开始（第一次迭代）和完成（最后一次迭代）时写入一个检查点；
4、只在目录中保留 5 个最近写入的检查点
'''

def train():
    '''
    使用预置的全连接网络训练鸢尾花分类模型
    :return:
    '''
    # Feature columns describe how to use the input.
    my_feature_columns = []
    for key in train_x.keys():
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))

    # Define run config for estimator save model checkpoints on disk.
    # save_checkpoints_steps和save_checkpoints_secs不能同时设置
    my_checkpointing_config = tf.estimator.RunConfig(
        model_dir="./model_dir",     # 指定检查点保存目录
        save_checkpoints_steps=200, # 每200步保存一次检查点
        #save_checkpoints_secs=30,  # 每30秒保存一次检查点
        keep_checkpoint_max=3, # 最多保留最新的3个检查点文件
    )

    # Build 2 hidden layer DNN with 10, 10 units respectively.
    classifier = tf.estimator.DNNClassifier(
        feature_columns=my_feature_columns,
        # Two hidden layers of 10 nodes each.
        hidden_units=[10, 10],
        # The model must choose between 3 classes.
        n_classes=3,
        config=my_checkpointing_config
    )

    # Train the model.
    classifier.train(
        input_fn=lambda: iris_data.train_input_fn(train_x, train_y, BATCH_SIZE),
        steps=TRAIN_STEPS
    )

    return classifier


if __name__ == "__main__":
    print("开始训练模型", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))
    classifier = train()
    print("训练模型结束", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))

