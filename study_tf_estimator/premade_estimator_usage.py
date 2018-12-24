# -*- coding: utf-8 -* -
'''
使用预创建的Estimator进行模型的训练、评估、预测
'''
import tensorflow as tf
import iris_data
import time

(train_x, train_y), (test_x, test_y) = iris_data.load_data()

BATCH_SIZE = 10
TRAIN_STEPS = 100


def train():
    '''
    使用预置的全连接网络训练鸢尾花分类模型
    :return:
    '''
    # Feature columns describe how to use the input.
    my_feature_columns = []
    for key in train_x.keys():
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))

    # Build 2 hidden layer DNN with 10, 10 units respectively.
    classifier = tf.estimator.DNNClassifier(
        feature_columns=my_feature_columns,
        # Two hidden layers of 10 nodes each.
        hidden_units=[10, 10],
        # The model must choose between 3 classes.
        n_classes=3
    )

    # Train the model.
    classifier.train(
        input_fn=lambda: iris_data.train_input_fn(train_x, train_y, BATCH_SIZE),
        steps=TRAIN_STEPS
    )

    return classifier


def eval(estimator):
    '''
    训练完模型后，进行模型评估
    :param estimator:
    :return:
    '''
    eval_result = estimator.evaluate(
        input_fn=lambda: iris_data.eval_input_fn(test_x, test_y, BATCH_SIZE)
    )
    print("\nTest set accuracy: {accuracy:0.3f}\n".format(**eval_result))


def predict(estimator):
    '''
    构造假的测试数据，进行模型预测
    :param estimator:
    :return:
    '''
    # Generate predictions from the model
    expected = ['Setosa', 'Versicolor', 'Virginica']
    predict_x = {
        'SepalLength': [5.1, 5.9, 6.9],
        'SepalWidth': [3.3, 3.0, 3.1],
        'PetalLength': [1.7, 4.2, 5.4],
        'PetalWidth': [0.5, 1.5, 2.1],
    }

    predictions = estimator.predict(
        input_fn=lambda: iris_data.eval_input_fn(predict_x, labels=None, batch_size=len(predict_x))
    )
    template = ('\nPrediction is "{}" ({:.1f}%), expected "{}"')

    for pred_dict, expec in zip(predictions, expected):
        class_id = pred_dict['class_ids'][0]
        probability = pred_dict["probabilities"][class_id]
        print(template.format(iris_data.SPECIES[class_id], 100 * probability, expec))


if __name__ == "__main__":
    print("开始训练模型",time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))
    classifier = train()
    print("训练模型结束，开始模型评估",time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))
    eval(classifier)
    print("评估模型结束，开始模型预测", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))
    predict(classifier)

