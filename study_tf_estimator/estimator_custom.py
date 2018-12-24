# -*- coding: utf-8 -* -
'''
自定义Estimator，在预置estimator（premade_estimator_usage.py）的基础上更改。可以对比premade_estimator_usage.py
'''

import tensorflow as tf
import iris_data
import time

(train_x, train_y), (test_x, test_y) = iris_data.load_data()

BATCH_SIZE = 10
TRAIN_STEPS = 100

def my_model(features, labels, mode, params):
    '''
    自定义的Estimator模型
    :param features:
    :param labels:
    :param mode:
    :param params:
    :return:
    '''
    """DNN with three hidden layers and learning_rate=0.1."""
    # Create three fully connected layers.
    net = tf.feature_column.input_layer(features, params['feature_columns'])
    for units in params['hidden_units']:
        net = tf.layers.dense(net, units=units, activation=tf.nn.relu)

    # Compute logits (1 per class).
    logits = tf.layers.dense(net, params['n_classes'], activation=None)

    # Compute predictions.
    predicted_classes = tf.argmax(logits, 1)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_classes[:, tf.newaxis],
            'probabilities': tf.nn.softmax(logits),
            'logits': logits,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Compute loss.
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Compute evaluation metrics.
    accuracy = tf.metrics.accuracy(labels=labels,
                                   predictions=predicted_classes,
                                   name='acc_op')
    metrics = {'accuracy': accuracy}
    tf.summary.scalar('accuracy', accuracy[1])

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)

    # Create training op.
    assert mode == tf.estimator.ModeKeys.TRAIN

    optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

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
    classifier = tf.estimator.Estimator(
        model_fn=my_model,
        params={
              "feature_columns" : my_feature_columns,
              "hidden_units" : [10, 10],
              "n_classes" : 3
        }
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