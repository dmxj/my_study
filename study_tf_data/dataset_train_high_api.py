# -*- coding: utf-8 -* -
'''
训练工作流程。使用高阶API。
'''
import tensorflow as tf
import numpy as np

'''
tf.train.MonitoredTrainingSession API 简化了在分布式设置下运行 TensorFlow 的很多方面。
MonitoredTrainingSession 使用 tf.errors.OutOfRangeError 表示训练已完成，
因此要将其与 tf.data API 结合使用，我们建议使用 Dataset.make_one_shot_iterator()。
'''


def model_func(data, label):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(32, activation=tf.nn.sigmoid),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    logits = model(data)
    return tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=logits)


num_epochs = 10
learning_rate = 0.0001
data = tf.data.Dataset.from_tensor_slices(np.arange(0, 2000, dtype=np.float32).reshape((1000, 2)))
labels = tf.data.Dataset.range(10).repeat(100)
dataset = tf.data.Dataset.zip((data, labels))
dataset = dataset.shuffle(buffer_size=1000)
dataset = dataset.batch(16)
dataset.repeat(num_epochs)
iterator = dataset.make_one_shot_iterator()

next_data, next_label = iterator.get_next()
loss = model_func(next_data, next_label)

training_op = tf.train.AdagradOptimizer(learning_rate).minimize(loss)

with tf.train.MonitoredTrainingSession() as sess:
    while not sess.should_stop():
        sess.run(training_op)

'''
要在 input_fn 中使用 Dataset（input_fn 属于 tf.estimator.Estimator），我们还建议使用 Dataset.make_one_shot_iterator()。例如：
'''


def dataset_input_fn():
    filenames = ["/var/data/file1.tfrecord", "/var/data/file2.tfrecord"]
    dataset = tf.data.TFRecordDataset(filenames)

    # Use `tf.parse_single_example()` to extract data from a `tf.Example`
    # protocol buffer, and perform any additional per-record preprocessing.
    def parser(record):
        keys_to_features = {
            "image_data": tf.FixedLenFeature((), tf.string, default_value=""),
            "date_time": tf.FixedLenFeature((), tf.int64, default_value=""),
            "label": tf.FixedLenFeature((), tf.int64,
                                        default_value=tf.zeros([], dtype=tf.int64)),
        }
        parsed = tf.parse_single_example(record, keys_to_features)

        # Perform additional preprocessing on the parsed data.
        image = tf.image.decode_jpeg(parsed["image_data"])
        image = tf.reshape(image, [299, 299, 1])
        label = tf.cast(parsed["label"], tf.int32)

        return {"image_data": image, "date_time": parsed["date_time"]}, label

    # Use `Dataset.map()` to build a pair of a feature dictionary and a label
    # tensor for each example.
    dataset = dataset.map(parser)
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(32)
    dataset = dataset.repeat(num_epochs)
    iterator = dataset.make_one_shot_iterator()

    # `features` is a dictionary in which each value is a batch of values for
    # that feature; `labels` is a batch of labels.
    features, labels = iterator.get_next()
    return features, labels
