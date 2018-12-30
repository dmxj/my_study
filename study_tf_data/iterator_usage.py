# -*- coding: utf-8 -* -
'''
构建数据集迭代器，用于遍历访问数据集中的元素。
目前支持四种迭代器（由易到难）：
1. 单次
2. 可初始化
3. 可重新初始化
4. 可馈送
'''
import tensorflow as tf


def demo_one_shot_iterator():
    '''
    单次迭代器
    注意：目前，单次迭代器是唯一易于与 Estimator 搭配使用的类型！
    :return:
    '''
    dataset = tf.data.Dataset.range(10)
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()

    with tf.Session() as sess:
        for i in range(100):
            value = sess.run(next_element)
            print(value)
            assert i == value


def demo_can_init_iterator():
    '''
    可初始化迭代器
    允许使用一个或多个tf.placeholder()张量参数化数据集的定义
    :return:
    '''
    max_value = tf.placeholder(tf.int64, shape=[])
    dataset = tf.data.Dataset.range(max_value)
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()

    sess = tf.Session()

    # Initialize an iterator over a dataset with 10 elements.
    sess.run(iterator.initializer, feed_dict={max_value: 10})
    for i in range(10):
        value = sess.run(next_element)
        assert i == value

    # Initialize the same iterator over a dataset with 100 elements.
    sess.run(iterator.initializer, feed_dict={max_value: 100})
    for i in range(100):
        value = sess.run(next_element)
        assert i == value

    sess.close()

def demo_can_re_init_iterator():
    '''
    可重新初始化迭代器
    可以通过多个不同的 Dataset 对象进行初始化。
    例如，您可能有一个训练输入管道，它会对输入图片进行随机扰动来改善泛化；
    还有一个验证输入管道，它会评估对未修改数据的预测。这些管道通常会使用不同的 Dataset 对象，这些对象具有相同的结构（即每个组件具有相同类型和兼容形状）。
    :return:
    '''
    # Define training and validation datasets with the same structure.
    training_dataset = tf.data.Dataset.range(100).map(
        lambda x: x + tf.random_uniform([], -10, 10, tf.int64))
    validation_dataset = tf.data.Dataset.range(50)

    # A reinitializable iterator is defined by its structure. We could use the
    # `output_types` and `output_shapes` properties of either `training_dataset`
    # or `validation_dataset` here, because they are compatible.
    iterator = tf.data.Iterator.from_structure(training_dataset.output_types,
                                               training_dataset.output_shapes)
    next_element = iterator.get_next()

    training_init_op = iterator.make_initializer(training_dataset)
    validation_init_op = iterator.make_initializer(validation_dataset)

    sess = tf.Session()

    # Run 20 epochs in which the training dataset is traversed, followed by the
    # validation dataset.
    for _ in range(20):
        # Initialize an iterator over the training dataset.
        sess.run(training_init_op)
        for _ in range(100):
            sess.run(next_element)

        # Initialize an iterator over the validation dataset.
        sess.run(validation_init_op)
        for _ in range(50):
            sess.run(next_element)

    sess.close()


def demo_can_feed_iterator():
    '''
    可馈送迭代器
    可以与 tf.placeholder 一起使用，以选择所使用的 Iterator（在每次调用 tf.Session.run 时）（通过熟悉的 feed_dict 机制）
    :return:
    '''
    # Define training and validation datasets with the same structure.
    training_dataset = tf.data.Dataset.range(100).map(
        lambda x:x + tf.random_uniform([],-10,10,tf.int64)
    ).repeat()
    validate_dataset = tf.data.Dataset.range(50)

    # A feedable iterator is defined by a handle placeholder and its structure. We
    # could use the `output_types` and `output_shapes` properties of either
    # `training_dataset` or `validation_dataset` here, because they have
    # identical structure.
    handle = tf.placeholder(tf.string,shape=[])
    iterator = tf.data.Iterator.from_string_handle(
        handle,training_dataset.output_types,training_dataset.output_shapes
    )
    next_element = iterator.get_next()

    # You can use feedable iterators with a variety of different kinds of iterator
    # (such as one-shot and initializable iterators).
    training_iterator = training_dataset.make_one_shot_iterator()
    validation_iterator = validate_dataset.make_initializable_iterator()

    sess = tf.Session()

    # The `Iterator.string_handle()` method returns a tensor that can be evaluated
    # and used to feed the `handle` placeholder.
    training_handle = sess.run(training_iterator.string_handle())
    validation_handle = sess.run(validation_iterator.string_handle())

    # Loop forever, alternating between training and validation.
    for _ in range(10):
        # Run 200 steps using the training dataset. Note that the training dataset is
        # infinite, and we resume from where we left off in the previous `while` loop
        # iteration.
        for _ in range(200):
            sess.run(next_element,feed_dict={handle:training_handle})

        # Run one pass over the validation dataset.
        sess.run(validation_iterator.initializer)
        for _ in range(50):
            sess.run(next_element,feed_dict={handle:validation_handle})

    sess.close()


if __name__ == "__main__":
    demo_one_shot_iterator()
    # demo_can_init_iterator()
    # demo_can_re_init_iterator()
    # demo_can_feed_iterator()
