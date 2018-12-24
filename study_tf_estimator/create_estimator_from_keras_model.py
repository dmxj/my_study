# -*- coding: utf-8 -* -
'''
将现有的Keras模型转换为Estomator。这样做之后，Keras 模型就可以利用 Estimator 的优势，例如分布式训练。
'''
import tensorflow as tf
import numpy as np


train_data = np.random.rand((10,299, 299, 3))
train_labels = np.arange(10).reshape([10])

# Instantiate a Keras inception v3 model.
keras_inception_v3 = tf.keras.applications.inception_v3.InceptionV3(weights=None)

# Compile model with the optimizer, loss, and metrics you'd like to train with.
keras_inception_v3.compile(optimizer=tf.keras.optimizers.SGD(lr=0.0001,momentum=0.9),
                           loss='categorical_crossentropy',
                           metrics='accuracy')

# Create an Estimator from the compiled Keras model. Note the initial model
# state of the keras model is preserved in the created Estimator.
est_inception_v3 = tf.keras.estimator.model_to_estimator(keras_model=keras_inception_v3)

# Treat the derived Estimator as you would with any other Estimator.
# First, recover the input name(s) of Keras model, so we can use them as the
# feature column name(s) of the Estimator input function:
print(keras_inception_v3.input_names) # => ['input_1']

# Once we have the input name(s), we can create the input function, for example,
# for input(s) in the format of numpy ndarray:
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"input_1":train_data},
    y=train_labels,
    num_epochs=1,
    shuffle=False
)

# To train, we call Estimator's train function:
est_inception_v3.train(input_fn=train_input_fn,steps=2000)


