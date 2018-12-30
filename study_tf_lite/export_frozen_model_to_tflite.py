# -*- coding: utf-8 -* -
'''
将frozen模型转换为tflite model
'''
import tensorflow as tf

graph_def_file = "/Users/rensike/Resources/models/tensorflow/mobilenet_v1_1.0_224/frozen_graph.pb"
input_arrays = ["input"]
output_arrays = ['MobilenetV1/Predictions/Softmax']

converter = tf.contrib.lite.TocoConverter.from_frozen_graph(
  graph_def_file, input_arrays, output_arrays)
tflite_model = converter.convert()
open("/Users/rensike/Resources/models/tensorflow/mobilenet_v1_1.0_224/converted_model.tflite", "wb").write(tflite_model)




