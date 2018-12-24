# -*- coding: utf-8 -* -
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
from tensorflow.contrib.slim import nets
import math

slim = tf.contrib.slim

# Load the data
images, labels = input_data.read_data_sets("/Users/rensike/Resources/datasets/mnist")

# Define the network
predictions = nets.vgg.vgg_16(images)

# Choose the metrics to compute:
names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
    'accuracy': slim.metrics.accuracy(predictions, labels),
    'precision': slim.metrics.precision(predictions, labels),
    'recall': slim.metrics.recall(labels, predictions),
})

# Create the summary ops such that they also print out to std output:
summary_ops = []
for metric_name, metric_value in names_to_values.iteritems():
    op = tf.summary.scalar(metric_name, metric_value)
    op = tf.Print(op, [metric_value], metric_name)
    summary_ops.append(op)

num_examples = 10000
batch_size = 32
num_batches = math.ceil(num_examples / float(batch_size))

# Setup the global step.
slim.get_or_create_global_step()

log_dir = "./result/output"  # Where the summaries are stored.
checkpoint_dir = "./result/checkpoint"  # Where the model checkpoint are stored.
eval_interval_secs = 10  # How often to run the evaluation.
slim.evaluation.evaluation_loop(
    'local',
    checkpoint_dir,
    log_dir,
    num_evals=num_batches,
    eval_op=names_to_updates.values(),
    summary_op=tf.summary.merge(summary_ops),
    eval_interval_secs=eval_interval_secs)
