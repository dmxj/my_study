# -*- coding: utf-8 -* -
"""
deep lab v3+
"""
import numpy as np
from PIL import Image

from utils import label_map_util

from matplotlib import gridspec
from matplotlib import pyplot as plt
import tensorflow as tf


class DeepLabModel(object):
    """Class to load deeplab model and run inference."""

    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
    INPUT_SIZE = 513
    FROZEN_GRAPH_NAME = 'frozen_inference_graph'

    def __init__(self, frozen_model, label_names):
        """Creates and loads pretrained deeplab model."""
        self.graph = tf.Graph()
        self.frozen_model = frozen_model
        self.label_names = label_names

        self.FULL_LABEL_MAP = np.arange(len(self.label_names)).reshape(len(self.label_names), 1)
        self.FULL_COLOR_MAP = label_map_util.label_to_color_image(self.FULL_LABEL_MAP)

        file_handle = open(self.frozen_model, "rb")
        graph_def = tf.GraphDef.FromString(file_handle.read())

        if graph_def is None:
            raise RuntimeError('Cannot find inference graph in tar archive.')

        with self.graph.as_default():
            tf.import_graph_def(graph_def, name='')

        self.sess = tf.Session(graph=self.graph)

    def predict(self, image, is_show=False):
        """Runs inference on a single image.

        Args:
          image: A PIL.Image object, raw input image.

        Returns:
          resized_image: RGB image resized from original input image.
          seg_map: Segmentation map of `resized_image`.
        """
        width, height = image.size
        resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
        target_size = (int(resize_ratio * width), int(resize_ratio * height))
        resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
        batch_seg_map = self.sess.run(
            self.OUTPUT_TENSOR_NAME,
            feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
        seg_map = batch_seg_map[0]

        if is_show:
            self.vis_segmentation(resized_image, seg_map)

        return resized_image, seg_map

    def vis_segmentation(self, image, seg_map):
        """Visualizes input image, segmentation map and overlay view."""
        plt.figure(figsize=(15, 5))
        grid_spec = gridspec.GridSpec(1, 4, width_ratios=[6, 6, 6, 1])

        plt.subplot(grid_spec[0])
        plt.imshow(image)
        plt.axis('off')
        plt.title('input image')

        plt.subplot(grid_spec[1])
        seg_image = label_map_util.label_to_color_image(seg_map).astype(np.uint8)
        plt.imshow(seg_image)
        plt.axis('off')
        plt.title('segmentation map')

        plt.subplot(grid_spec[2])
        plt.imshow(image)
        plt.imshow(seg_image, alpha=0.7)
        plt.axis('off')
        plt.title('segmentation overlay')

        unique_labels = np.unique(seg_map)
        ax = plt.subplot(grid_spec[3])
        plt.imshow(
            self.FULL_COLOR_MAP[unique_labels].astype(np.uint8), interpolation='nearest')
        ax.yaxis.tick_right()
        plt.yticks(range(len(unique_labels)), self.label_names[unique_labels])
        plt.xticks([], [])
        ax.tick_params(width=0.0)
        plt.grid('off')
        plt.show()

if __name__ == '__main__':
    LABEL_NAMES = np.asarray([
        'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
        'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
        'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv'
    ])
    FROZEN_GRAPH_MODEL = "/Users/rensike/Resources/models/tensorflow/deeplabv3_pascal_trainval/frozen_inference_graph.pb"

    model = DeepLabModel(FROZEN_GRAPH_MODEL,LABEL_NAMES)

    test_image = "./images/img1.jpg"
    model.predict(Image.open(test_image),is_show=True)

