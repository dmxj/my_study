# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""Convert raw PASCAL dataset to TFRecord for object_detection.

Example usage:
    python object_detection/dataset_tools/create_pascal_tf_record.py \
        --data_dir=/home/user/VOCdevkit \
        --year=VOC2012 \
        --output_path=/home/user/pascal.record
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import io
import logging
import os

from lxml import etree
import PIL.Image
import tensorflow as tf
import numpy as np

from utils import dataset_util
from utils import label_map_util

flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'Root directory to raw PASCAL VOC dataset.')
flags.DEFINE_string('set', 'train', 'Convert training set, validation set or '
                                    'merged set.')
flags.DEFINE_string('annotations_dir', 'Annotations',
                    '(Relative) path to annotations directory.')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_string('label_map_path', 'data/pascal_label_map.pbtxt',
                    'Path to label map proto')
flags.DEFINE_boolean('ignore_difficult_instances', False, 'Whether to ignore '
                                                          'difficult instances')
flags.DEFINE_boolean('include_segment_class', False, 'Whether to include '
                                                     'class segment')
flags.DEFINE_boolean('include_segment_object', False, 'Whether to include '
                                                      'object segment')
FLAGS = flags.FLAGS

SETS = ['train', 'val', 'trainval', 'test']


def dict_to_tf_example(data,
                       dataset_directory,
                       label_map_dict,
                       ignore_difficult_instances=False,
                       image_subdirectory='JPEGImages',
                       mask_filename=None,
                       segment_class_subdirectory='SegmentationClass',
                       segment_object_subdirectory='SegmentationObject',
                       include_segment_class=False,
                       include_segment_object=False):
    """Convert XML derived dict to tf.Example proto.

    Notice that this function normalizes the bounding box coordinates provided
    by the raw data.

    Args:
      data: dict holding PASCAL XML fields for a single image (obtained by
        running dataset_util.recursive_parse_xml_to_dict)
      dataset_directory: Path to root directory holding PASCAL dataset
      label_map_dict: A map from string label names to integers ids.
      ignore_difficult_instances: Whether to skip difficult instances in the
        dataset  (default: False).
      image_subdirectory: String specifying subdirectory within the
        PASCAL dataset directory holding the actual image data.
      mask_filename: String sengment image filename.
      segment_class_subdirectory: String specifying subdirectory within the
        PASCAL dataset directory holding the actual class segment image data.
      segment_object_subdirectory: String specifying subdirectory within the
        PASCAL dataset directory holding the actual object segment image data.
      include_segment_class: is contains class segment data.
      include_segment_object: is contains object segment data.

    Returns:
      example: The converted tf.Example.

    Raises:
      ValueError: if the image pointed to by data['filename'] is not a valid JPEG
    """
    # img_path = os.path.join(data['folder'], image_subdirectory, data['filename'])
    img_path = os.path.join(image_subdirectory, data['filename'])
    full_path = os.path.join(dataset_directory, img_path)
    with tf.gfile.GFile(full_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    if image.format != 'JPEG':
        raise ValueError('Image format not JPEG')
    key = hashlib.sha256(encoded_jpg).hexdigest()

    mask_path = None
    nonzero_x_indices = None
    nonzero_y_indices = None
    mask_np = None
    if include_segment_class:
        mask_path = os.path.join(dataset_directory, segment_class_subdirectory, mask_filename)

    if include_segment_object:
        mask_path = os.path.join(dataset_directory, segment_object_subdirectory, mask_filename)

    if mask_path is not None:
        with tf.gfile.GFile(mask_path, 'rb') as fid:
            encoded_mask_png = fid.read()
        encoded_png_io = io.BytesIO(encoded_mask_png)
        mask = PIL.Image.open(encoded_png_io)
        if mask.format != 'PNG':
            raise ValueError('Mask format not PNG')

        mask_np = np.asarray(mask)
        nonbackground_indices_x = np.any(mask_np != 2, axis=0)
        nonbackground_indices_y = np.any(mask_np != 2, axis=1)
        nonzero_x_indices = np.where(nonbackground_indices_x)
        nonzero_y_indices = np.where(nonbackground_indices_y)

    width = int(data['size']['width'])
    height = int(data['size']['height'])

    xmins = []
    ymins = []
    xmaxs = []
    ymaxs = []
    classes = []
    classes_text = []
    truncated = []
    poses = []
    difficult_obj = []
    masks = []
    if 'object' in data:
        for obj in data['object']:
            difficult = bool(int(obj['difficult']))
            if ignore_difficult_instances and difficult:
                continue
            difficult_obj.append(int(difficult))

            if include_segment_object or include_segment_class:
                xmins.append(float(np.min(nonzero_x_indices)) / width)
                xmaxs.append(float(np.max(nonzero_x_indices)) / width)
                ymins.append(float(np.min(nonzero_y_indices)) / height)
                ymaxs.append(float(np.max(nonzero_y_indices)) / height)

                mask_remapped = (mask_np != 2).astype(np.uint8)
                masks.append(mask_remapped)
            else:
                xmins.append(float(obj['bndbox']['xmin']) / width)
                ymins.append(float(obj['bndbox']['ymin']) / height)
                xmaxs.append(float(obj['bndbox']['xmax']) / width)
                ymaxs.append(float(obj['bndbox']['ymax']) / height)
            classes_text.append(obj['name'].encode('utf8'))
            classes.append(label_map_dict[obj['name']])
            if "truncated" in obj:
                truncated.append(int(obj['truncated']))
            poses.append(obj['pose'].encode('utf8'))

    feature_dict = {
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(
            data['filename'].encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(
            data['filename'].encode('utf8')),
        'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
        'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
        'image/object/truncated': dataset_util.int64_list_feature(truncated),
        'image/object/view': dataset_util.bytes_list_feature(poses),
    }

    if include_segment_class or include_segment_object:
        encoded_mask_png_list = []
        for mask in masks:
            img = PIL.Image.fromarray(mask)
            output = io.BytesIO()
            img.save(output, format='PNG')
            encoded_mask_png_list.append(output.getvalue())
        feature_dict['image/object/mask'] = (
            dataset_util.bytes_list_feature(encoded_mask_png_list))

    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return example


def main(_):
    if FLAGS.set not in SETS:
        raise ValueError('set must be in : {}'.format(SETS))

    data_dir = FLAGS.data_dir

    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)

    label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)

    print('Reading from PASCAL dataset.')
    examples_path = os.path.join(data_dir, 'ImageSets', 'Main',
                                 FLAGS.set + '.txt')
    if FLAGS.include_segment_class or FLAGS.include_segment_object:
        examples_path = os.path.join(data_dir, 'ImageSets', 'Segmentation',
                                     FLAGS.set + '.txt')
    annotations_dir = os.path.join(data_dir, FLAGS.annotations_dir)
    examples_list = dataset_util.read_examples_list(examples_path)
    for idx, example in enumerate(examples_list):
        if idx % 100 == 0:
            logging.info('On image %d of %d', idx, len(examples_list))
        path = os.path.join(annotations_dir, example + '.xml')
        mask_filename = None
        if FLAGS.include_segment_class or FLAGS.include_segment_object:
            mask_filename = example + ".png"
        with tf.gfile.GFile(path, 'r') as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']

        tf_example = dict_to_tf_example(data, FLAGS.data_dir, label_map_dict,
                                        FLAGS.ignore_difficult_instances, mask_filename=mask_filename,
                                        include_segment_class=FLAGS.include_segment_class,
                                        include_segment_object=FLAGS.include_segment_object)
        writer.write(tf_example.SerializeToString())

    writer.close()


if __name__ == '__main__':
    tf.app.run()
