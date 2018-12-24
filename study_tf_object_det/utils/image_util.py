# -*- coding: utf-8 -* -
import numpy as np
from PIL import Image


def load_image_into_numpy_array(image):
    '''
    将图片加载为numpy数组
    :param image: PIL图片
    :return:
    '''
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


def load_numpy_array_into_image(np_arr):
    '''
    将numpy array转换为图片
    :param np_arr:
    :return:
    '''
    im = Image.fromarray(np_arr.astype('uint8')).convert('RGB')
    return im
