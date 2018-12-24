# -*- coding: utf-8 -* -
"""
使用batchgenerators进行数据增广
"""
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from batchgenerators.augmentations.color_augmentations import augment_contrast
from batchgenerators.augmentations.spatial_transformations import augment_resize,augment_zoom,augment_mirroring

def show_img(data):
    plt.figure(figsize=(16, 10))
    data_size = data.shape[0]
    for i in range(data_size):
        plt.subplot(1,data_size,i+1)
        plt.imshow(data[i])
    plt.show()

def constrast_augment(data):
    '''
    对比度增广
    :return:
    '''
    result = augment_contrast(data)
    return result

def resize_augment(data):
    '''
    尺寸变换增广
    :param data:
    :return:
    '''
    data_result, seg_result = augment_resize(data,(500,500))
    return data_result, seg_result

def zoom_augment(data):
    '''
    zoom变换增广
    :param data:
    :return:
    '''
    data_result, seg_result = augment_zoom(data,(0.5,0.8))
    return data_result, seg_result

def mirror_augment(data):
    '''
    镜像变换增广
    :param data:
    :return:
    '''
    data_result, seg_result = augment_mirroring(data)
    return data_result, seg_result

if __name__ == "__main__":
    im1 = np.expand_dims(cv2.imread("img1.jpg"),0)
    im2 = np.expand_dims(cv2.imread("img2.jpg"),0)
    data = np.concatenate((im1,im2),0)

    data_copy = data.copy()

    # contrast_augment(data_copy)
    # resize_augment(data_copy)
    # zoom_augment(data_copy)
    data_result, seg_result = zoom_augment(data_copy)

    show_img(data_result)


