# -*- coding: utf-8 -* -
"""
使用batchgenerators进行数据转换
"""
from batchgenerators.transforms.color_transforms import ContrastAugmentationTransform
from batchgenerators.transforms.spatial_transforms import MirrorTransform
from batchgenerators.transforms.spatial_transforms import SpatialTransform
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.transforms.abstract_transforms import RndTransform
from batchgenerators.transforms.crop_and_pad_transforms import RandomCropTransform
from batchgenerators.transforms.crop_and_pad_transforms import CenterCropSegTransform
from batchgenerators.transforms.crop_and_pad_transforms import CenterCropTransform
from batchgenerators.transforms.noise_transforms import RicianNoiseTransform
from batchgenerators.transforms.resample_transforms import ResampleTransform
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
import data_loader_usage as my_data_loader
from skimage.data import camera
import numpy as np


def combine_transform():
    '''
    组合变换：对比度+镜像
    :return:
    '''
    my_transforms = []
    # 对比度变换
    brightness_transform = ContrastAugmentationTransform((0.3, 3.), preserve_range=True)
    my_transforms.append(brightness_transform)

    # 镜像变换
    mirror_transform = MirrorTransform(axes=(2, 3))
    my_transforms.append(mirror_transform)

    all_transform = Compose(my_transforms)

    batchgen = my_data_loader.DataLoader(camera(), 4)
    multithreaded_generator = MultiThreadedAugmenter(batchgen, all_transform, 4, 2)

    # 显示转换效果
    my_data_loader.plot_batch(multithreaded_generator.__next__())


def spatial_transforms():
    '''
    空间变换：变形、缩放、旋转
    :return:
    '''
    # 变形+旋转+缩放
    spatial_transform = SpatialTransform(camera().shape, np.array(camera().shape) // 2,
                                         do_elastic_deform=True, alpha=(0., 1500.), sigma=(30., 50.),
                                         do_rotation=True, angle_z=(0, 2 * np.pi),
                                         do_scale=True, scale=(0.3, 3.),
                                         border_mode_data='constant', border_cval_data=0, order_data=1,
                                         random_crop=False)
    my_transforms = []
    my_transforms.append(spatial_transform)
    all_transforms = Compose(my_transforms)
    batchgen = my_data_loader.DataLoader(camera(), 4)
    multithreaded_generator = MultiThreadedAugmenter(batchgen, all_transforms, 4, 2)

    # 显示转换效果
    my_data_loader.plot_batch(multithreaded_generator.__next__())


def random_transform():
    '''
    随机地对某些批次的数据进行变换
    :return:
    '''
    spatial_transform = SpatialTransform(camera().shape, np.array(camera().shape) // 2,
                                         do_elastic_deform=True, alpha=(0., 1500.), sigma=(30., 50.),
                                         do_rotation=True, angle_z=(0, 2 * np.pi),
                                         do_scale=True, scale=(0.3, 3.),
                                         border_mode_data='constant', border_cval_data=0, order_data=1,
                                         random_crop=False)

    sometimes_spatial_transform = RndTransform(spatial_transform, prob=0.5)
    batchgen = my_data_loader.DataLoader(camera(), 4)
    multithreaded_generator = MultiThreadedAugmenter(batchgen, Compose([sometimes_spatial_transform]), 4, 2)
    for _ in range(4):
        my_data_loader.plot_batch(multithreaded_generator.__next__())


def crop_transform_random():
    '''
    从图片的任意位置剪切出一个固定的尺寸
    :return:
    '''
    crop_size = (128, 128)
    batchgen = my_data_loader.DataLoader(camera(), 4)

    # 随机地从图片上剪切除（128，128）尺寸的图片块
    randomCrop = RandomCropTransform(crop_size=crop_size)
    spatial_transform = SpatialTransform(crop_size, np.array(crop_size) // 2,
                                         do_elastic_deform=True, alpha=(0., 1500.), sigma=(30., 50.),
                                         do_rotation=True, angle_z=(0, 2 * np.pi),
                                         do_scale=True, scale=(0.5, 2),
                                         border_mode_data='constant', border_cval_data=0, order_data=1,
                                         random_crop=False)
    multithreaded_generator = MultiThreadedAugmenter(batchgen, Compose([randomCrop, spatial_transform]), 4, 2)
    my_data_loader.plot_batch(multithreaded_generator.__next__())


def crop_transform_center():
    '''
    从图片的正正中心剪切图片块
    :return:
    '''
    crop_size = (128, 128)
    batchgen = my_data_loader.DataLoader(camera(), 4)
    centerCrop = CenterCropTransform(crop_size=crop_size)
    multithreaded_generator = MultiThreadedAugmenter(batchgen, Compose([centerCrop]), 4, 2)
    my_data_loader.plot_batch(multithreaded_generator.__next__())


def crop_transform_center_seg():
    '''
    从图片的正正中心剪切分割
    :return:
    '''
    crop_size = (128, 128)
    batchgen = my_data_loader.DataLoader(camera(), 4)
    centerCropSeg = CenterCropSegTransform(output_size=crop_size)
    multithreaded_generator = MultiThreadedAugmenter(batchgen, Compose([centerCropSeg]), 4, 2)
    my_data_loader.plot_batch(multithreaded_generator.__next__())


def noise_transform():
    '''
    为图片加噪声点
    :return:
    '''
    batchgen = my_data_loader.DataLoader(camera(), 4)
    noise_transform = RicianNoiseTransform(noise_variance=(0, 200))
    multithreaded_generator = MultiThreadedAugmenter(batchgen, noise_transform, 4, 2)
    my_data_loader.plot_batch(multithreaded_generator.__next__())


def resampling_transform():
    '''
    对图片进行重采样
    :return:
    '''
    batchgen = my_data_loader.DataLoader(camera(), 4)
    resample_transform = ResampleTransform(zoom_range=(0.05, 0.2))
    multithreaded_generator = MultiThreadedAugmenter(batchgen, resample_transform, 4, 2)
    my_data_loader.plot_batch(multithreaded_generator.__next__())


if __name__ == "__main__":
    # spatial_transforms()
    # crop_transform_center()
    # crop_transform_center_seg()
    # noise_transform()
    resampling_transform()
