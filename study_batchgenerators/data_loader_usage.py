# -*- coding: utf-8 -* -
"""
使用batchgenerators进行数据分批次加载
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from skimage import data
from batchgenerators.dataloading.data_loader import SlimDataLoaderBase


class DataLoader(SlimDataLoaderBase):
    def __init__(self, data, BATCH_SIZE=2, num_batches=None):
        super(DataLoader, self).__init__(data, BATCH_SIZE, num_batches)
        # data is now stored in self._data.

    def generate_train_batch(self):
        # usually you would now select random instances of your data. We only have one therefore we skip this
        img = self._data

        # The camera image has only one channel. Our batch layout must be (b, c, x, y). Let's fix that
        img = np.tile(img[None, None], (self.batch_size, 1, 1, 1))

        # now construct the dictionary and return it. np.float32 cast because most networks take float
        return {'data': img.astype(np.float32), 'some_other_key': 'some other value'}


def plot_batch(batch):
    batch_size = batch["data"].shape[0]
    plt.figure(figsize=(16, 10))
    for i in range(batch_size):
        plt.subplot(1, batch_size, i + 1)
        plt.imshow(batch["data"][i, 0], cmap="gray")
    plt.show()


if __name__ == "__main__":
    batchgen = DataLoader(data.camera(), 4)
    batch = batchgen.generate_train_batch()
    plot_batch(batch)
