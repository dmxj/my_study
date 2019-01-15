"""
使用VIS进行图片显示
"""
from matplotlib import pyplot as plt
from gluoncv.utils import viz
from PIL import Image
import numpy as np

image = np.array(Image.open("./images/biking.jpg"))
print(image.shape)
bboxes = np.array([[100,100,300,400]])
cids = np.array([1])
classes = ["person","cat"]

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1)
ax = viz.plot_bbox(image, bboxes, labels=cids, class_names=classes, ax=ax)
plt.show()
