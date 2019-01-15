# -*- coding: utf-8 -* -
'''
使用faster rcnn尝试模型的batch inference
'''
from maskrcnn_benchmark.config import cfg
from predictor import Predictor
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

config_file = "./configs/e2e_faster_rcnn_R_50_FPN_1x_caffe2.yaml"

# update the config options with the config file
cfg.merge_from_file(config_file)
# manual override some options
cfg.merge_from_list(["MODEL.DEVICE", "cpu"])

p = Predictor(
    cfg,
    min_image_size=800,
    confidence_threshold=0.7,
)
# load image list and then run prediction
image_file_list = ["./images/img1.jpg","./images/img2.jpg","./images/img3.jpg","./images/img4.jpg","./images/img5.jpg"]
images = [np.array(Image.open(img_file)) for img_file in image_file_list]

predictions = p.run_on_batch_image(images)

plt.figure(num='astronaut',figsize=(8,8))  #创建一个名为astronaut的窗口,并设置大小

for i,img in enumerate(predictions):
    plt.subplot(2,3,i+1)     #将窗口分为两行两列四个子图，则可显示四幅图片
    plt.title(os.path.basename(image_file_list[i]))   #第一幅图片标题
    plt.imshow(img)      #绘制第一幅图片
    plt.axis('off')  # 不显示坐标尺寸

plt.show()

