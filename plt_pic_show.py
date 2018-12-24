import os
from PIL import Image
import matplotlib.pyplot as plt

img = Image.open("004.jpg")

plt.figure("Image") # 图像窗口名称
plt.imshow(img)
plt.axis('off') # 关掉坐标轴为 off
plt.title('image') # 图像题目
plt.show()