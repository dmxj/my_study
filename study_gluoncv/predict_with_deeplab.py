#encoding:utf-8
"""
使用deeplab预训练模型进行预测
参考：https://gluon-cv.mxnet.io/build/examples_segmentation/demo_deeplab.html
"""
import mxnet as mx
from mxnet import image
from matplotlib import pyplot as plt
from mxnet.gluon.data.vision import transforms
import gluoncv
import os
# using cpu
ctx = mx.cpu(0)

'''
url = 'https://github.com/zhanghang1989/image-data/blob/master/encoding/' + \
    'segmentation/ade20k/ADE_val_00001755.jpg?raw=true'
filename = 'ade20k_example.jpg'
gluoncv.utils.download(url, filename, True)
'''

# 加载图像
img_path = "./images"
im_fname = os.path.join(img_path,"img2.jpg")
img = image.imread(im_fname)
plt.imshow(img.asnumpy())
plt.show()

# 转换图像
transform_fn = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([.485, .456, .406], [.229, .224, .225])
])
img = transform_fn(img)
img = img.expand_dims(0).as_in_context(ctx)

# 加载预训练模型
model = gluoncv.model_zoo.get_model('deeplab_resnet101_ade', pretrained=True)
output = model.demo(img)
predict = mx.nd.squeeze(mx.nd.argmax(output, 1)).asnumpy()

# 添加调色板
from gluoncv.utils.viz import get_color_pallete
import matplotlib.image as mpimg
mask = get_color_pallete(predict, 'ade20k')
mask.save('deeplab_output.png')

##############################################################################
# 显示预测的结果
mmask = mpimg.imread('deeplab_output.png')
plt.imshow(mmask)
plt.show()





