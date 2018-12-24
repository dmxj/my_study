from matplotlib import pyplot as plt
from gluoncv import model_zoo, data, utils
import os

# 加载mask rcnn预训练模型(默认保存到：~/.mxnet/models)
net = model_zoo.get_model('mask_rcnn_resnet50_v1b_coco', pretrained=True)

# 准备图片，并预处理
# im_fname = utils.download('https://github.com/dmlc/web-data/blob/master/gluoncv/detection/biking.jpg?raw=true',path='biking.jpg')
img_path = "./images"
im_fname = os.path.join(img_path,"img1.jpg")
x, orig_img = data.transforms.presets.rcnn.load_test(im_fname)

ids, scores, bboxes, masks = [xx[0].asnumpy() for xx in net(x)]

# paint segmentation mask on images directly
width, height = orig_img.shape[1], orig_img.shape[0]
masks = utils.viz.expand_mask(masks, bboxes, (width, height), scores)
orig_img = utils.viz.plot_mask(orig_img, masks)

# identical to Faster RCNN object detection
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1)
ax = utils.viz.plot_bbox(orig_img, bboxes, scores, ids,
                         class_names=net.classes, ax=ax)
plt.show()




