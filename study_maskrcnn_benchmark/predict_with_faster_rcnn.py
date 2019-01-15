# -*- coding: utf-8 -* -
"""
使用faster rcnn进行模型预测
"""
from maskrcnn_benchmark.config import cfg
from predictor import Predictor
from PIL import Image
import numpy as np
import cv2

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
# load image and then run prediction
image = np.array(Image.open("./images/img2.jpg"))
predictions = p.run_on_opencv_image(image)

print("prediction labels shape:",p.predictions.get_field("labels").shape)
print("prediction labels:",p.predictions.get_field("labels"))
print("prediction bbox shape:",p.predictions.bbox.shape)
print("prediction bbox:",p.predictions.bbox)
scores = p.predictions.get_field("scores")
print("prediction scores shape:",scores.shape)
print("prediction scores:",scores)

# p.top_predictions.get_field("labels")
# p.top_predictions.bbox
# p.top_predictions.get_field("scores")

cv2.imwrite("./results/img2_det.jpg",predictions[:,:,[2,1,0]])
cv2.imshow("predicition result",predictions[:,:,[2,1,0]])

cv2.waitKey(0)
cv2.destroyAllWindows()

