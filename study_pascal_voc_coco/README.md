## Pascal Voc数据集说明
dataset/voc文件夹下，即为Pascal voc的数据目录格式的示例，从原始的voc2012中摘取出部分数据。
+ Annotations:  
标注文件夹，里面每一个xml文件对应JPEGImages文件夹中的一个图片,xml中标注了bounding box的位置以及对应的分类（总共20类）。  
检测框的标注为矩形框的左上角坐标和右下角坐标：xmin、ymin、xmax、ymax，以左上角为原点，且均为整数。
+ ImageSets:  
数据列表，包含两个文件夹：Main和Segmentation，Main文件夹中的数据列表用于目标检测任务，Segmentation文件夹中的数据列表用于语义分割或者实例分割任务，两个文件夹中的数据列表不重叠。  
两个文件夹中各自分别有train.txt、val.txt、trainval.txt，文件中的每一行都是一个文件名（不包含文件后缀），train.txt包含训练用的数据，val.txt包含验证用的数据，trainval.txt包含两者的数据的总和。
+ JPEGImages:  
原始图片文件数据，包括检测和分割任务使用到的所有的图片。文件夹中的每一个图像文件对应Annotations文件夹中的一个xml标注文件。
+ SegmentationClass:  
用于语义分割任务的图片，在JPEGImages能找到原图。
+ SegmentationObject:  
用于实例分割任务的图片，和SegmentationClass的原图片完全对应。

### 通过脚本将Pasca Voc数据转换成tf record格式
参考：https://github.com/tensorflow/models/blob/master/research/object_detection/dataset_tools/create_pascal_tf_record.py  
脚本：sh create_pascal_tf_record.sh

## Coco数据集说明
dataset/coco文件夹下，即为Coco数据目录格式的实例，从coco2014的数据集中摘取出的部分数据。
*注：coco中的标注分为三种：instances_xxx.json(目标实例数据)；person_keypoints_xxx.json(人体关键点数据)；captions_xxx.json(image caption看图说话的数据)*  
+ annotations:  
标注文件夹，里面包含instances_test.json、instances_train.json、instances_val.json三个标注文件，对应test、train、val三个文件夹中的原始图片。
+ test:  
测试数据集原始图片。
+ train:  
训练数据集原始图片。
+ val:  
验证数据集原始图片。

### 通过脚本将Coco数据转换成tf record格式
参考：https://github.com/tensorflow/models/blob/master/research/object_detection/dataset_tools/create_coco_tf_record.py
脚本：sh create_coco_tf_record.sh





