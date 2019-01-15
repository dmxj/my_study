# encoding:utf-8
from gluoncv import model_zoo
from gluoncv import data as gdata
net = model_zoo.get_model('gan', pretrained=True)

# val_dataset = gdata.VOCDetection(root="/Users/rensike/Files/temp/voc_mini",
#             splits=[(0, 'val')])
#
# print(len(val_dataset))
# print(val_dataset.classes)
# print(val_dataset.CLASSES)
# print(val_dataset.index_map)

