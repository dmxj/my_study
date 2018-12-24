# -*- coding: utf-8 -* -
'''
VOC格式数据转换为COCO格式
'''
from dataset_convert import pascal_voc_xml2json

voc_xml_path = "/Users/rensike/Files/temp/voc_mini/Annotations"
output_coco_json_path = "/Users/rensike/Files/temp/voc_mini/instance.json"

pascal_voc_xml2json.run_convert(voc_xml_path,output_coco_json_path)

