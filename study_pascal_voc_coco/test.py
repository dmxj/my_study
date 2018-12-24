# -*- coding: utf-8 -* -
import json

annotations_file_path = "/Users/rensike/Files/data/coco/coco_annotations_minival/instances_minival2014.json"

annotations_json = json.load(open(annotations_file_path,"r"))

print(annotations_json.keys())

print(len(annotations_json["categories"]))

json.dump(annotations_json["categories"],open("./annotations_categories.json","w"))