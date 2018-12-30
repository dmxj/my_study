# -*- coding: utf-8 -* -
import glob
import os
import shutil
from PIL import Image,ImageDraw
import json
import cv2
import numpy as np

# img_list = glob.glob("images/*.jpg")
# for img in img_list:
#     print(img[7:-4])

# import mmcv
#
# set_list = "/Users/rensike/Files/temp/voc_mini/ImageSets/Main/val.txt"
# file_list = mmcv.list_from_file(set_list)
#
# for file_path in file_list:
#     filename = file_path + ".xml"
#     source_path = os.path.join("/Users/rensike/Files/temp/voc_mini/Annotations",filename)
#     dst_path = os.path.join("/Users/rensike/Files/temp/voc_mini_val",filename)
#
#     shutil.copyfile(source_path,dst_path)

# source_train_path = "/Users/rensike/Work/data/shougang_cold/poc/batch_1/train"
# dist_train_path = "/Users/rensike/Work/data/shougang_cold/poc/batch_1_jpg/train"
#
# source_val_path = "/Users/rensike/Work/data/shougang_cold/poc/batch_1/val"
# dist_val_path = "/Users/rensike/Work/data/shougang_cold/poc/batch_1_jpg/val"
#
# if not os.path.exists(dist_train_path):
#     os.makedirs(dist_train_path)
#
# if not os.path.exists(dist_val_path):
#     os.makedirs(dist_val_path)
#
# for class_name in os.listdir(source_train_path):
#     dist_path = os.path.join(dist_train_path,class_name)
#     if not os.path.exists(dist_path):
#         os.makedirs(dist_path)
#
#     for img_path in glob.glob(os.path.join(source_train_path,class_name,"*.bmp")):
#         img_name = os.path.split(img_path)[-1].rsplit(".",1)[0]
#         im = Image.open(img_path)
#         im.save(os.path.join(dist_path,img_name + ".jpg"))
#
# for class_name in os.listdir(source_val_path):
#     dist_path = os.path.join(dist_val_path,class_name)
#     if not os.path.exists(dist_path):
#         os.makedirs(dist_path)
#
#     for img_path in glob.glob(os.path.join(source_val_path,class_name,"*.bmp")):
#         img_name = os.path.split(img_path)[-1].rsplit(".",1)[0]
#         im = Image.open(img_path)
#         im.save(os.path.join(dist_path,img_name + ".jpg"))

def pcb_img_ps():
    datapath = "/Users/rensike/Work/pcb/ps"
    for i in range(7):
        im = Image.open(os.path.join(datapath,"{}.jpeg".format(i)))

        anno = json.load(open(os.path.join(datapath,"{}.json".format(i)),"r"))
        xy_0,xy_1 = anno["shapes"][0]["points"]

        draw = ImageDraw.Draw(im)
        draw.rectangle((xy_0[0], xy_0[1], xy_1[0], xy_1[1]), "black")

        im.save(os.path.join(datapath,"results","{}.jpeg".format(i)))

        del draw


def pcb_img_crop():
    datapath = "/Users/rensike/Work/pcb/ps"
    for i in range(7):
        # im = cv2.imread(os.path.join(datapath,"{}.jpeg".format(i)),0)

        im = Image.open(os.path.join(datapath,"{}.jpeg".format(i)))

        # print(im.shape)

        anno = json.load(open(os.path.join(datapath,"{}.json".format(i)),"r"))
        xy_0,xy_1 = anno["shapes"][0]["points"]

        print("xy_0[1]:",xy_0[1])
        print("xy_1[1]:",xy_1[1])
        print("xy_0[0]:",xy_0[0])
        print("xy_1[0]:",xy_1[0])

        im = np.array(im)
        crop_img = im[xy_0[1]:xy_1[1],xy_0[0]:xy_1[0]]

        # draw = ImageDraw.Draw(im)
        # draw.rectangle((xy_0[0], xy_0[1], xy_1[0], xy_1[1]), "black")

        Image.fromarray(crop_img).save(os.path.join(datapath,"ps_croped","{}.jpeg".format(i)))


        # cv2.imwrite(os.path.join(datapath,"ps_croped","{}.jpeg".format(i)),crop_img)

if __name__ == '__main__':
    pcb_img_crop()