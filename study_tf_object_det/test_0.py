# -*- coding: utf-8 -* -
import glob

img_list = glob.glob("images/*.jpg")
for img in img_list:
    print(img[7:-4])
