# -*- coding: utf-8 -* -
'''
在图片上绘制框和文字
'''
import numpy as np
import os
import cv2
from PIL import Image,ImageDraw,ImageFont

def draw_bbox_use_pil(image_path,bbox_list,label_list,isShow=False,savePath=None):
    assert len(bbox_list) == len(label_list),"bounding box num must equal to label num"
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype('arial.ttf', 24)
    except IOError:
        font = ImageFont.load_default()

    for (bbox,label) in list(zip(bbox_list,label_list)):
        xmin,ymin,xmax,ymax = bbox[0],bbox[1],bbox[2],bbox[3]
        draw.line((xmin, ymin, xmin, ymax), fill=128, width=2)
        draw.line((xmin, ymin, xmax, ymin), fill=128, width=2)
        draw.line((xmin, ymax, xmax, ymax), fill=128, width=2)
        draw.line((xmax, ymin, xmax, ymax), fill=128, width=2)

        display_str_heights = font.getsize(label)[1]
        # Each display_str has a top and bottom margin of 0.05x.
        total_display_str_height = (1 + 2 * 0.05) * display_str_heights

        if ymin > total_display_str_height:
            text_bottom = ymin
        else:
            text_bottom = ymax + total_display_str_height

        text_width, text_height = font.getsize(label)
        margin = np.ceil(0.05 * text_height)

        draw.text((xmin + margin, text_bottom - text_height - margin), label, fill="green", spacing=1, font=font)

    if isShow:
        img.show()

    if savePath is not None:
        img.save(savePath)

def draw_bbox_use_cv2(image_path,bbox_list,label_list,isShow=False,savePath=None):
    img = cv2.imread(image_path)
    for (bbox,label) in list(zip(bbox_list,label_list)):
        xmin,ymin,xmax,ymax = bbox[0],bbox[1],bbox[2],bbox[3]

        cv2.rectangle(img, (round(xmin), round(ymin)), (round(xmax), round(ymax)), (0, 255, 0), 3)

        # Each display_str has a top and bottom margin of 0.05x.
        total_display_str_height = (1 + 2 * 0.05) * 24

        if ymin > total_display_str_height:
            text_bottom = ymin
        else:
            text_bottom = ymax + total_display_str_height

        text_width, text_height = cv2.getTextSize(label,cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.8,2)
        margin = np.ceil(0.05 * text_height)
        cv2.putText(img, label, (int(xmin + margin), int(text_bottom - text_height - margin)), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    1.8, (255, 0, 0), 2)

    if isShow:
        cv2.imshow(os.path.basename(image_path),img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if savePath is not None:
        cv2.imwrite(savePath,img)

if __name__ == '__main__':
    # ymin,xmin,ymax,xmax
    bbox_list = [
        [0.38530465960502625, 0.4303162097930908, 0.9935078620910645, 0.7767336368560791],
        [0.2521260678768158, 0.0, 0.9947431087493896, 0.969138503074646],
        [0.24669665098190308, 0.005808502435684204, 0.9966098666191101, 0.9132076501846313]
    ]

    label_list = ["person","person","person"]

    img_path = "/Users/rensike/Files/temp/voc_very_mini/JPEGImages/2011_000182.jpg"

    width, height = Image.open(img_path).size

    # xmin,ymin,xmax,ymax
    bbox_list = [[bbox[1]*width,bbox[0]*height,bbox[3]*width,bbox[2]*height] for bbox in bbox_list]

    print(bbox_list)

    draw_bbox_use_pil(img_path,bbox_list,label_list,isShow=True)

