#!/usr/bin/env bash
python create_pascal_tf_record.py \
        --data_dir=./dataset/voc/VOC2012 \
        --label_map_path=./data/pascal_label_map.pbtxt \
        --output_path=./results/pascal_tf_record/pascal.record