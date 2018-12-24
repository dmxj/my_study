#!/usr/bin/env bash
python create_coco_tf_record.py --logtostderr \
      --include_masks=True \
      --train_image_dir="./dataset/coco/train/" \
      --val_image_dir="./dataset/coco/val/" \
      --test_image_dir="./dataset/coco/test/" \
      --train_annotations_file="./dataset/coco/annotations/instances_train.json" \
      --val_annotations_file="./dataset/coco/annotations/instances_val.json" \
      --testdev_annotations_file="./dataset/coco/annotations/instances_test.json" \
      --output_dir="./results/coco_tf_record/"