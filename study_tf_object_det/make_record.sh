#!/usr/bin/env bash
export object_detection_path=/usr/local/src/github/models/research
python $object_detection_path/object_detection/dataset_tools/
create_pet_tf_record.py --label_map_path=$object_detection_path/object_detection/data/pet_label_map.pbtxt --data_dir=`pwd`
--output_dir=`pwd`

