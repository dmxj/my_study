#!/usr/bin/env bash
INPUT_TYPE=image_tensor
# path to pipeline config file
PIPELINE_CONFIG_PATH=/workspace/tf_object_detection/faster_rcnn_resnet101_pets.config
# path to model.ckpt
TRAINED_CKPT_PREFIX=/workspace/tf_object_detection/train_pets_model_dir/model.ckpt-50000
# path to folder that will be used for export
EXPORT_DIR=/workspace/tf_object_detection/model_export
# path to tensorflow object detection model research dir
TF_MODEL_LIB_PATH=/usr/local/src/github/models/research
python $TF_MODEL_LIB_PATH/object_detection/export_inference_graph.py \
    --input_type=${INPUT_TYPE} \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --trained_checkpoint_prefix=${TRAINED_CKPT_PREFIX} \
    --output_directory=${EXPORT_DIR}
