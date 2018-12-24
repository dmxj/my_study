#!/usr/bin/env bash
# path to pipeline config file
PIPELINE_CONFIG_PATH=/workspace/tf_object_detection/faster_rcnn_resnet101_pets.config
# path to model directory
MODEL_DIR=/workspace/tf_object_detection/train_pets_model_dir
# train steps
NUM_TRAIN_STEPS=50000
SAMPLE_1_OF_N_EVAL_EXAMPLES=1
# path to tensorflow object detection model research dir
TF_MODEL_LIB_PATH=/usr/local/src/github/models/research
python $TF_MODEL_LIB_PATH/object_detection/model_main.py \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --model_dir=${MODEL_DIR} \
    --num_train_steps=${NUM_TRAIN_STEPS} \
    --sample_1_of_n_eval_examples=$SAMPLE_1_OF_N_EVAL_EXAMPLES \
    --alsologtostderr
