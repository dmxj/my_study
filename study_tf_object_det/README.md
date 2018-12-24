### tensorflow object detection的使用
需要先在环境变量中设置tf models的python path：
export PYTHONPATH=$PYTHONPATH:/usr/local/src/github/models/research:/usr/local/src/github/models/research/slim

+ make_record.sh:使用tf object detection中的脚本对pets数据集生成tf record，在pets数据集解压后的目录里面执行
+ train_pets.sh:使用预训练模型fine tune模型
+ export_model.sh:将训练好的模型转换并导出，导出的结果文件中包含用于预测的frozen模型：frozen_inference_graph.pb
+ eval_model.sh:评估模型

+ object_detection.py:使用模型进行预测的类
+ test.py:对每张图片进行预测并把分类得分大于某个阈值的预测结果保存到json文件中