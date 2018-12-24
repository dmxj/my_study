## TF-SLIM 基本使用：
TF-Slim 是 TensorFlow 中一个用来构建、训练、评估复杂模型的轻量化库。  
教程参考：https://blog.csdn.net/u014061630/article/details/80632736  
原文地址：https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/slim/README.md
#### Slim的优点：
*Slim 模块可以使模型的构建、训练、评估变得简单* 
+ 允许用户用紧凑的代码定义模型。这主要由 arg_scope、大量的高级 layers 和 variables 来实现。这些工具增加了代码的可读性和维护性，减少了复制、粘贴超参数值出错的可能性，并且简化了超参数的调整。
+ 通过提供常用的 regularizers 来简化模型的开发。很多常用的计算机视觉模型（例如 VGG、AlexNet）在 Slim 里面已经有了实现。这些模型开箱可用，并且能够以多种方式进行扩展（例如，给内部的不同层添加 multiple heads）。
+ Slim使得 “复杂模型的扩展” 及 “从一些现存的模型 ckpt 开始训练” 变得容易。  

#### 文件顺序：
+ basic_usage.py 
+ repeat_stack_usage.py  
+ arg_scope_usage.py  
+ define_vgg16.py
+ slim_net_usage.py
+ slim_train_usage.py
+ restore_model_part_usage.py
+ restore_model_diff_varname.py
+ slim_finetune_usage.py
+ slim_evaluate_loop.py
+ tf_slim_image_model_usage.py




