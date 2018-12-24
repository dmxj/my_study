# 学习TensorFlow Estimator
参考：https://tensorflow.google.cn/guide/estimators?hl=zh-cn 

#### 什么是Estimator
```angular2html
 Estimator - 一种可极大地简化机器学习编程的高阶 TensorFlow API。Estimator 会封装下列操作：
· 训练
· 评估
· 预测
· 导出以供使用
您可以使用我们提供的预创建的 Estimator，也可以编写自定义 Estimator。所有 Estimator（无论是预创建的还是自定义）都是基于 tf.estimator.Estimator 类的类。
```

#### Estimator的优势
```angular2html
Estimator 具有下列优势：
· 您可以在本地主机上或分布式多服务器环境中运行基于 Estimator 的模型，而无需更改模型。此外，您可以在 CPU、GPU 或 TPU 上运行基于 Estimator 的模型，而无需重新编码模型。
· Estimator 简化了在模型开发者之间共享实现的过程。
· 您可以使用高级直观代码开发先进的模型。简言之，采用 Estimator 创建模型通常比采用低阶 TensorFlow API 更简单。
· Estimator 本身在 tf.layers 之上构建而成，可以简化自定义过程。
· Estimator 会为您构建图。
· Estimator 提供安全的分布式训练循环，可以控制如何以及何时：
    · 构建图
    · 初始化变量
    · 开始排队
    · 处理异常
    · 创建检查点文件并从故障中恢复
    · 保存 TensorBoard 的摘要
使用 Estimator 编写应用时，您必须将数据输入管道从模型中分离出来。这种分离简化了不同数据集的实验流程。
```

#### 预创建的模型
预创建的 Estimator 会为您创建和管理 Graph 和 Session 对象。此外，借助预创建的 Estimator，您只需稍微更改下代码，就可以尝试不同的模型架构。


#### 使用预创建的Estimator的TesorFlow程序通常包含下列四个步骤：
1. 编写一个或多个数据集导入函数。例如，您可以创建一个函数来导入训练集，并创建另一个函数来导入测试集。每个数据集导入函数都必须返回两个对象：
    · 一个字典，其中键是特征名称，值是包含相应特征数据的张量（或 SparseTensor）
    · 一个包含一个或多个标签的张量
例如，以下代码展示了输入函数的基本框架：
```angular2html
def input_fn(dataset):
   ...  # manipulate dataset, extracting the feature dict and the label
   return feature_dict, label
```
2. 定义特征列。 每个 tf.feature_column 都标识了特征名称、特征类型和任何输入预处理操作。
例如，以下代码段创建了三个存储整数或浮点数据的特征列。
前两个特征列仅标识了特征的名称和类型。
第三个特征列还指定了一个 lambda，该程序将调用此 lambda 来调节原始数据：
```angular2html
# Define three numeric feature columns.
population = tf.feature_column.numeric_column('population')
crime_rate = tf.feature_column.numeric_column('crime_rate')
median_education = tf.feature_column.numeric_column('median_education',
                    normalizer_fn=lambda x: x - global_education_mean)
```
3. 实例化相关的预创建的 Estimator。 例如，下面是对名为 LinearClassifier 的预创建 Estimator 进行实例化的示例代码：
```angular2html
# Instantiate an estimator, passing the feature columns.
estimator = tf.estimator.LinearClassifier(
    feature_columns=[population, crime_rate, median_education],
    )
```
4. 调用训练、评估或推理方法。例如，所有 Estimator 都提供训练模型的 train 方法。
```angular2html
# my_training_set is the function created in Step 1
estimator.train(input_fn=my_training_set, steps=2000)
```

### 文件顺序
+ 将keras模型转化为Estimator:  
create_estimator_from_keras_model.py  
+ 使用预置的全连接网络进行鸢尾花分类：  
iris_data.py  
premade_estimator_usage.py  
+ 训练时保存模型检查点：
estimator_ckpt_save.py
+ 从目录中恢复模型检查点，并进行预测：
estimator_ckpt_restore.py
+ 各种类型的feature column的使用：
estimator_feature_columns.py
+ 自定义Estimator:
estimator_custom.py



