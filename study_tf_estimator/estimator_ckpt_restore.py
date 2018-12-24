# -*- coding: utf-8 -* -
'''
从磁盘中恢复模型检查点
'''
import tensorflow as tf
import iris_data

# 定义保存好checkpoints的目录
model_dir = "./model_dir"

# 加载训练数据用来获取features_columns
(train_x, _), (_, _) = iris_data.load_data()
my_feature_columns = []
for key in train_x.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))

# 需要定义和训练时一样的结构的Estimator
classifier = tf.estimator.DNNClassifier(
        feature_columns=my_feature_columns,
        # Two hidden layers of 10 nodes each.
        hidden_units=[10, 10],
        # The model must choose between 3 classes.
        n_classes=3,
        # 定义恢复检查点的模型目录
        model_dir=model_dir
    )

# 构造预测数据
predict_x = {
    'SepalLength': [5.1, 5.9, 6.9],
    'SepalWidth': [3.3, 3.0, 3.1],
    'PetalLength': [1.7, 4.2, 5.4],
    'PetalWidth': [0.5, 1.5, 2.1],
}
predictions = classifier.predict(
    input_fn=lambda :iris_data.eval_input_fn(predict_x,labels=None,batch_size=len(predict_x))
)

for pred in predictions:
    class_id = pred["class_ids"][0]
    prob = pred["probabilities"][class_id]
    print("predict result is :{}, prob is {:.1f}%".format(iris_data.SPECIES[class_id],prob*100))


