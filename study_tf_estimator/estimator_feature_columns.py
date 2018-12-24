# -*- coding: utf-8 -* -
'''
特征列
'''
import tensorflow as tf

# ============== 【numeric_column】数值列，默认数据类型（tf.float32）============

# Defaults to a tf.float32 scalar.
numeric_feature_column = tf.feature_column.numeric_column(key="SepalLength")

# Represent a tf.float64 scalar.
numeric_feature_column2 = tf.feature_column.numeric_column(key="SepalLength", dtype=tf.float64)

# Represent a 10-element vector in which each cell contains a tf.float32.
vector_feature_column = tf.feature_column.numeric_column(key="Bowling", shape=10)

# Represent a 10x5 matrix in which each cell contains a tf.float32.
matrix_feature_column = tf.feature_column.numeric_column(key="MyMatrix", shape=[10,5])

# ============= 【bucketized_column】分桶列，用于存储one-hot编码形式的类别类型 =============
'''
假设：
日期 < 1960 年，表示为 [1, 0, 0, 0]；
日期 >= 1960 年但 < 1980 年，表示为 [0, 1, 0, 0]；
日期 >= 1980 年但 < 2000 年，表示为 [0, 0, 1, 0]；
日期 >= 2000 年，表示为 [0, 0, 0, 1]。
矢量边界为：1960,1980,2000,将年份分割成4类
'''
# First, convert the raw input to a numeric column.
numeric_feature_column3 = tf.feature_column.numeric_column("Year")

# Then, bucketize the numeric column on the years 1960, 1980, and 2000.
bucketized_feature_column = tf.feature_column.bucketized_column(
                                source_column = numeric_feature_column,
                                boundaries = [1960, 1980, 2000]
                            )

# ============= 【categorical_column_with_identity】分类标识列，将分类标识的数字表示为one-hot编码，可视为分桶列的一种特殊情况 ============
'''
0 ===> [1,0,0,0]
1 ===> [0,1,0,0]
2 ===> [0,0,1,0]
3 ===> [0,0,0,1]
'''
# Create categorical output for an integer feature named "my_feature_b",
# The values of my_feature_b must be >= 0 and < num_buckets
identity_feature_column = tf.feature_column.categorical_column_with_identity(
    key='my_feature_b',
    num_buckets=4) # Values [0, 4)

# In order for the preceding call to work, the input_fn() must return
# a dictionary containing 'my_feature_b' as a key. Furthermore, the values
# assigned to 'my_feature_b' must belong to the set [0, 4).
def input_fn():
    Label_values = [1,2,3,4]
    return ({ 'my_feature_a':[7, 9, 5, 2], 'my_feature_b':[3, 1, 2, 2]},
            [Label_values])

# 【categorical_column_with_vocabulary_list|categorical_column_with_vocabulary_file】
# =============== 分类词汇列，将字符串表示为one-hot类别编码 ===================
'''
"banana"    ===>    [1,0,0]
"apple"     ===>    [0,1,0]
"grape"     ===>    [0,0,1]
'''
# categorical_column_with_vocabulary_list 根据明确的词汇表将每个字符串映射到一个整数
# Key="color",here "color" is feature name from input fn,
# create a categorical feature by mapping the input to one of
# the elements in the vocabulary list.
vocabulary_feature_column = tf.feature_column.categorical_column_with_vocabulary_list(
        key="color",
        vocabulary_list=["kitchenware", "electronics", "sports"])

# 如果单词太多，上面的方法就不方便了，可以将所有的单词放入文件中，用categorical_column_with_vocabulary_file读取
# Given input "feature_name_from_input_fn" which is a string,
# create a categorical feature to our model by mapping the input to one of
# the elements in the vocabulary file
vocabulary_feature_column2 = tf.feature_column.categorical_column_with_vocabulary_file(
        key="color",
        vocabulary_file="words.txt",
        vocabulary_size=3)

# ============= 【categorical_column_with_hash_bucket】经过哈希处理的列，将特征通过hash再取模分为不同的类型 ================
'''
tf.feature_column.categorical_column_with_hash_bucket 函数使您能够指定类别的数量。
对于这种类型的特征列，模型会计算输入的哈希值，然后使用模运算符将其置于其中一个 hash_bucket_size 类别中，如以下伪代码所示：
# pseudocode
feature_id = hash(raw_feature) % hash_buckets_size
'''
hashed_feature_column = tf.feature_column.categorical_column_with_hash_bucket(
        key = "some_feature",
        hash_buckets_size = 100) # The number of categories


# =========== 组合列，将多个特征组合为一个特征 ================
'''
例如经度和纬度，这两个特征分开的意义不大，合成一个特征比较好。
'''
def make_dataset(latitude, longitude, labels):
    assert latitude.shape == longitude.shape == labels.shape

    features = {'latitude': latitude.flatten(),
                'longitude': longitude.flatten()}
    labels=labels.flatten()

    return tf.data.Dataset.from_tensor_slices((features, labels))

# Bucketize the latitude and longitude using the `edges`
latitude_bucket_fc = tf.feature_column.bucketized_column(
    tf.feature_column.numeric_column('latitude'),
    [-45,0,45]) # 纬度边界

longitude_bucket_fc = tf.feature_column.bucketized_column(
    tf.feature_column.numeric_column('longitude'),
    [[-135,-90,-45,0,45,90,135]]) # 经度边界

# Cross the bucketized columns, using 5000 hash bins.
crossed_lat_lon_fc = tf.feature_column.crossed_column(
    [latitude_bucket_fc, longitude_bucket_fc], 5000)

fc = [
    latitude_bucket_fc,
    longitude_bucket_fc,
    crossed_lat_lon_fc]

# Build and train the Estimator.
est = tf.estimator.LinearRegressor(fc, ...)


''' 指标列和嵌入列从不直接处理特征，而是将分类列视为输入。'''
# ============= 指标列，将每个类别视为独热矢量中的一个元素，其中匹配类别的值为 1，其余类别为 0 =============
'''
0 ===> [1,0,0,0]
1 ===> [0,1,0,0]
2 ===> [0,0,1,0]
3 ===> [0,0,0,1]
'''
# Create any type of categorical column.
categorical_column = ...

# Represent the categorical column as an indicator column.
indicator_column = tf.feature_column.indicator_column(categorical_column)


# ============= 嵌入列，使用较低维度的普通矢量而非独热向量来表示数据 =============
categorical_column = ... # Create any categorical column

# Represent the categorical column as an embedding column.
# This means creating an embedding vector lookup table with one element for each category.
embedding_dimensions = 100 # 设置嵌入向量的维度
embedding_column = tf.feature_column.embedding_column(
    categorical_column=categorical_column,
    dimension=embedding_dimensions)



'''
============ 将特征列传递给 Estimator ================
'''
'''
如下面的列表所示，并非所有 Estimator 都支持所有类型的 feature_columns 参数：
    - LinearClassifier 和 LinearRegressor：接受所有类型的特征列。
    - DNNClassifier 和 DNNRegressor：只接受密集列。其他类型的列必须封装在 indicator_column 或 embedding_column 中。
    - DNNLinearCombinedClassifier 和 DNNLinearCombinedRegressor：
        - linear_feature_columns 参数接受任何类型的特征列。
        - dnn_feature_columns 参数只接受密集列。
'''





