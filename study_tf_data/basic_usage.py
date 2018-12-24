# -*- coding: utf-8 -* -
import tensorflow as tf

# ********************构建数据集***********************
'''创建只包含一个组件的dataset'''
dataset1 = tf.data.Dataset.from_tensor_slices(tf.random_uniform([4,10]))
print("数据集dataset1中每个元素的类型：",dataset1.output_types)  # ==> "tf.float32"
print("数据集dataset1中每个元素的shape：",dataset1.output_shapes) # ==> "(10,)"


'''创建包含两个组件的dataset'''
dataset2 = tf.data.Dataset.from_tensor_slices(
    (
        tf.random_uniform([4]),
        tf.random_uniform([4,100],maxval=100,dtype=tf.int32)
    )
)
print("数据集dataset2中每个组件元素的类型：",dataset2.output_types) # ==> "(tf.float32, tf.int32)"
print("数据集dataset2中每个组件元素的shape:",dataset2.output_shapes) # ==> "((), (100,))"


'''将数据集合并到一个对象中'''
dataset3 = tf.data.Dataset.zip((dataset1,dataset2))
print("数据集dataset3中各个组件元素的类型：",dataset3.output_types) # ==> (tf.float32, (tf.float32, tf.int32))
print("数据集dataset3中各个组件元素的shape：",dataset3.output_shapes) # ==> "(10, ((), (100,)))"


'''为数据集中每个组件进行命名，用于区分不同组件数据集的含义，例如：可以表示训练样本的不同特征'''
dataset4 = tf.data.Dataset.from_tensor_slices(
    {
        "a":tf.random_uniform([4]),
        "b":tf.random_uniform([4,10],maxval=100,dtype=tf.int32)
    }
)
print("数据集dataset4中各个组件元素的类型：",dataset4.output_types)       # ==> "{'a': tf.float32, 'b': tf.int32}"
print("数据集dataset4中各个组件元素的shape：",dataset4.output_shapes)   # ==> "{'a': (), 'b': (100,)}"


# ********************对数据集中的元素进行操作***********************
'''对数据集中的每一个元素进行操作'''
dataset1 = dataset1.map(lambda x: -x)
dataset2 = dataset2.flat_map(lambda x,y:x+1)        # ???????


# ********************遍历数据集中的元素***********************
'''
iterator_usage.py
'''












