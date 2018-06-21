# -*- coding: UTF-8 -*-

# 《TensorFlow实战》05 简单神经网络


from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
sess = tf.InteractiveSession()  # 创建一个会话

def weight_variable(shape):
    '''初始化权重函数,truncated_normal创建 标准差为0.1的截断正态函数'''
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    '''初始化偏置函数，由于使用ReLU要加一些正值0.1，避免死亡节点（dead neurons）'''
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)

def conv2d(x, W):
    '''x:输入 w:卷积参数 例[5, 5, 1, 32]:5, 5为卷积核尺寸
    1:为多少channel 彩色是3 灰度是1
    32:为卷积核的数量（这个卷积层会提取的多少类特征）
    strides:卷积模板移动的步长，[1, 2, 2, 1]官方前后必须为1，中间两个分别代表横向和竖向
    padding：代表边界处理方式，SAME即输入和输出保持同样尺寸'''
    return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = 'SAME')

def max_pool_2x2(x):
    '''池化层函数 max_pool:最大池化函数'''
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                          padding='SAME')

#x:特征
x = tf.placeholder(tf.float32, [None, 784])
#y_真实的label
y_ = tf.placeholder(tf.float32, [None, 10])
'''卷积神经网络会利用到原有的空间结构信息，因此需要将1D的输入向量转化为2D图片结构（1x784->28x28）
因为只有一个颜色通道，故最终尺寸为[-1, 28, 28, 1]其中：-1代表样本数量不固定，1代表颜色通道数量'''
x_image = tf.reshape(x, [-1, 28, 28, 1])

'''先定义weights和bias,然后使用conv2d函数进行卷积操作并加上偏置，
接着使用ReLU激活函数进行非线性处理，最好使用max_pool_2x2对卷积的输出结果进行池化操作'''
w_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1)+b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

#定义第二个卷积层,不同在于特征变为64
w_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2)+b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

'''经历两次2x2步长的最大池化，边长变为1/4，图片尺寸由28x28->7x7
由于第二个卷积层的卷积核数量为64，其输出tensor尺寸为7x7x64。
使用tf.reshape函数对其变形，转化为1D向量，然后连接一个全连接层，
隐含节点为1024，并使用Relu激活函数'''
w_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1)+b_fc1)

#为减轻过拟合，使用一个dropout层
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#dropout层输出连softmax层，得到最后的概率输出
w_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2)+b_fc2)


# 定义损失函数 cross_entropy
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y_conv),
                                              reduction_indices=[1]))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# 评测准确率
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 开始训练，mini-batch为50， 20000次迭代，每100次输出评测结果
tf.global_variables_initializer().run()
for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, train accuracy %g"%(i, train_accuracy))

    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

# 在测试集上进行测试
print("test accuracy%g"%accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))