import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pickle
import tensorflow as tf
sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 70 * 325]) 
y_ = tf.placeholder(tf.float32, shape=[None, 2])


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1) 
    return tf.Variable(initial)
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape) 
    return tf.Variable(initial)
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_7x5(x):
    return tf.nn.max_pool(x, ksize=[1, 7, 5, 1],
        strides=[1, 7, 5, 1], padding='SAME')

def max_pool_5x5(x):
    return tf.nn.max_pool(x, ksize=[1, 5, 5, 1],
        strides=[1, 5, 5, 1], padding='SAME')

def max_pool_7x1(x):
    return tf.nn.max_pool(x, ksize=[1, 7, 1, 1],
        strides=[1, 7, 1, 1], padding='SAME')

def max_pool_2x5(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 5, 1],
        strides=[1, 2, 5, 1], padding='SAME')

def max_pool_2x1(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 1, 1],
        strides=[1, 2, 1, 1], padding='SAME')

def average_pool_5x5(x):
    return tf.nn.avg_pool(x, ksize=[1, 5, 5, 1],
        strides=[1, 5, 5, 1], padding='SAME')

def average_pool_2x5(x):
    return tf.nn.avg_pool(x, ksize=[1, 2, 5, 1],
        strides=[1, 2, 5, 1], padding='SAME')

def average_pool_7x5(x):
    return tf.nn.avg_pool(x, ksize=[1, 7, 5, 1],
        strides=[1, 7, 5, 1], padding='SAME')

x_image = tf.reshape(x, [-1,70,325,1])

W_conv1 = weight_variable([5, 5, 1, 32]) 
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) 
h_pool1 = average_pool_7x5(h_conv1) # ------------------------> convolution

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) 
h_pool2 = average_pool_5x5(h_conv2) # ------------------------> convolution

nn = 1024 # ---------------------------------------------------> network

W_fc1 = weight_variable([ 2 * 13 * 64, nn])
b_fc1 = bias_variable([nn])
h_pool2_flat = tf.reshape(h_pool2, [-1, 2 * 13 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32) 
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([nn, 2]) 
b_fc2 = bias_variable([2])
y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2

cross_entropy = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y_conv))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1)) 

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) 
sess.run(tf.global_variables_initializer())

# load model
saver = tf.train.Saver()
ckpt = tf.train.get_checkpoint_state('.')
if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)

#names_labels = open("../SR-ARE-score/names_labels.txt", "r").readlines()
names_pickle = open("../SR-ARE-score/names_onehots.pickle", "rb")
pickle_data = pickle.load(names_pickle)

onehots_data = pickle_data["onehots"].reshape(len(pickle_data["onehots"]), 70 * 325)
#rating_data = np.array([ ([0,1] if line[-2] == "1" else [1,0]) for line in names_labels], dtype=float)

file = open("labels.txt", "w")

for i in range(len(onehots_data)):
    # print("test accuracy %g"%accuracy.eval(feed_dict={
    # x: np.array([onehots_data[i]]), y_: np.array([rating_data[i]]), keep_prob: 1.0}))

    acc = float(accuracy.eval(feed_dict={
    x: np.array([onehots_data[i]]), y_: np.array([[1, 0]]), keep_prob: 1.0}))

    if acc == 1:
        file.write("0\n")
    else:
         file.write("1\n")
    
file.close()