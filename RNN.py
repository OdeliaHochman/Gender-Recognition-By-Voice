import numpy as np
from os import walk
import os
from python_speech_features import mfcc
import scipy.io.wavfile as wav
import tensorflow.compat.v1 as tf

tf.disable_eager_execution()


def prepare_data(path):
    data_features_matrix = np.empty((0, 13), float)
    data_labels = np.empty((0, 1), int)
    for (dirpath, dirnames, filenames) in walk(path):
        for filename in filenames:
            f = os.path.basename(dirpath)
            if f == "male":
                a = np.array([[0]])
                data_labels = np.vstack([data_labels, a])
            else:
                a = np.array([[1]])
                data_labels = np.vstack([data_labels, a])
            (rate, sig) = wav.read(dirpath + "\\" + filename)
            mfcc_feat = mfcc(sig, rate)
            mfcc_feat_act = np.mean(mfcc_feat, 0)
            features = np.reshape(mfcc_feat_act, 13)
            data_features_matrix = np.vstack([data_features_matrix, features])  # add row to matrix
    return data_features_matrix, data_labels


train_data_x, train_data_y = prepare_data("C:\\Users\\odelia\\Desktop\\deep learning\\train")
test_data_x, test_data_y = prepare_data("C:\\Users\\odelia\\Desktop\\deep learning\\test")

train_data_x = np.array(train_data_x).reshape(1400, 15, 13, 1)
train_data_y = np.insert(train_data_y, 1, 1 - train_data_y[:, 0], axis=1)
train_data_y = np.array(train_data_y).reshape(1400, 15, 2)

test_data_x = np.array(test_data_x).reshape(420, 15, 13, 1)
test_data_y = np.insert(test_data_y, 1, 1 - test_data_y[:, 0], axis=1)
test_data_y = np.array(test_data_y).reshape(420, 15, 2)

possible_label = 2
num_past_features = 13
num_of_epochs = 100
cell_size = 30

x = tf.placeholder(tf.float32, [None, num_past_features, 1])
y = tf.placeholder(tf.float32, [None, possible_label])

lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(cell_size, forget_bias=1.0)
output, _ = tf.nn.dynamic_rnn(lstm_cell, x, dtype=tf.float32)

output = tf.transpose(output, [1, 0, 2])
last = output[-1]

W = tf.Variable(tf.truncated_normal([cell_size, possible_label], stddev=0.1))
b = tf.Variable(tf.constant(0.1, shape=[possible_label]))

z = tf.matmul(last, W) + b
res = tf.nn.softmax(z)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(res), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(res, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()

acc_trace = tf.summary.scalar('accuracy', accuracy)
loss_trace = tf.summary.scalar('loss', cross_entropy)

with tf.Session() as sess:
    file_writer1 = tf.summary.FileWriter('RNN/train', sess.graph)
    file_writer2 = tf.summary.FileWriter('RNN/test', sess.graph)
    file_writer3 = tf.summary.FileWriter('RNN/loss', sess.graph)
    sess.run(init)
    for epoch in range(num_of_epochs):
        acc = 0
        tr = 0
        for i in range(1400):
            batch_xs = train_data_x[i]
            batch_ys = train_data_y[i]
            _, curr_loss = sess.run([train_step, cross_entropy], feed_dict={x: batch_xs, y: batch_ys})
            acc += accuracy.eval(feed_dict={x: batch_xs, y: batch_ys})
            tr = tr + 1
        print("step %d, loss %g, training accuracy %g" % (epoch, curr_loss, acc / tr))
    acc = 0
    tr = 0
    for i in range(420):
        batch_xs = test_data_x[i]
        batch_ys = test_data_y[i]
        sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})
        acc += accuracy.eval(feed_dict={x: batch_xs, y: batch_ys})
        tr = tr + 1
    print("test accuracy %g" % (acc / tr))
