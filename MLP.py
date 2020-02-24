from os import walk
import os
from python_speech_features import mfcc
import scipy.io.wavfile as wav
import tensorflow as tf
import numpy as np

def prepare_data(path):
    x = np.empty((0, 13), float)
    y = np.empty((0, 1), int)
    for (dirpath, dirnames, filenames) in walk(path):
        for filename in filenames:
            f = os.path.basename(dirpath)
            if f == "male":
                a = np.array([[0]])
                y = np.vstack([y, a])
            else:
                a = np.array([[1]])
                y = np.vstack([y, a])
            (rate, sig) = wav.read(dirpath + "\\" + filename)
            mfcc_feat = mfcc(sig, rate)
            mfcc_acc = np.mean(mfcc_feat, 0)
            features = np.reshape(mfcc_acc, 13)
            x = np.vstack([x, features])
    return x, y


def logistic_fun(z):
    return 1 / (1.0 + np.exp(-z))

def test(path):
    really_female = 0
    really_male = 0
    classified_female = 0
    classified_male = 0
    really_and_class_female = 0
    really_and_class_male = 0
    for (dirpath, dirnames, filenames) in walk(path):
        for filename in filenames:
            f = os.path.basename(dirpath)
            if f == "female":
                really_female = really_female+1
            if f == "male":
                really_male = really_male+1
            (rate, sig) = wav.read(dirpath + "\\" + filename)
            mfcc_feat = mfcc(sig, rate)
            mfcc_acc = np.mean(mfcc_feat, 0)
            features = np.reshape(mfcc_acc, 13)
            prediction = y.eval(session=sess, feed_dict={x: [features]})[0][0]
            if prediction > 0.5:
                classified_female = classified_female+1
                if f == 'female':
                    really_and_class_female = really_and_class_female+1
            else:
                classified_male = classified_male+1
                if f == 'male':
                    really_and_class_male = really_and_class_male+1
    print('ReallyFemale: %d  ReallyMale: %d ClassifiedFemale %d ClassifiedMale: %d RealyAndClassFemale: %d RealyAndClassMale: %d' %
          (really_female, really_male, classified_female, classified_male, really_and_class_female, really_and_class_male))


features = 13
(hidden1_size, hidden2_size, hidden3_size, hidden4_size) = (70, 100, 60, 30)
x = tf.placeholder(tf.float32, [None, features])
y_ = tf.placeholder(tf.float32, [None, 1])
W1 = tf.Variable(tf.truncated_normal([features, hidden1_size], stddev=0.1))
b1 = tf.Variable(tf.constant(0.1, shape=[hidden1_size]))
z1 = tf.nn.relu(tf.matmul(x, W1)+b1)
W2 = tf.Variable(tf.truncated_normal([hidden1_size, 1], stddev=0.1))
b2 = tf.Variable(tf.constant(0.1, shape=[hidden2_size]))
z2 = tf.nn.relu(tf.matmul(z1, W2)+b2)
W3 = tf.Variable(tf.truncated_normal([hidden2_size, hidden3_size], stddev=0.1))
b3 = tf.Variable(tf.constant(0.1, shape=[hidden3_size]))
z3 = tf.nn.relu(tf.matmul(z2, W3)+b3)
W4 = tf.Variable(tf.truncated_normal([hidden3_size, hidden4_size], stddev=0.1))
b4 = tf.Variable(tf.constant(0.1, shape=[hidden4_size]))
z4 = tf.nn.relu(tf.matmul(z3, W4)+b4)
W5 = tf.Variable(tf.truncated_normal([hidden4_size, 1], stddev=0.1))
b5 = tf.Variable(0.)

y = tf.nn.sigmoid(tf.matmul(z4, W5)+b5)
loss1 = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_, logits=y)
loss = tf.reduce_mean(loss1)
update = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

source =("C:\\Users\\odelia\\Desktop\\deep learning\\train")

data_x, data_y = prepare_data(source)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for i in range(0, 10000):
    sess.run(update, feed_dict={x: data_x, y_: data_y})
    if i % 1000 == 0:
        print('Iteration:', i, ' W5:', sess.run(W5), ' b5:', sess.run(b5), ' loss:',
          loss.eval(session=sess, feed_dict={x: data_x, y_: data_y}))


test("C:\\Users\\odelia\\Desktop\\deep learning\\test")
