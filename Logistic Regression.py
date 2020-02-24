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
            mfcc_feat_act = np.mean(mfcc_feat, 0)
            features = np.reshape(mfcc_feat_act, 13)
            x = np.vstack([x, features])
    return x, y

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
eps = 1e-12
x = tf.placeholder(tf.float32, [None, features])
y_ = tf.placeholder(tf.float32, [None, 1])
W = tf.Variable(tf.zeros([features,1]))
b = tf.Variable(tf.zeros([1]))
y = tf.nn.sigmoid(tf.matmul(x,W)+b)
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
        print('Iteration:', i, ' W:', sess.run(W), ' b:', sess.run(b), ' loss:',
          loss.eval(session=sess, feed_dict={x: data_x, y_: data_y}))


def logistic_fun(z):
    return 1 / (1.0 + np.exp(-z))



test("C:\\Users\\odelia\\Desktop\\deep learning\\test")