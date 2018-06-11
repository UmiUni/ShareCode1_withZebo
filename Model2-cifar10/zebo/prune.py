import os
import sys
import time
import pickle
import numpy as np
import pandas as pd
sys.path.append("/home/pkugoodspeed/git/Jogchat/ShareCode1_withZebo/Model2-cifar10") # ABSOLUTE PATH for "Model2-cifar10"
from alexnet import alexnet
from cifar10_input import cifar10

class Algorithm2:
    train_set = None
    test_set = None
    x = None
    y = None
    keep_prob = None
    sess = None
    weights = None
    weight_arr = None
    accuracy = None
    sparsity = None
    thresholds = None
    error_toler = None
    current_error = None

    def __init__(self, error_toler, achieved_sparsity, extra_sparsity):
        """
        error_toler: float, error tolerence
        achieved_sparsity: dict mapping from layer name to achieved sparsities
        extra_sparsity: dict mapping form layer name to the extra sparsities for each layer
        """
        self.train_set = cifar10(data_dir='cifar-10-batches-py/', batch_size=100, subset='train', use_distortion=True)
        self.test_set = cifar10(data_dir='cifar-10-batches-py/', batch_size=100, subset='test', use_distortion=False)
        self.error_toler = error_toler
        self.current_error = 1.
        self.sparsity = dict(
            [(key, achieved_sparsity[key] + extra_sparsity[key]) for key in achieved_sparsity.keys()])
        for key in self.sparsity.keys():
            sparsity[key] = max(0., min(1., sparsity[key])) # confine the sparsity between 0 and 1
        
        self.x = tf.placeholder(tf.float32, [None, 32, 32, 3], name="x")
        self.y = tf.placeholder(tf.float32, [None, 10], name="y_")
        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")

        self.weights, logits = alexnet(self.x, self.keep_prob)

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y_))
        train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)

        correct_prediction = tf.equal(tf.argmax(logit, 1), tf.argmax(y_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        saver = tf.train.Saver(self.weights)
        
        self.sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, './models/alexnet_ckpt')  # Load weights, not very familiar with this statment
        for key in self.sparsity.keys():
            self.weight_arr[key] = self.sess.run(self.weights[key])
        

    def getTestError(self):
        test_acc = 0.
        for _ in range(self.test_set.num_iter_per_epoch):
            x_data, y_data = self.test_set.next_batch()
            test_acc = test_acc + sess.run(accuracy, feed_dict={self.x: x_data, self.y: y_data, self.keep_prob: 1.0})
        test_acc /= 1. * test_cifar10.num_iter_per_epoch
        return 1. - test_acc

    def getTrainError(self):
        train_acc = 0.
        for _ in range(self.train_set.num_iter_per_epoch):
            x_data, y_data = self.train_set.next_batch()
            train_acc = train_acc + sess.run(accuracy, feed_dict={self.x: x_data, self.y: y_data, self.keep_prob: 1.0})
        train_acc /= 1. * train_cifar10.num_iter_per_epoch
        return 1. - train_acc

    def getThred(self):
        for key, sps in self.sparsity.items():
            assert sps >= 0 and sps <= 1, "Invalid sparsity value"
            if sps == 0.:
                self.thresholds[key] = -np.inf
            elif sps == 1:
                self.thresholds[key] = np.inf
            else:
                flat = sorted(map(abs, np.reshape(self.weight_arr[key], [-1])))
                self.thresholds[key] = flat[int(len(flat) * sps)]

    def train(self):
        """ Train certain number of epochs """
        # You need to set keep_prop value during training
        pass

    def prune(self):
        for key in self.sparsity.keys():
            warray = self.sess.run(self.weights[key])
            warray[abs(warray) < self.thresholds[key]] = 0
            self.weights[key].load(warray)


    def run(self):
        self.current_error = self.getTestError()
        self.getThresholds()
        while True:
            cached_error = self.current_error + 1.
            cached_weights = dict(
                [(key, self.sess.run(self.weights[key])) for key in self.sparsity.keys()])  # Needs cache the weights of the previous step
            while cached_error >= self.current_error:
                cached_error = self.current_error
                self.train()
                self.prune()
                self.current_error = self.getTestError()  # Can introduce a patience variable
            if cached_error > self.error_toler:
                for key in self.sparsity.keys():
                    self.sparsity[key] -= 0.01 # According to the paper, every step, the sparsity of each layer decrease bty 1.%
                    self.sparsity[key] = max(0., self.sparsity[key])
                    self.weights[key].load(self.weight_arr[key]) # Not very clear about this part
                    self.getThresholds()
                continue
            else:
                for key in self.sparsity.keys():
                    self.weights[key].load(cached_weights[key])
                    self.weight_arr[key] = cached_weights[key]

    def dump(self):
        """ Output network """
        pass


if __name__ == "__main__":
    algo2 = Algorithm2(error_toler, achieved_sparsity, extra_sparsity)
    # achived_sparsity should be read from a json or some other types of files
    algo2.run()
    algo2.dump()