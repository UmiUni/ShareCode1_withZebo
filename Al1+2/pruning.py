import tensorflow as tf
from alexnet import alexnet
from cifar10_input import cifar10
import time
import numpy as np
import pickle

train_cifar10 = cifar10(data_dir='cifar-10-batches-py/', batch_size=100, subset='train', use_distortion=True)
test_cifar10 = cifar10(data_dir='cifar-10-batches-py/', batch_size=100, subset='test', use_distortion=False)
error_tolerance = 0.30
decrease_rate = 0.01
extra_sparsiy = 0.05
epoch = 2
display_step = 100

with open('achieved_sparsity', 'rb') as f:
    sparsity = pickle.load(f)

for layer_name in sparsity.keys():
    sparsity[layer_name] += extra_sparsiy
    if sparsity[layer_name] > 1.:
        sparsity[layer_name] = 1.

threshold = {
    'w_conv1': 0.,
    'w_conv2': 0.,
    'w_conv3': 0.,
    'w_conv4': 0.,
    'w_conv5': 0.,
    'w_fc1': 0.,
    'w_fc2': 0.,
    'w_fc3': 0.
}

def get_threshold(sess, weight, sparsity):
    if sparsity > 1. or sparsity < 0.:
        raise ValueError('Sparsity out of range')
    if sparsity == 1.:
        return np.inf
    if sparsity == 0.:
        return -np.inf
    weight_arr = sess.run(weight)
    flat = np.reshape(weight_arr, [-1])
    flat_list = sorted(map(abs,flat))
    return flat_list[int(len(flat_list) * sparsity)]


def prune(sess, weight, threshold):
    weight_arr = sess.run(weight)
    under_threshold = abs(weight_arr) < threshold
    weight_arr[under_threshold] = 0
    return weight_arr

with tf.Graph().as_default() as pruning:

    x = tf.placeholder(tf.float32, [None, 32, 32, 3], name="x")
    y_ = tf.placeholder(tf.float32, [None, 10], name="y_")
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")

    weights, logit = alexnet(x, keep_prob)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logit, labels=y_))
    train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)

    correct_prediction = tf.equal(tf.argmax(logit, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    saver = tf.train.Saver(weights)

with tf.Session(graph=pruning) as sess:

    def get_error():
        test_acc = 0.
        for _ in range(test_cifar10.num_iter_per_epoch):
            batch = test_cifar10.next_batch()
            test_acc = test_acc + sess.run(accuracy, feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        test_acc = test_acc / test_cifar10.num_iter_per_epoch
        return 1. - test_acc

    def update_threshold():
        for layer_name in sparsity.keys():
            threshold[layer_name] = get_threshold(sess, weights[layer_name], sparsity[layer_name])
        return None

    def prune_network():
        for layer_name in sparsity.keys():
            sparse_weight = prune(sess, weights[layer_name], threshold[layer_name])
            weights[layer_name].load(sparse_weight)
        return None

    def log_print(s):
        print(s)
        f.write(s + '\n')

    sess.run(tf.global_variables_initializer())

    saver.restore(sess, './models/alexnet_ckpt')

    pruning.finalize()

    f = open('log_pruning.txt', 'w')
    
    t = 0
    log_print('-------------------- Iter %d --------------------' % t)
    update_threshold()
    prune_network()

    current_error = get_error()

    while True:
        while True:
            for _ in range(1, train_cifar10.num_iter_per_epoch * epoch + 1):
                batch = train_cifar10.next_batch()
                if _ % display_step == 0:
                    train_acc = sess.run(accuracy, feed_dict={x:batch[0], y_:batch[1], keep_prob: 1.0})
                    log_print("Re-train step %d, error rate %g" % (_ , 1. - train_acc))
                sess.run(train_op, feed_dict={x:batch[0], y_:batch[1], keep_prob:0.5})

            log_print('Pruning network.')
            prune_network()

            previous_error = current_error
            current_error = get_error()
            t += 1 

            if current_error >= previous_error:
                log_print('After pruning, previous error %g is no more than current error %g. Evaluating pruned network.' % (previous_error, current_error))
                break
            else:
                log_print('After pruning, previous error %g is more than current error %g. Retraining network.' % (previous_error, current_error))
                log_print('-------------------- Iter %d --------------------' % t)


        if current_error <= error_tolerance:
            log_print('Current error %g is less than or equal to error toerance %g. Save pruned network to ./models/alexnet_pruning_ckpt.' % (current_error, error_tolerance))
            for layer_name in sparsity.keys():
                log_print('Layer %s, Sparsity %g' % (layer_name, sparsity[layer_name]))
            break
        else:
            log_print('Current error %g is more than error toerance %g. Decrese sparsity by 1%% in each layer and retrain the network.' % (current_error, error_tolerance))
            for layer_name in sparsity.keys():
                sparsity[layer_name] -= decrease_rate
                if sparsity[layer_name] < 0:
                    sparsity[layer_name] = 0.
            update_threshold()
            log_print('-------------------- Iter %d --------------------' % t)
    saver.save(sess, './models/alexnet_pruning_ckpt')
    f.close()
