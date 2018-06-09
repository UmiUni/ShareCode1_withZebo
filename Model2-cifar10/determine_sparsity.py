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
epoch = 2
display_step = 100
sparsity = {
    'w_conv1': 1.0,
    'w_conv2': 1.0,
    'w_conv3': 1.0,
    'w_conv4': 1.0,
    'w_conv5': 1.0,
    'w_fc1': 1.0,
    'w_fc2': 1.0,
    'w_fc3': 1.0
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

with tf.Graph().as_default() as determine_sparsity:

    x = tf.placeholder(tf.float32, [None, 32, 32, 3], name="x")
    y_ = tf.placeholder(tf.float32, [None, 10], name="y_")
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")

    weights, logit = alexnet(x, keep_prob)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logit, labels=y_))
    train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)

    correct_prediction = tf.equal(tf.argmax(logit, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    saver = tf.train.Saver(weights)

with tf.Session(graph=determine_sparsity) as sess:

    f = open('log_determine_sparsity.txt', 'w')

    def get_error():
        test_acc = 0.
        for _ in range(test_cifar10.num_iter_per_epoch):
            batch = test_cifar10.next_batch()
            test_acc = test_acc + sess.run(accuracy, feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        test_acc = test_acc / test_cifar10.num_iter_per_epoch
        return 1. - test_acc

    sess.run(tf.global_variables_initializer())

    saver.restore(sess, './models/alexnet_ckpt')

    determine_sparsity.finalize()

    for layer_name in list(sparsity.keys())[::-1]:
        print('Start pruning Layer: %s' % layer_name)
        f.write('Start pruning Layer: %s\n' % layer_name)
        threshold = get_threshold(sess, weights[layer_name], sparsity[layer_name])
        sparse_weight = prune(sess, weights[layer_name], threshold)
        weights[layer_name].load(sparse_weight)

        t = 0
        error_before_retrain = get_error()
        print('Iter %d, error rate %g' % (t, error_before_retrain))
        f.write('Iter %d, error rate %g\n' % (t, error_before_retrain))

        while error_before_retrain > error_tolerance:
            for _ in range(1, train_cifar10.num_iter_per_epoch * epoch + 1):
                batch = train_cifar10.next_batch()
                if _ % display_step == 0:
                    train_acc = sess.run(accuracy, feed_dict={x:batch[0], y_:batch[1], keep_prob: 1.0})
                    print("Re-train step %d, error rate %g" % (_ , 1. - train_acc))
                    f.write("Re-train step %d, error rate %g\n" % (_ , 1. - train_acc))
                sess.run(train_op, feed_dict={x:batch[0], y_:batch[1], keep_prob:0.5})

            threshold = get_threshold(sess, weights[layer_name], sparsity[layer_name])
            sparse_weight = prune(sess, weights[layer_name], threshold)
            weights[layer_name].load(sparse_weight)

            t += 1
            error_after_retrain = get_error()

            print('Iter %d, error rate %g' % (t, error_after_retrain))
            f.write('Iter %d, error rate %g\n' % (t, error_after_retrain))

            if error_after_retrain <= error_before_retrain:
                print('Current error rate %g is less than previous error rate %g, dscrease sparsity by 1%%' % (error_after_retrain, error_before_retrain))
                f.write('Current error rate %g is less than previous error rate %g, dscrease sparsity by 1%%\n' % (error_after_retrain, error_before_retrain))
                if sparsity[layer_name] > 0:
                    sparsity[layer_name] -= decrease_rate
                    if sparsity[layer_name] == 0:
                        break

            error_before_retrain = error_after_retrain


        print('Achieved sparsity in Layer %s is %g' % (layer_name, sparsity[layer_name]))
        f.write('Achieved sparsity in Layer %s is %g\n' % (layer_name, sparsity[layer_name]))

    with open('achieved_sparsity', 'wb') as f_s:
        pickle.dump(sparsity, f_s)

    print('Done')
    f.write('Done\n')

    for layer_name in sparsity.keys():
        print('Achieved sparsity in Layer %s is %g' % (layer_name, sparsity[layer_name]))
        f.write('Achieved sparsity in Layer %s is %g\n' % (layer_name, sparsity[layer_name]))
    f.close()
