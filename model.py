import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import time

TRAIN = True
PRUNE = True

mnist = input_data.read_data_sets('./MNIST-data', one_hot=True)
sess = tf.Session()

def alexnet(inputs, weights, keep_prob):
    def conv2d(name, x, W, b):
        return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, W, strides=[1,1,1,1], padding="SAME", name=name), b))

    def max_pool(name, x):
        return tf.nn.max_pool(x, ksize=[1,3,3,1], strides=[1,2,2,1], padding="SAME", name=name)

    def lrn(name, x):
        return tf.nn.lrn(x, depth_radius=2, alpha=0.0001, beta=0.75, bias=1.0, name=name)

    x = tf.reshape(inputs, [-1,28,28,1])

    conv1 = conv2d('conv1', x, weights['w_conv1'], weights['b_conv1'])
    lrn1 = lrn('lrn1', conv1)
    pool1 = max_pool('pool1', lrn1)

    conv2 = conv2d('conv2', pool1, weights['w_conv2'], weights['b_conv2'])
    lrn2 = lrn('lrn2', conv2)
    pool2 = max_pool('pool2', lrn2)

    conv3 = conv2d('conv3', pool2, weights['w_conv3'], weights['b_conv3'])

    conv4 = conv2d('conv4', conv3, weights['w_conv4'], weights['b_conv4'])

    conv5 = conv2d('conv5', conv4, weights['w_conv5'], weights['b_conv5'])
    pool5 = max_pool('pool5', conv5)

    flattened = tf.reshape(pool5, [-1, weights['w_fc1'].get_shape().as_list()[0]])
    fc1 = tf.nn.relu(tf.matmul(flattened, weights["w_fc1"]) + weights["b_fc1"], name='fc1')
    drpo1 = tf.nn.dropout(fc1, keep_prob, name='drop1')

    fc2 = tf.nn.relu(tf.matmul(drpo1, weights["w_fc2"]) + weights["b_fc2"], name='fc2')
    drpo2 = tf.nn.dropout(fc2, keep_prob, name='drpo2')

    fc3 = tf.matmul(drpo2, weights["w_fc3"]) + weights["b_fc3"]

    return fc3

def test(predict_logit):
    correct_prediction = tf.equal(tf.argmax(predict_logit, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = 0
    for i in range(20):
        batch = mnist.test.next_batch(500)
        result = result + sess.run(accuracy, feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
    result = result / 20.0
    return result

def get_threshold(weight, percentage=0.8):
    flat = tf.reshape(weight, [-1])
    flat_list = sorted(map(abs,sess.run(flat)))
    return flat_list[int(len(flat_list) * percentage)]

def prune(weights, threshold):
    weight_arr = sess.run(weights)
    under_threshold = abs(weight_arr) < threshold
    weight_arr[under_threshold] = 0
    return weight_arr, ~under_threshold

def delete_none_grads(grads):
    count = 0
    length = len(grads)
    while(count < length):
        if(grads[count][0] == None):
            del grads[count]
            length -= 1
        else:
            count += 1

def transfer_to_sparse(weight):
    weight_arr = sess.run(weight)
    values = weight_arr[weight_arr != 0]
    indices = np.transpose(np.nonzero(weight_arr))
    shape = list(weight_arr.shape)
    return [indices, values, shape]

weights = {
    'w_conv1': tf.Variable(tf.truncated_normal([3, 3, 1, 64], dtype=tf.float32, stddev=0.1), name='w_conv1'),
    'w_conv2': tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32, stddev=0.1), name='w_conv2'),
    'w_conv3': tf.Variable(tf.truncated_normal([3, 3, 128, 256], dtype=tf.float32, stddev=0.1), name='w_conv3'),
    'w_conv4': tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32, stddev=0.1), name='w_conv4'),
    'w_conv5': tf.Variable(tf.truncated_normal([3, 3, 256, 128], dtype=tf.float32, stddev=0.1), name='w_conv5'),
    'w_fc1': tf.Variable(tf.truncated_normal([4*4*128, 1024], dtype=tf.float32, stddev=0.1), name='w_fc1'),
    'w_fc2': tf.Variable(tf.random_normal([1024, 1024], dtype=tf.float32, stddev=0.1), name='w_fc2'),
    'w_fc3': tf.Variable(tf.random_normal([1024, 10], dtype=tf.float32, stddev=0.1), name='w_fc3'),

    'b_conv1': tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32), trainable=True, name='b_conv1'),
    'b_conv2': tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32), trainable=True, name='b_conv2'),
    'b_conv3': tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32), trainable=True, name='b_conv3'),
    'b_conv4': tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32), trainable=True, name='b_conv4'),
    'b_conv5': tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32), trainable=True, name='b_conv5'),
    'b_fc1': tf.Variable(tf.constant(0.0, shape=[1024], dtype=tf.float32), trainable=True, name='b_fc1'),
    'b_fc2': tf.Variable(tf.constant(0.0, shape=[1024], dtype=tf.float32), trainable=True, name='b_fc2'),
    'b_fc3': tf.Variable(tf.constant(0.0, shape=[10], dtype=tf.float32), trainable=True, name='b_fc3')
}

if TRAIN:
    x = tf.placeholder(tf.float32, [None, 784], name="x")
    y_ = tf.placeholder(tf.float32, [None, 10], name="y_")
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")

    logit = alexnet(x, weights, keep_prob)

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=y_)
    train_op = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(logit, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess.run(tf.global_variables_initializer())

    for i in range(1, 5001):
        batch = mnist.train.next_batch(50)
        if i % 100 == 0:
            train_acc = sess.run(accuracy, feed_dict={x:batch[0], y_:batch[1], keep_prob:0.5})
            print("step %d, training acc %g" % (i , train_acc))
        sess.run(train_op, feed_dict={x:batch[0], y_:batch[1], keep_prob:0.5})

    start = time.clock()
    test_acc = test(logit)
    end = time.clock()
    print("test acc %g" % test_acc + ", inference time:" + str(end - start))
    saver = tf.train.Saver()
    saver.save(sess, "./models/alexnet_ckpt")

if PRUNE:
    ITERATION = 20
    PERCENTAGE = 0.7
    print("total pruning iteration: %d. pruning percentage each iter: %g" % (ITERATION, PERCENTAGE))
    saver = tf.train.Saver()
    saver.restore(sess, './models/model_ckpt_dense')
    p = 1.0
    for i in range(1, ITERATION + 1:
        tf.reset_default_graph()
        p = p * PERCENTAGE
        print("\033[0;31miteration %d, p=%g\033[0m" % (i, p))

        weights_idx = {}

        for name in sorted(list(weights.keys())):
            if not name.startswith('b'):
                threshold = get_threshold(weights[name], percentage=(1.0 - p))
                sparse_weights, idx = prune(weights[name], threshold)
                weights_idx[name] = idx
                sess.run(tf.assign(weights[name], sparse_weights))
                print("none-zero in %s : %d" % (name, np.sum(sparse_weights != 0)))

        for var in tf.global_variables():
            if sess.run(tf.is_variable_initialized(var)) == False:
                sess.run(var.initializer)

        x = tf.placeholder(tf.float32, [None, 784], name="x")
        y_ = tf.placeholder(tf.float32, [None, 10], name="y_")
        keep_prob = tf.placeholder(tf.float32, name="keep_prob")
        logit = alexnet(x, weights, keep_prob)
        test_acc = test(logit)
        print("\033[0;31mtest acc after iteration %d pruning: %g\033[0m" % (i, test_acc))

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=y_)
        trainer = tf.train.AdamOptimizer(1e-4)
        grads = trainer.compute_gradients(cross_entropy)

    #     delete_none_grads(grads)

        count = 0
        for grad, var in grads:
            name = var.name.split(':')[0]
            if not name.startswith('b'):
                idx_in = tf.cast(tf.constant(weights_idx[name]), tf.float32)
                grads[count] = (tf.multiply(idx_in, grad), var)
            count += 1

        train_op = trainer.apply_gradients(grads)

        correct_prediction = tf.equal(tf.argmax(logit, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        for var in tf.global_variables():
            if sess.run(tf.is_variable_initialized(var)) == False:
                sess.run(var.initializer)

        for j in range(5000):
            batch = mnist.train.next_batch(50)
            if (j == 100 or j == 500 or j == 1000 or j == 1500 or j == 4999):
                train_acc = sess.run(accuracy, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
                print("retraining step %d, acc %g" % (j, train_acc))
            sess.run(train_op, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

        for name in sorted(list(weights.keys())):
            weight_arr = sess.run(weights[name])
            if not name.startswith('b'):
                print("none-zero in %s after retrain: %d" % (name, np.sum(weight_arr != 0)))

        test_acc = test(logit)
        print("\033[0;31mtest acc after iteration %d pruning and retraining: %g\033[0m" % (i, test_acc))

        saver = tf.train.Saver(weights)
        saver.save(sess, "./models/alexnet_pruning_ckpt", global_step=i)

