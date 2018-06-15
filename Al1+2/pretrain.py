import tensorflow as tf
from alexnet import alexnet
from cifar10_input import cifar10
import time

DISPLAY_STEP = 100
EPOCHS = 30
train_cifar10 = cifar10(data_dir='cifar-10-batches-py/', batch_size=100, subset='train', use_distortion=True)
test_cifar10 = cifar10(data_dir='cifar-10-batches-py/', batch_size=100, subset='test', use_distortion=False)

with tf.Graph().as_default() as g_train:

    x = tf.placeholder(tf.float32, [None, 32, 32, 3], name="x")
    y_ = tf.placeholder(tf.float32, [None, 10], name="y_")
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")

    weights, logit = alexnet(x, keep_prob)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logit, labels=y_))
    train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)

    correct_prediction = tf.equal(tf.argmax(logit, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    saver = tf.train.Saver()

with tf.Session(graph=g_train) as sess:
    f = open('log_train.txt', 'w')
    def get_acc():
        test_acc = 0.
        for _ in range(test_cifar10.num_iter_per_epoch):
            batch = test_cifar10.next_batch()
            test_acc = test_acc + sess.run(accuracy, feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        test_acc = test_acc / test_cifar10.num_iter_per_epoch
        return test_acc

    sess.run(tf.global_variables_initializer())

    g_train.finalize()
    for epoch in range(1, EPOCHS + 1):
        for _ in range(1, train_cifar10.num_iter_per_epoch + 1):
            batch = train_cifar10.next_batch()
            if _ % DISPLAY_STEP == 0:
                train_acc = sess.run(accuracy, feed_dict={x:batch[0], y_:batch[1], keep_prob: 1.0})
                print("epoch %d, step %d, training acc %g" % (epoch, _ , train_acc))
                f.write("epoch %d, step %d, training acc %g\n" % (epoch, _ , train_acc))
            sess.run(train_op, feed_dict={x:batch[0], y_:batch[1], keep_prob:0.5})

    start = time.clock()
    
    test_acc = 0
    for _ in range(test_cifar10.num_iter_per_epoch):
        batch = test_cifar10.next_batch()
        test_acc = test_acc + sess.run(accuracy, feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
    test_acc = test_acc / test_cifar10.num_iter_per_epoch

    end = time.clock()
    print("test acc %g" % test_acc + ", inference time:" + str(end - start))
    f.write("test acc %g" % test_acc + ", inference time:" + str(end - start) + '\n')
    saver.save(sess, "./models/alexnet_ckpt")
    f.close()