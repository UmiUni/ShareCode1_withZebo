import tensorflow as tf

def alexnet(x, keep_prob):
    def conv2d(name, x, W, b):
        return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, W, strides=[1,1,1,1], padding="SAME", name=name), b))

    def max_pool(name, x):
        return tf.nn.max_pool(x, ksize=[1,3,3,1], strides=[1,2,2,1], padding="SAME", name=name)

    def lrn(name, x):
        return tf.nn.lrn(x, depth_radius=4, alpha=0.0001, beta=0.75, bias=1.0, name=name)

    weights = {
        'w_conv1': tf.Variable(tf.truncated_normal([3, 3, 3, 32], dtype=tf.float32, stddev=0.1), name='w_conv1'),
        'w_conv2': tf.Variable(tf.truncated_normal([3, 3, 32, 64], dtype=tf.float32, stddev=0.1), name='w_conv2'),
        'w_conv3': tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32, stddev=0.1), name='w_conv3'),
        'w_conv4': tf.Variable(tf.truncated_normal([3, 3, 128, 256], dtype=tf.float32, stddev=0.1), name='w_conv4'),
        'w_conv5': tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32, stddev=0.1), name='w_conv5'),
        'w_fc1': tf.Variable(tf.truncated_normal([4*4*256, 1024], dtype=tf.float32, stddev=0.1), name='w_fc1'),
        'w_fc2': tf.Variable(tf.truncated_normal([1024, 1024], dtype=tf.float32, stddev=0.1), name='w_fc2'),
        'w_fc3': tf.Variable(tf.truncated_normal([1024, 10], dtype=tf.float32, stddev=0.1), name='w_fc3'),

        'b_conv1': tf.Variable(tf.random_normal([32], dtype=tf.float32), trainable=True, name='b_conv1'),
        'b_conv2': tf.Variable(tf.random_normal([64], dtype=tf.float32), trainable=True, name='b_conv2'),
        'b_conv3': tf.Variable(tf.random_normal([128], dtype=tf.float32), trainable=True, name='b_conv3'),
        'b_conv4': tf.Variable(tf.random_normal([256], dtype=tf.float32), trainable=True, name='b_conv4'),
        'b_conv5': tf.Variable(tf.random_normal([256], dtype=tf.float32), trainable=True, name='b_conv5'),
        'b_fc1': tf.Variable(tf.random_normal([1024], dtype=tf.float32), trainable=True, name='b_fc1'),
        'b_fc2': tf.Variable(tf.random_normal([1024], dtype=tf.float32), trainable=True, name='b_fc2'),
        'b_fc3': tf.Variable(tf.random_normal([10], dtype=tf.float32), trainable=True, name='b_fc3')
    }

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

    return weights, fc3

