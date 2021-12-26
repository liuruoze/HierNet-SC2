import numpy as np
import tensorflow as tf


def dense_layer(data, out_dim, name, func=tf.nn.relu, trainable=True, norm=False, is_training=False):
    in_dim = data.get_shape().as_list()[-1]
    shape = [in_dim, out_dim]
    d = 1.0 / 500

    with tf.variable_scope(name):
        w_init = tf.random_uniform_initializer(-d, d)
        b_init = tf.random_uniform_initializer(-d, d)

        w = tf.get_variable(name="weights", shape=shape, initializer=w_init, trainable=trainable)
        b = tf.get_variable(name="bias", shape=[out_dim], initializer=b_init, trainable=trainable)

        output = tf.matmul(data, w) + b

        if norm:
            output = batch_norm(output, is_training, 'BN', trainable)

        if func is not None:
            output = func(output)

    return output


def conv2d_layer(data, filter_size, filter_num, name, stride=1, func=tf.nn.relu, trainable=True):
    in_dim = data.get_shape().as_list()[-1]
    shape = [filter_size, filter_size, in_dim, filter_num]
    d = 1.0 / np.sqrt(filter_size * filter_size * in_dim)

    with tf.variable_scope(name):
        w_init = tf.random_uniform_initializer(-d, d)
        b_init = tf.random_uniform_initializer(-d, d)

        w = tf.get_variable(name="weights", shape=shape, initializer=w_init, trainable=trainable)
        b = tf.get_variable(name="bias", shape=[filter_num], initializer=b_init, trainable=trainable)

        output = tf.nn.conv2d(data, w, strides=[1, stride, stride, 1], padding='SAME', data_format="NHWC") + b
        if func is not None:
            output = func(output)

    return output


def max_pool(data, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME'):
    val = tf.nn.max_pool(data, ksize=ksize, strides=strides, padding=padding)

    return val


def batch_norm(data, is_training, name, trainable=True):
    with tf.variable_scope(name):
        data_norm = tf.contrib.layers.batch_norm(data, is_training=is_training, trainable=trainable,
                                                 zero_debias_moving_mean=True, decay=0.9)

    return data_norm

# def batch_norm(x, train_phase, scope_bn, trainable=True):
#     with tf.variable_scope(scope_bn):
#         beta = tf.Variable(tf.constant(0.0, shape=[x.shape[-1]]), name='beta', trainable=False)
#         gamma = tf.Variable(tf.constant(1.0, shape=[x.shape[-1]]), name='gamma', trainable=False)
#         axises = np.arange(len(x.shape) - 1)
#         batch_mean, batch_var = tf.nn.moments(x, axes=[0], name='moments')
#         ema = tf.train.ExponentialMovingAverage(decay=0.999)
#
#         def mean_var_with_update():
#             ema_apply_op = ema.apply([batch_mean, batch_var])
#             with tf.control_dependencies([ema_apply_op]):
#                 return tf.identity(batch_mean), tf.identity(batch_var)
#
#         mean, var = tf.cond(train_phase, mean_var_with_update,
#                             lambda: (ema.average(batch_mean), ema.average(batch_var)))
#         normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
#     return normed
