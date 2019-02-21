#coding=utf-8
from __future__ import print_function
import tensorflow as tf
import numpy as np

def con2d(layer_name, in_value, out_channels, kernel_size=[3, 3], strides=[1, 1, 1, 1], trainable=True):
    in_channels = in_value.get_shape()[-1]
    with tf.variable_scope(layer_name):
        w = tf.get_variable(name="weights", trainable=trainable,
                            shape=[kernel_size[0], kernel_size[1], in_channels, out_channels],
                            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable(name="biases", trainable=trainable,
                            shape=[out_channels],
                            initializer=tf.constant_initializer(0.0))
        x = tf.nn.conv2d(input=in_value, filter=w, strides=strides, padding="SAME", name="conv")
        x = tf.nn.bias_add(value=x, bias=b, name="bias_add")
        x = tf.nn.relu(x, name="relu")
        return x


def max_pooling(layer_name, in_value, kernel_size=[1, 2, 2, 1], strides=[1, 2, 2, 1]):
    x = tf.nn.max_pool(value=in_value, ksize=kernel_size, strides=strides, padding="SAME", name=layer_name)
    return x


def avg_pooling(layer_name, in_value, kernel_size=[1, 2, 2, 1], strides=[1, 2, 2, 1]):
    x = tf.nn.avg_pool(value=in_value, ksize=kernel_size, strides=strides, padding="SAME", name=layer_name)
    return x


def FC_layer(layer_name, in_value, out_nodes):
    shape = in_value.get_shape()
    if len(shape) == 4:
        size = shape[1].value * shape[2].value * shape[3].value
    else:
        size = shape[-1].value
    with tf.variable_scope(layer_name):
        w = tf.get_variable(name="weights", shape=[size, out_nodes], initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable(name="biase", shape=[out_nodes], initializer=tf.constant_initializer(value=0.0))
        flat_in = tf.reshape(in_value, [-1, size])

        x = tf.nn.bias_add(tf.matmul(flat_in, w), b)
        x = tf.nn.relu(x)
        return x


def batch_norm(in_value):
    epsilon = 1e-3  # e-3 代表 10的-3次方， 1e-3 代表 1×10的-3次方
    batch_mean, batch_var = tf.nn.moments(in_value, [0])  # moments 返回 x 的均值和方差
    x = tf.nn.batch_normalization(in_value, batch_mean, batch_var, offset=None, scale=None, variance_epsilon=epsilon)
    return x


def loss(logits, labels):
    with tf.name_scope("loss") as scope:
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels, name="cross_entropy")
        loss = tf.reduce_mean(cross_entropy, name="loss")
        tf.summary.scalar(name=scope+'/loss', tensor=loss)
        return loss


def accuracy(logits, labels):
    with tf.name_scope("accuracy") as scope:
        correct = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        correct = tf.cast(correct, tf.float32)
        accuracy = tf.reduce_mean(correct)*100.0
        tf.summary.scalar(name=scope+"accuracy", tensor=accuracy)
        return accuracy


def optimizer(learning_rate, loss, global_step):
    with tf.name_scope("optimizer"):
        train_op = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss=loss, global_step=global_step)
        return train_op


def load_pretrain_weights(pertrain_dir, sess, skip_layer=None):
    data_dict = np.load(pertrain_dir, encoding="latin1").item()
    # keys = sorted(data_dict.key())
    for key in data_dict:
        if key not in skip_layer:
            with tf.variable_scope(key, reuse=True):
                for subkey, data in zip(("weights", "biases"), data_dict[key]):
                    sess.run(tf.get_variable(subkey).assign(data))
