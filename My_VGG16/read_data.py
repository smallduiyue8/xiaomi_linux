#coding=utf-8
from __future__ import print_function
import tensorflow as tf
import numpy as np
import os


def read_cifar10(data_dir, is_train, shuffle, batch_size, n_classes=10):
    img_width = 32
    img_height = 32
    img_depth = 3
    img_bytes = img_depth * img_height * img_width
    label_bytes = 1

    with tf.name_scope("input"):

        if is_train:
            filenames = [os.path.join(data_dir, "data_batch_%d.bin" % i) for i in np.arange(1, 6)]   # filenames is a list
        else:
            filenames = [os.path.join(data_dir, "test_batch.bin")]
        filename_queue = tf.train.string_input_producer(filenames)  # 这个函数需要传入一个文件名list，系统会自动将它转为一个文件名队列。
        # tf.FixedLengthRecordReader是读取固定长度字节数信息(针对bin文件使用FixedLengthRecordReader读取比较合适)，
        # 结果表明下次调用时会接着上次读取的位置继续读取文件，而不会从头开始读取。
        reader = tf.FixedLengthRecordReader(label_bytes + img_bytes)
        key, value = reader.read(filename_queue)
        record_bytes = tf.decode_raw(value, tf.uint8)  # tf.decode_raw函数的意思是将原来编码为字符串类型的变量重新变回来
        label = tf.slice(record_bytes, [0], [label_bytes])
        label = tf.cast(label, tf.int32)  # cast()用于改变某个张量的数据类型

        image_raw = tf.slice(record_bytes, [label_bytes], [img_bytes])
        image_raw = tf.reshape(image_raw, [img_depth, img_height, img_width])
        image = tf.transpose(image_raw, (1, 2, 0))  #  convert from D/H/W to H/W/D
        image = tf.cast(image, tf.float32)

        # data argumentation (数据增强）
        # image = tf.random_crop(image, [24, 24, 3])  # randomly crop the image size to 24 x 24
        # image = tf.image.random_flip_left_right(image)
        # image = tf.image.random_brightness(image, max_delta=63)
        # image = tf.image.random_contrast(image, lower=0.2, upper=1.8)

        image = tf.image.per_image_standardization(image)  # 图片标准化， 减去均值，除以标准差

        if shuffle:
            images, label_batch = tf.train.shuffle_batch([image, label],
                                                         batch_size=batch_size,
                                                         num_threads=64,
                                                         capacity=20000,
                                                         min_after_dequeue=3000)
        else:
            images, label_batch = tf.train.batch([image, label],
                                                 batch_size=batch_size,
                                                 num_threads=64,
                                                 capacity=2000)
        ## ONE-HOT
        n_classes = n_classes
        label_batch = tf.one_hot(label_batch, depth=n_classes)
        label_batch = tf.cast(label_batch, dtype=tf.int32)
        label_batch = tf.reshape(label_batch, [batch_size, n_classes])

        return images, label_batch
