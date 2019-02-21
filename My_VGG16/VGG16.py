#coding=utf-8
from __future__ import print_function
import tensorflow as tf
import My_tool


def vgg16(images, num_classes, with_bn=True):
    x = My_tool.con2d(layer_name="conv1_1", in_value=images, out_channels=64, trainable=True)
    x = My_tool.con2d(layer_name="conv1_2", in_value=x, out_channels=64, trainable=True)
    x = My_tool.max_pooling(layer_name="pool1", in_value=x, kernel_size=[1, 2, 2, 1], strides=[1, 2, 2, 1])

    x = My_tool.con2d(layer_name="conv2_1", in_value=x, out_channels=128, trainable=True)
    x = My_tool.con2d(layer_name="conv2_2", in_value=x, out_channels=128, trainable=True)
    x = My_tool.max_pooling(layer_name="pool2", in_value=x, kernel_size=[1, 2, 2, 1], strides=[1, 2, 2, 1])

    x = My_tool.con2d(layer_name="conv3_1", in_value=x, out_channels=256, trainable=True)
    x = My_tool.con2d(layer_name="conv3_2", in_value=x, out_channels=256, trainable=True)
    x = My_tool.con2d(layer_name="conv3_3", in_value=x, out_channels=256, trainable=True)
    x = My_tool.max_pooling(layer_name="pool3", in_value=x, kernel_size=[1, 2, 2, 1], strides=[1, 2, 2, 1])

    x = My_tool.con2d(layer_name="conv4_1", in_value=x, out_channels=512, trainable=True)
    x = My_tool.con2d(layer_name="conv4_2", in_value=x, out_channels=512, trainable=True)
    x = My_tool.con2d(layer_name="conv4_3", in_value=x, out_channels=512, trainable=True)
    x = My_tool.max_pooling(layer_name="pool4", in_value=x, kernel_size=[1, 2, 2, 1], strides=[1, 2, 2, 1])

    x = My_tool.con2d(layer_name="conv5_1", in_value=x, out_channels=512, trainable=True)
    x = My_tool.con2d(layer_name="conv5_2", in_value=x, out_channels=512, trainable=True)
    x = My_tool.con2d(layer_name="conv5_3", in_value=x, out_channels=512, trainable=True)
    x = My_tool.max_pooling(layer_name="pool5", in_value=x, kernel_size=[1, 2, 2, 1], strides=[1, 2, 2, 1])

    x = My_tool.FC_layer(layer_name="fc6", in_value=x, out_nodes=4096)
    if with_bn:
        x = My_tool.batch_norm(x)
    x = My_tool.FC_layer(layer_name="fc7", in_value=x, out_nodes=4096)
    if with_bn:
        x = My_tool.batch_norm(x)
    x = My_tool.FC_layer(layer_name="fc8", in_value=x, out_nodes=num_classes)

    return x

