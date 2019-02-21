#coding=utf-8
from __future__ import print_function
import tensorflow as tf
import VGG16
import My_tool
import read_data
import os


data_dir = "./data/cifar-10-batches-bin/"
checkpoint_path = "./logs/model/"
pre_tain_weights = "./vgg16_pretrain/vgg16.npy"
tra_log_dir = "./logs/train/"
val_log_dir = "./logs/val/"
load_pretrain_weights = True
batch_size = 32
IMG_W = 32
IMG_H = 32
IMG_D = 3
N_CLASSES = 10
learning_rate = 0.01
MAX_STEP = 30000
def train():
    with tf.name_scope("input"):
        train_image_batch, train_label_batch = read_data.read_cifar10(data_dir=data_dir, is_train=True, shuffle=True, batch_size=batch_size)
        val_image_batch, val_label_batch = read_data.read_cifar10(data_dir=data_dir, is_train=False, shuffle=True, batch_size=batch_size)

        x = tf.placeholder(tf.float32, shape=[batch_size, IMG_W, IMG_H, IMG_D])
        y_ = tf.placeholder(tf.int16, shape=[batch_size, N_CLASSES])
        # print(0)


    logits = VGG16.vgg16(x, N_CLASSES, True)
    loss = My_tool.loss(logits=logits, labels=y_)
    accuracy = My_tool.accuracy(logits, y_)
    # print(1)


    my_global_step = tf.Variable(0, trainable=False, name="global_step")
    train_op = My_tool.optimizer(learning_rate=learning_rate, loss=loss, global_step=my_global_step)
    # print(2)

    saver = tf.train.Saver(tf.global_variables())
    summary_op = tf.summary.merge_all()
    # print(3)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    # print(4)

    ckpt = tf.train.get_checkpoint_state(checkpoint_path)
    if ckpt and ckpt.model_checkpoint_path:
        print("<load check_point_model...>")
        saver.restore(sess=sess, save_path=ckpt.model_checkpoint_path)
        print("<load check_point_model done!>")
    elif load_pretrain_weights:
        print("<load pre_train_weights...>")
        My_tool.load_pretrain_weights(pertrain_dir=pre_tain_weights, sess=sess, skip_layer=["fc6", "fc7", "fc8"])
        print("<load pre_train_weights done!>")
    # print(5)

    coord = tf.train.Coordinator()  # 创建协调器，用于关闭多线程
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    # print(6)

    tra_summary_writer = tf.summary.FileWriter(logdir=tra_log_dir, graph=sess.graph)
    val_summary_writer = tf.summary.FileWriter(logdir=val_log_dir, graph=sess.graph)
    # print(7)

    try:
        # print(8)
        for step in range(MAX_STEP):
            i = sess.run(my_global_step)
            print("training_step  %d" % i)
            if coord.should_stop() or i >= MAX_STEP:
                break
            train_images, train_labels = sess.run([train_image_batch, train_label_batch])
            _, tra_loss, tra_accuracy, summary_str = sess.run([train_op, loss, accuracy, summary_op],
                                                              feed_dict={x: train_images, y_: train_labels})
            if (i % 50 == 0) or (i+1 == MAX_STEP):
                tra_summary_writer.add_summary(summary_str, global_step=i)
                print("Step  %d, tra_loss = %.4f, tra_accuracy = %.4f%%" % (i, tra_loss, tra_accuracy))

            if (i % 200 == 0) or (i + 1 == MAX_STEP):
                val_images, val_labels = sess.run([val_image_batch, val_label_batch])
                _, val_loss, val_accuracy, summary_str = sess.run([train_op, loss, accuracy, summary_op],
                                                                  feed_dict={x: val_images, y_: val_labels})
                val_summary_writer.add_summary(summary_str, global_step=i)
                print("**Step  %d, val_loss = %.4f, val_accuracy = %.4f%%" % (i, val_loss, val_accuracy))

            if (i % 1000 == 0) or (i + 1 == MAX_STEP):
                save_path = saver.save(sess=sess, save_path=os.path.join(checkpoint_path, "vgg16"), global_step=my_global_step)
                print("Model saved in path: %s" % save_path)

    except tf.errors.OutOfRangeError:
        print('<something error!>')
    finally:
        print("Done training -- epoch limit reached or Manual stop")
        coord.request_stop()

    coord.join(threads)
    sess.close()

train()