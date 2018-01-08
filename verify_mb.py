import tensorflow as tf
from ippso.data.core import DataLoader
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.python.ops import init_ops
import os
import numpy as np
from datetime import datetime

# load the mb dataset
from ippso.data.mb import loaded_data

# parameters
batch_size = 200
training_epoch = 100
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

################################# load the whole dataset #################################
DataLoader.mode = 1
training_data=loaded_data.train['images']
training_label=loaded_data.train['labels']
validation_data=loaded_data.validation['images']
validation_label=loaded_data.validation['labels']
test_data=loaded_data.test['images']
test_label=loaded_data.test['labels']

# function of produce batch data
def produce_tf_batch_data(images, labels, batch_size):
    """
    produce batch data given batch_size

    :param images: images
    :type images: list
    :param labels: labels
    :type labels: list
    :param batch_size: batch size
    :type batch_size: int
    :return: a list of tensor containing the data
    :rtype: list
    """
    train_image = tf.cast(images, tf.float32)
    train_image = tf.reshape(train_image, [-1, 28, 28, 1])
    train_label = tf.cast(labels, tf.int32)
    #create input queues
    queue_images, queue_labels = tf.train.slice_input_producer([train_image, train_label], shuffle=True)
    queue_images = tf.image.per_image_standardization(queue_images)
    image_batch, label_batch = tf.train.batch([queue_images, queue_labels], batch_size=batch_size, num_threads=2,
                                              capacity=batch_size * 3)
    return image_batch, label_batch

################################# build the CNN #################################
is_training = tf.placeholder(tf.int8, [])
training_data, training_label = produce_tf_batch_data(training_data, training_label, batch_size)
validation_data, validation_label = produce_tf_batch_data(validation_data, validation_label, batch_size)
test_data, test_label = produce_tf_batch_data(test_data, test_label, batch_size)
bool_is_training = tf.cond(tf.equal(is_training, tf.constant(0, dtype=tf.int8)), lambda: tf.constant(True, dtype=tf.bool), lambda : tf.constant(False, dtype=tf.bool))
# when is_training==0 use training data, when is_training==1 use validation data, when is_training==2 use test data
X, y_ = tf.cond(tf.equal(is_training, tf.constant(0, dtype=tf.int8)), lambda : (training_data, training_label),
                lambda : tf.cond(tf.equal(is_training, tf.constant(1,dtype=tf.int8)), lambda : (validation_data, validation_label), lambda : (test_data, test_label)))
true_Y = tf.cast(y_, tf.int64)

# add input layer
output_list = []
output_list.append(X)

with slim.arg_scope([slim.conv2d, slim.fully_connected],
                    activation_fn=tf.nn.crelu,
                    normalizer_fn=slim.batch_norm,
                    weights_regularizer=None,
                    normalizer_params={'is_training': bool_is_training, 'decay': 0.99}
                    ):
    # add conv layer
    name_scope = 'conv_1'
    with tf.variable_scope(name_scope):
        filter_size, stride_size, feature_map_size = (2, 1, 26)
        conv = slim.conv2d(output_list[-1], feature_map_size, filter_size, stride_size,
                             weights_initializer=initializers.xavier_initializer(),
                             biases_initializer=init_ops.constant_initializer(0.1, dtype=tf.float32))
        output_list.append(conv)

    # add conv layer
    name_scope = 'conv_2'
    with tf.variable_scope(name_scope):
        filter_size, stride_size, feature_map_size = (6, 3, 82)
        conv = slim.conv2d(output_list[-1], feature_map_size, filter_size, stride_size,
                             weights_initializer=initializers.xavier_initializer(),
                             biases_initializer=init_ops.constant_initializer(0.1, dtype=tf.float32))
        output_list.append(conv)

    # add conv layer
    name_scope = 'conv_3'
    with tf.variable_scope(name_scope):
        filter_size, stride_size, feature_map_size = (8, 4, 114)
        conv = slim.conv2d(output_list[-1], feature_map_size, filter_size, stride_size,
                             weights_initializer=initializers.xavier_initializer(),
                             biases_initializer=init_ops.constant_initializer(0.1, dtype=tf.float32))
        output_list.append(conv)

    # add conv layer
    name_scope = 'conv_4'
    with tf.variable_scope(name_scope):
        filter_size, stride_size, feature_map_size = (7, 4, 107)
        conv = slim.conv2d(output_list[-1], feature_map_size, filter_size, stride_size,
                           weights_initializer=initializers.xavier_initializer(),
                           biases_initializer=init_ops.constant_initializer(0.1, dtype=tf.float32))
        output_list.append(conv)

    # add first fully-connected layer
    name_scope = 'fully-connected_1'
    with tf.variable_scope(name_scope):
        hidden_neuron_num = 1686
        input_data = slim.flatten(output_list[-1])
        full = slim.fully_connected(input_data, num_outputs=hidden_neuron_num,
                                      weights_initializer=initializers.xavier_initializer(),
                                      biases_initializer=init_ops.constant_initializer(0.1,
                                                                                       dtype=tf.float32))
        output_list.append(full)

    # add fully-connected output layer
    name_scope = 'fully-connected_2'
    with tf.variable_scope(name_scope):
        hidden_neuron_num = 10
        input_data = slim.flatten(output_list[-1])
        full = slim.fully_connected(input_data, num_outputs=hidden_neuron_num,
                                    weights_initializer=initializers.xavier_initializer(),
                                    biases_initializer=init_ops.constant_initializer(0.1,
                                                                                     dtype=tf.float32))
        output_list.append(full)

    # create tensors
    with tf.name_scope('loss'):
        logits = output_list[-1]
        cross_entropy = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_Y, logits=logits))
        loss = cross_entropy
    with tf.name_scope('train'):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        optimizer = tf.train.AdamOptimizer()
        train_op = slim.learning.create_train_op(loss, optimizer)
    with tf.name_scope('test'):
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), true_Y), tf.float32))

################################# train the CNN and output the results #################################
def test_one_epoch(sess, accuracy, cross_entropy, is_training, data_length, training_mode, X, true_Y):
    """
    test one epoch on validation data or test data
    :param sess: tensor session
    :param data_length: data length of validation or test data
    :param accuracy: accuracy variable in tensor session
    :param cross_entropy: cross_entropy variable in tensor session
    :param is_training: is_training variable in tensor session
    :param training_mode: training mode. 0:training, 1:validation, 2:test
    :return:
    """
    total_step = data_length // batch_size
    accuracy_list = []
    loss_list = []
    for _ in range(total_step):
        accuracy_str, loss_str, X_str, true_Y_str = sess.run([accuracy, cross_entropy, X, true_Y], {is_training: training_mode})
        accuracy_list.append(accuracy_str)
        loss_list.append(loss_str)
    mean_accu = np.mean(accuracy_list)
    mean_loss = np.mean(loss_list)
    return mean_accu, mean_loss

print('===start training===')
tf.reset_default_graph()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    steps_in_each_epoch = (training_data.shape[0] // batch_size)
    total_steps = int(training_epoch * steps_in_each_epoch)
    coord = tf.train.Coordinator()
    # threads = tf.train.start_queue_runners(sess, coord)
    try:
        threads = []
        for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
            threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))
        # training for the epoch number
        for i in range(total_steps):
            if coord.should_stop():
                break
            train_op_str, accuracy_str, loss_str, X_str, true_Y_str, is_training_str = sess.run(
                [train_op, accuracy, cross_entropy, X, true_Y, is_training],
                {is_training: 0}
            )
            if i % (2 * steps_in_each_epoch) == 0:
                mean_validation_accu, mean_validation_loss = test_one_epoch(sess, accuracy, cross_entropy,
                                                                                 is_training,
                                                                                 validation_data.shape[0], 1, X, true_Y)
                print('{}, {}, Step:{}/{}, training_loss:{}, acc:{}, validation_loss:{}, acc:{}'.format(
                datetime.now(), i // steps_in_each_epoch, i, total_steps, loss_str,
                accuracy_str, mean_validation_loss, mean_validation_accu))
                mean_test_accu, mean_test_loss = test_one_epoch(sess, accuracy,
                                                                     cross_entropy, is_training,
                                                                     test_data.shape[0],
                                                                     2, X, true_Y)
                print('test_loss:{}, acc:{}'.format(mean_test_loss, mean_test_accu))
        # validate and test the last epoch
        mean_validation_accu, mean_validation_loss = test_one_epoch(sess, accuracy, cross_entropy,
                                                                    is_training,
                                                                    validation_data.shape[0], 1, X, true_Y)
        print('===final result after the last epoch===')
        print('{}, {}, Step:{}/{}, training_loss:{}, acc:{}, validation_loss:{}, acc:{}'.format(
            datetime.now(), i // steps_in_each_epoch, i, total_steps, loss_str,
            accuracy_str, mean_validation_loss, mean_validation_accu))
        mean_test_accu, mean_test_loss = test_one_epoch(sess, accuracy,
                                                        cross_entropy, is_training,
                                                        test_data.shape[0],
                                                        2, X, true_Y)
        print('test_loss:{}, acc:{}'.format(mean_test_loss, mean_test_accu))

    except Exception as e:
        print(e)
        coord.request_stop(e)
    finally:
        print('finally...')
        coord.request_stop()
        coord.join(threads)
    print('===finish training===')








































