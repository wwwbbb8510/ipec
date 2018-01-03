import numpy as np
import logging
from datetime import datetime
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import init_ops
from ippso.ip.decoder import Decoder
from .particle import Particle
import os

def initialise_cnn_evaluator(training_epoch=None, batch_size=None, training_data=None, training_label=None, validation_data=None,
                             validation_label=None, max_gpu=None, first_gpu_id=None, class_num=None, regularise=0, dropout=0,
                             mean_centre=None, mean_divisor=None, stddev_divisor=None, test_data=None, test_label=None, optimise=False):
    training_epoch = 5 if training_epoch is None else training_epoch
    max_gpu = None if max_gpu is None else max_gpu
    batch_size = 200 if batch_size is None else batch_size
    class_num = 10 if class_num is None else class_num
    if training_data is None and training_label is None and validation_data is None and validation_label is None:
        from ippso.data.mnist import get_training_data, get_validation_data, get_test_data
        training_data = get_training_data()['images'] if training_data is None else training_data
        training_label = get_training_data()['labels'] if training_label is None else training_label
        validation_data = get_validation_data()['images'] if validation_data is None else validation_data
        validation_label = get_validation_data()['labels'] if validation_label is None else validation_label
        test_data = get_test_data()['images'] if test_data is None else test_data
        test_label = get_test_data()['labels'] if test_label is None else test_label
    return CNNEvaluator(training_epoch, batch_size, training_data, training_label,
                        validation_data, validation_label, max_gpu, first_gpu_id, class_num, regularise, dropout, mean_centre, mean_divisor, stddev_divisor,
                        test_data=test_data, test_label=test_label, optimise=optimise)


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


class Evaluator:
    """
    Evaluator
    """
    def __init__(self):
        """
        constructor
        """

class CNNEvaluator(Evaluator):
    """
    CNN evaluator
    """
    def __init__(self, training_epoch, batch_size, training_data, training_label,
                 validation_data, validation_label, max_gpu, first_gpu_id, class_num=10, regularise=0, dropout=0,
                 mean_centre=None, mean_divisor=None, stddev_divisor=None, test_data=None, test_label=None, optimise=False):
        """
        constructor

        :param training_epoch: the training epoch before evaluation
        :type training_epoch: int
        :param batch_size: batch size
        :type batch_size: int
        :param training_data: training data
        :type training_data: numpy.array
        :param training_label: training label
        :type training_label: numpy.array
        :param validation_data: validation data
        :type validation_data: numpy.array
        :param validation_label: validation label
        :type validation_label: numpy.array
        :param test_data: test data
        :type test_data: numpy.array
        :param test_label: test label
        :type test_label: numpy.array
        :param class_num: class number
        :type class_num: int
        :param max_gpu: max number of gpu to be used
        :type max_gpu: int
        :param first_gpu_id: the first gpu ID. The GPUs will start from the first gpu ID
        and continue using the following GPUs until reaching the max_gpu number
        :type first_gpu_id: int
        """
        self.training_epoch = training_epoch
        self.batch_size = batch_size
        self.training_data = training_data
        self.training_label = training_label
        self.validation_data = validation_data
        self.validation_label = validation_label
        self.test_data = test_data
        self.test_label = test_label
        self.class_num = class_num
        self.max_gpu = max_gpu
        self.first_gpu_id = first_gpu_id
        self.regularise = regularise
        self.dropout = dropout
        self.optimise = optimise

        self.training_data_length = self.training_data.shape[0]
        self.validation_data_length = self.validation_data.shape[0]
        self.test_data_length = self.test_data.shape[0] if self.test_data is not None else 0
        self.decoder = Decoder(mean_centre=mean_centre, mean_divisor=mean_divisor, stddev_divisor=stddev_divisor)

        # set visible cuda devices
        if self.max_gpu is not None:
            for i in range(self.max_gpu):
                gpu_id = i
                if self.first_gpu_id is not None:
                    gpu_id = self.first_gpu_id + i
                print('CUDA DEVICES-{} enabled'.format(gpu_id))
                os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(gpu_id)

    def eval(self, particle):
        """
        evaluate the particle

        :param particle: particle
        :type particle: Particle
        :return:
        """
        logging.info('===start evaluating Particle-%d===', particle.id)
        tf.reset_default_graph()
        is_training, train_op, accuracy, cross_entropy, num_connections, merge_summary, regularization_loss, X, true_Y = self.build_graph(particle)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            steps_in_each_epoch = (self.training_data_length // self.batch_size)
            total_steps = int(self.training_epoch * steps_in_each_epoch)
            coord = tf.train.Coordinator()
            # threads = tf.train.start_queue_runners(sess, coord)
            try:
                threads = []
                for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                    threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))
                for i in range(total_steps):
                    if coord.should_stop():
                        break
                    _, accuracy_str, loss_str, regularization_loss_str,  _ = sess.run(
                        [train_op, accuracy, cross_entropy, regularization_loss, merge_summary],
                        {is_training: 0}
                    )
                    if i % (2 * steps_in_each_epoch) == 0:
                        mean_validation_accu, mean_validation_loss, _ = self.test_one_epoch(sess, accuracy, cross_entropy,
                                                                                         is_training,
                                                                                         self.validation_data_length, 1, X, true_Y)
                        logging.debug('{}, {}, indi:{}, Step:{}/{}, ce_loss:{}, reg_loss:{}, acc:{}, validation_ce_loss:{}, acc:{}'.format(
                            datetime.now(), i // steps_in_each_epoch, particle.id, i, total_steps, loss_str, regularization_loss_str,
                            accuracy_str, mean_validation_loss, mean_validation_accu))
                        if self.optimise and self.test_data is not None:
                            mean_test_accu, mean_test_loss, _ = self.test_one_epoch(sess, accuracy,
                                                                                 cross_entropy, is_training,
                                                                                 self.test_data_length,
                                                                                 2, X, true_Y)
                            logging.debug('test_ce_loss:{}, acc:{}'.format(mean_test_loss, mean_test_accu))
                # validate the last epoch
                mean_validation_accu, mean_validation_loss, stddev_validation_acccu = self.test_one_epoch(sess, accuracy, cross_entropy,
                                                                                 is_training,
                                                                                 self.validation_data_length, 1, X, true_Y)
                logging.debug('{}, validation_loss:{}, acc:{}'.format(datetime.now(), mean_validation_loss, mean_validation_accu))
                if self.optimise and self.test_data is not None:
                    mean_test_accu, mean_test_loss, _ = self.test_one_epoch(sess, accuracy,
                                                                         cross_entropy, is_training,
                                                                         self.test_data_length,
                                                                         2, X, true_Y)
                    logging.debug('test_ce_loss:{}, acc:{}'.format(mean_test_loss, mean_test_accu))

            except Exception as e:
                print(e)
                coord.request_stop(e)
            finally:
                logging.debug('finally...')
                coord.request_stop()
                coord.join(threads)
            logging.info('fitness of the particle: mean accuracy - %f, standard deviation of accuracy - %f, # of connections - %d', mean_validation_accu, stddev_validation_acccu, num_connections)
            logging.info('===finish evaluating Particle-%d===', particle.id)
            return mean_validation_accu, stddev_validation_acccu, num_connections

    def test_one_epoch(self, sess, accuracy, cross_entropy, is_training, data_length, training_mode, X, true_Y):
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
        total_step = data_length // self.batch_size
        accuracy_list = []
        loss_list = []
        for _ in range(total_step):
            accuracy_str, loss_str = sess.run([accuracy, cross_entropy, X, true_Y], {is_training: training_mode})
            logging.debug('is_training: {}, X shape: {}, true_Y shape: {}'.format(training_mode, str(X.shape), str(true_Y.shape)))
            accuracy_list.append(accuracy_str)
            loss_list.append(loss_str)
        mean_accu = np.mean(accuracy_list)
        mean_loss = np.mean(loss_list)
        stddev_accu = np.std(accuracy_list)
        return mean_accu, mean_loss, stddev_accu

    def build_graph(self, particle):
        """
        evaluate the particle

        :param particle: particle
        :type particle: Particle
        :return:
        """
        is_training = tf.placeholder(tf.int8, [])
        training_data, training_label = produce_tf_batch_data(self.training_data, self.training_label, self.batch_size)
        validation_data, validation_label = produce_tf_batch_data(self.validation_data, self.validation_label, self.batch_size)
        test_data, test_label = produce_tf_batch_data(self.test_data, self.test_label, self.batch_size)
        X, y_ = training_data, training_label
        bool_is_training = True
        if is_training == 1:
            X, y_ = (validation_data, validation_label)
            bool_is_training = False
        elif is_training == 2:
            X, y_ = (test_data, test_label)
            bool_is_training = False
        true_Y = tf.cast(y_, tf.int64)

        name_preffix = 'I_{}'.format(particle.id)
        output_list = []
        output_list.append(X)
        num_connections = 0

        regulariser = None
        if self.regularise is not None and self.regularise > 0:
            regulariser = slim.l2_regularizer(self.regularise)

        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                activation_fn=tf.nn.crelu,
                normalizer_fn=slim.batch_norm,
                weights_regularizer=regulariser,
                normalizer_params={'is_training': bool_is_training, 'decay': 0.99}
                            ):
            i = 0
            for interface in particle.x:
                # conv layer
                field_values = self.decoder.decode_2_field_values(interface)
                if particle.layers['conv'].check_interface_in_type(interface):
                    name_scope = '{}_conv_{}'.format(name_preffix, i)
                    with tf.variable_scope(name_scope):
                        filter_size, mean, stddev, feature_map_size, stride_size = self.decoder.filter_conv_fields(field_values)
                        conv_H = slim.conv2d(output_list[-1], feature_map_size, filter_size, stride_size,
                                             weights_initializer=tf.truncated_normal_initializer(mean=mean, stddev=stddev),
                                             biases_initializer=init_ops.constant_initializer(0.1, dtype=tf.float32))
                        output_list.append(conv_H)
                        # update for next usage
                        last_output_feature_map_size = feature_map_size
                        num_connections += feature_map_size * stride_size ^ 2 + feature_map_size
                # pooling layer
                elif particle.layers['pooling'].check_interface_in_type(interface):
                    name_scope = '{}_pooling_{}'.format(name_preffix, i)
                    with tf.variable_scope(name_scope):
                        kernel_size, stride_size, kernel_type = self.decoder.filter_pooling_fields(field_values)
                        if kernel_type == 0:
                            pool_H = slim.max_pool2d(output_list[-1], kernel_size=kernel_size, stride=stride_size,
                                                     padding='SAME')
                        else:
                            pool_H = slim.avg_pool2d(output_list[-1], kernel_size=kernel_size, stride=stride_size,
                                                     padding='SAME')
                        output_list.append(pool_H)
                        # pooling operation does not change the number of channel size, but channge the output size
                        last_output_feature_map_size = last_output_feature_map_size
                        num_connections += last_output_feature_map_size
                # fully-connected layer
                elif particle.layers['full'].check_interface_in_type(interface):
                    name_scope = '{}_fully-connected_{}'.format(name_preffix, i)
                    with tf.variable_scope(name_scope):
                        last_interface = particle.x[i-1]
                        if not particle.layers['full'].check_interface_in_type(last_interface):  # use the previous setting to calculate this input dimension
                            input_data = slim.flatten(output_list[-1])
                            input_dim = input_data.get_shape()[1].value
                        else:  # current input dim should be the number of neurons in the previous hidden layer
                            input_data = output_list[-1]
                            last_filed_values = self.decoder.decode_2_field_values(last_interface)
                            _, _, input_dim = last_filed_values['num_of_neurons'] + 1

                        mean, stddev, hidden_neuron_num = self.decoder.filter_full_fields(field_values)
                        if i < particle.length - 1:
                            full_H = slim.fully_connected(input_data, num_outputs=hidden_neuron_num,
                                                          weights_initializer=tf.truncated_normal_initializer(mean=mean,
                                                                                                              stddev=stddev),
                                                          biases_initializer=init_ops.constant_initializer(0.1,
                                                                                                           dtype=tf.float32))
                        else:
                            # hard-code the number of units of the last layer to 10 for now
                            full_H = slim.fully_connected(input_data, num_outputs=self.class_num,
                                                          activation_fn=None,
                                                          weights_initializer=tf.truncated_normal_initializer(mean=mean,
                                                                                                              stddev=stddev),
                                                          biases_initializer=init_ops.constant_initializer(0.1,
                                                                                                           dtype=tf.float32))
                        # add dropout
                        if self.dropout is not None and self.dropout > 0:
                            full_dropout_H = slim.dropout(full_H, self.dropout)
                            logging.debug('dropout rate: {}'.format(self.dropout))
                        else:
                            full_dropout_H = full_H
                        output_list.append(full_dropout_H)
                        num_connections += input_dim * hidden_neuron_num + hidden_neuron_num
                # disabled layer
                elif particle.layers['disabled'].check_interface_in_type(interface):
                    name_scope = '{}_disabled_{}'.format(name_preffix, i)
                else:
                    logging.error('Invalid Interface: %s', str(interface))
                    raise Exception('invalid interface')
                i = i+1

            with tf.name_scope('loss'):
                logits = output_list[-1]
                if self.regularise is not None and self.regularise > 0:
                    regularization_loss = tf.add_n(tf.losses.get_regularization_losses())
                else:
                    regularization_loss = tf.constant(0.0)
                cross_entropy = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_Y, logits=logits))
                loss = regularization_loss + cross_entropy
            with tf.name_scope('train'):
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                if update_ops:
                    updates = tf.group(*update_ops)
                    loss = control_flow_ops.with_dependencies([updates], loss)
                optimizer = tf.train.AdamOptimizer()
                train_op = slim.learning.create_train_op(loss, optimizer)
            with tf.name_scope('test'):
                accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), true_Y), tf.float32))

        tf.summary.scalar('ce_loss', cross_entropy)
        tf.summary.scalar('reg_loss', regularization_loss)
        tf.summary.scalar('accuracy', accuracy)
        merge_summary = tf.summary.merge_all()

        return is_training, train_op, accuracy, cross_entropy, num_connections, merge_summary, regularization_loss, X, true_Y

