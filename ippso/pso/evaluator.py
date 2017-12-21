import numpy as np
import logging
from datetime import datetime
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import init_ops
from ippso.ip.decoder import Decoder
from .particle import Particle

def initialise_cnn_evaluator(training_epoch=None, batch_size=None, training_data=None, training_label=None, validation_data=None,
                             validation_label=None):
    training_epoch = 5 if training_epoch is None else training_epoch
    batch_size = 200 if batch_size is None else batch_size
    if training_data is None and training_label is None and validation_data is None and validation_label is None:
        from ippso.data.mnist import get_training_data, get_validation_data
        training_data = get_training_data()['images'] if training_data is None else training_data
        training_label = get_training_data()['labels'] if training_label is None else training_label
        validation_data = get_validation_data()['images'] if validation_data is None else validation_data
        validation_label = get_validation_data()['labels'] if validation_label is None else validation_label
    return CNNEvaluator(training_epoch, batch_size, training_data, training_label, validation_data, validation_label)


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
    def __init__(self, training_epoch, batch_size, training_data, training_label, validation_data, validation_label):
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
        """
        self.training_epoch = training_epoch
        self.batch_size = batch_size
        self.training_data = training_data
        self.training_label = training_label
        self.validation_data = validation_data
        self.validation_label = validation_label

        self.training_data_length = self.training_data.shape[0]
        self.validation_data_length = self.validation_data.shape[0]
        self.decoder = Decoder()

    def eval(self, particle):
        """
        evaluate the particle

        :param particle: particle
        :type particle: Particle
        :return:
        """
        logging.info('===start evaluating Particle-%d===', particle.id)
        tf.reset_default_graph()
        is_training, train_op, accuracy, cross_entropy, num_connections, merge_summary = self.build_graph(particle)
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
                    _, accuracy_str, loss_str, _ = sess.run([train_op, accuracy, cross_entropy, merge_summary],
                                                            {is_training: True})
                    if i % (2 * steps_in_each_epoch) == 0:
                        test_total_step = self.validation_data_length // self.batch_size
                        test_accuracy_list = []
                        test_loss_list = []
                        for _ in range(test_total_step):
                            test_accuracy_str, test_loss_str = sess.run([accuracy, cross_entropy], {is_training: False})
                            test_accuracy_list.append(test_accuracy_str)
                            test_loss_list.append(test_loss_str)
                        mean_test_accu = np.mean(test_accuracy_list)
                        mean_test_loss = np.mean(test_loss_list)
                        logging.debug('{}, {}, indi:{}, Step:{}/{}, train_loss:{}, acc:{}, test_loss:{}, acc:{}'.format(
                            datetime.now(), i // steps_in_each_epoch, particle.id, i, total_steps, loss_str,
                            accuracy_str, mean_test_loss, mean_test_accu))
                # validate the last epoch
                test_total_step = self.validation_data_length // self.batch_size
                test_accuracy_list = []
                test_loss_list = []
                for _ in range(test_total_step):
                    test_accuracy_str, test_loss_str = sess.run([accuracy, cross_entropy], {is_training: False})
                    test_accuracy_list.append(test_accuracy_str)
                    test_loss_list.append(test_loss_str)
                mean_test_accu = np.mean(test_accuracy_list)
                mean_test_loss = np.mean(test_loss_list)
                logging.debug('{}, test_loss:{}, acc:{}'.format(datetime.now(), mean_test_loss, mean_test_accu))
                mean_acc = mean_test_accu

            except Exception as e:
                print(e)
                coord.request_stop(e)
            finally:
                logging.debug('finally...')
                coord.request_stop()
                coord.join(threads)
            logging.info('fitness of the particle: mean accuracy - %f, standard deviation of accuracy - %f, # of connections - %d', mean_test_accu, np.std(test_accuracy_list), num_connections)
            logging.info('===finish evaluating Particle-%d===', particle.id)
            return mean_test_accu, np.std(test_accuracy_list), num_connections

    def build_graph(self, particle):
        """
        evaluate the particle

        :param particle: particle
        :type particle: Particle
        :return:
        """
        is_training = tf.placeholder(tf.bool, [])
        training_data, training_label = produce_tf_batch_data(self.training_data, self.training_label, self.batch_size)
        validation_data, validation_label = produce_tf_batch_data(self.validation_data, self.validation_label, self.batch_size)
        X = tf.cond(is_training, lambda: training_data, lambda: validation_data)
        y_ = tf.cond(is_training, lambda: training_label, lambda: validation_label)
        true_Y = tf.cast(y_, tf.int64)

        name_preffix = 'I_{}'.format(particle.id)
        output_list = []
        output_list.append(X)
        num_connections = 0

        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                activation_fn=tf.nn.crelu,
                normalizer_fn=slim.batch_norm,
                # weights_regularizer=slim.l2_regularizer(0.005),
                normalizer_params={'is_training': is_training, 'decay': 0.99}):
            i = 0
            for interface in particle.x:
                # conv layer
                field_values = self.decoder.decode_2_field_values(interface)
                if particle.layers['conv'].check_interface_in_type(interface):
                    name_scope = '{}_conv_{}'.format(name_preffix, i)
                    with tf.variable_scope(name_scope):
                        filter_size, mean, stddev, feature_map_size, stride_size = self.decoder.filter_conv_fields(field_values)
                        conv_H = slim.conv2d(output_list[-1], feature_map_size, filter_size,
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
                            full_H = slim.fully_connected(input_data, num_outputs=10,
                                                          activation_fn=None,
                                                          weights_initializer=tf.truncated_normal_initializer(mean=mean,
                                                                                                              stddev=stddev),
                                                          biases_initializer=init_ops.constant_initializer(0.1,
                                                                                                           dtype=tf.float32))
                        output_list.append(full_H)
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
                # regularization_loss = tf.add_n(tf.losses.get_regularization_losses())
                cross_entropy = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_Y, logits=logits))
            with tf.name_scope('train'):
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                if update_ops:
                    updates = tf.group(*update_ops)
                    cross_entropy = control_flow_ops.with_dependencies([updates], cross_entropy)
                optimizer = tf.train.AdamOptimizer()
                train_op = slim.learning.create_train_op(cross_entropy, optimizer)
            with tf.name_scope('test'):
                accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), true_Y), tf.float32))

        tf.summary.scalar('loss', cross_entropy)
        tf.summary.scalar('accuracy', accuracy)
        merge_summary = tf.summary.merge_all()

        return is_training, train_op, accuracy, cross_entropy, num_connections, merge_summary

