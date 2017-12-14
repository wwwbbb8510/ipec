import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import init_ops
from ippso.ip.decoder import Decoder

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
        :param batch_size:
        """
        self.training_epoch = training_epoch
        self.batch_size = batch_size
        self.training_data = training_data
        self.training_label = training_label
        self.validation_data = validation_data
        self.validation_label = validation_label
        self.decoder = Decoder()

    def eval(self, particle):
        """
        evaluate the particle

        :param particle:
        :type particle:
        :return:
        """

    def build_graph(self, particle):
        """
        evaluate the particle

        :param particle:
        :type particle:
        :return:
        """
        is_training = tf.placeholder(tf.bool, [])
        X = tf.cond(is_training, lambda: self.training_data, lambda: self.validation_data)
        y_ = tf.cond(is_training, lambda: self.training_label, lambda: self.validation_label)
        true_Y = tf.cast(y_, tf.int64)

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
                    name_scope = 'conv_{}'.format(i)
                    with tf.variable_scope(name_scope):
                        filter_size = field_values['filter_size']
                        mean = field_values['mean']
                        stddev = field_values['std_dev']
                        feature_map_size = field_values['num_of_feature_maps']
                        stride_size = field_values['stride_size']
                        conv_H = slim.conv2d(output_list[-1], feature_map_size, filter_size,
                                             weights_initializer=tf.truncated_normal_initializer(mean=mean, stddev=stddev),
                                             biases_initializer=init_ops.constant_initializer(0.1, dtype=tf.float32))
                        output_list.append(conv_H)
                        # update for next usage
                        last_output_feature_map_size = feature_map_size
                        num_connections += feature_map_size * stride_size ^ 2 + feature_map_size
                # pooling layer
                elif particle.layers['pooling'].check_interface_in_type(interface):
                    name_scope = 'pooling_{}'.format(i)
                    with tf.variable_scope(name_scope):
                        kernel_size = field_values['kernel_size']
                        stride_size = field_values['stride_size']
                        kernel_type = field_values['type']
                        if kernel_type == 0:
                            pool_H = slim.max_pool2d(output_list[-1], kernel_size=kernel_size, stride=kernel_size,
                                                     padding='SAME')
                        else:
                            pool_H = slim.avg_pool2d(output_list[-1], kernel_size=kernel_size, stride=kernel_size,
                                                     padding='SAME')
                        output_list.append(pool_H)
                        # pooling operation does not change the number of channel size, but channge the output size
                        last_output_feature_map_size = last_output_feature_map_size
                        num_connections += last_output_feature_map_size
                # fully-connected layer
                elif particle.layers['full'].check_interface_in_type(interface):
                    name_scope = 'fully-connected_{}'.format(i)
                    with tf.variable_scope(name_scope):
                        last_interface = particle.x[i-1]
                        if not particle.layers['full'].check_interface_in_type(last_interface):  # use the previous setting to calculate this input dimension
                            input_data = slim.flatten(output_list[-1])
                            input_dim = input_data.get_shape()[1].value
                        else:  # current input dim should be the number of neurons in the previous hidden layer
                            input_data = output_list[-1]
                            last_filed_values = self.decoder.decode_2_field_values(last_interface)
                            input_dim = last_filed_values['num_of_neurons']
                        mean = field_values['mean']
                        stddev = field_values['std_dev']
                        hidden_neuron_num = field_values['num_of_neurons']
                        if i < interface.ip.length - 1:
                            full_H = slim.fully_connected(input_data, num_outputs=hidden_neuron_num,
                                                          weights_initializer=tf.truncated_normal_initializer(mean=mean,
                                                                                                              stddev=stddev),
                                                          biases_initializer=init_ops.constant_initializer(0.1,
                                                                                                           dtype=tf.float32))
                        else:
                            full_H = slim.fully_connected(input_data, num_outputs=hidden_neuron_num,
                                                          activation_fn=None,
                                                          weights_initializer=tf.truncated_normal_initializer(mean=mean,
                                                                                                              stddev=stddev),
                                                          biases_initializer=init_ops.constant_initializer(0.1,
                                                                                                           dtype=tf.float32))
                        output_list.append(full_H)
                        num_connections += input_dim * hidden_neuron_num + hidden_neuron_num
                # disabled layer
                elif particle.layers['disabled'].check_interface_in_type(interface):
                    name_scope = 'disabled_{}'.format(i)
                else:
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
                    # global_step = tf.get_variable("global_step", [], initializer=tf.constant_initializer(0.0), trainable=False)
                    # self.train_data_length//self.batch_size
                    #                 lr = tf.train.exponential_decay(0.1, step, 550*30, 0.9, staircase=True)
                    #                 optimizer = tf.train.GradientDescentOptimizer(lr)
                optimizer = tf.train.AdamOptimizer()
                train_op = slim.learning.create_train_op(cross_entropy, optimizer)
            with tf.name_scope('test'):
                accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), true_Y), tf.float32))

            tf.summary.scalar('loss', cross_entropy)
            tf.summary.scalar('accuracy', accuracy)
            merge_summary = tf.summary.merge_all()

            return is_training, train_op, accuracy, cross_entropy, num_connections, merge_summary

