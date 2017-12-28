from .core import IPStructure
from .core import Subnet
from .core import Interface
from .core import ip_2_bin_ip

import numpy as np
import logging

class Decoder:
    """
    Decoder clss
    """

    def __init__(self):
        """
        constructor
        """

    def decode_2_field_values(self, interface):
        """
        decode an interface into a list of values and their corresponding fields

        :param interface: an interface including ip and subnet
        :type interface: Interface
        :return: (filed, value) dict
        :rtype: dict
        """
        fields = interface.ip_structure.fields
        fields_length = interface.ip_structure.fields_length
        ip = interface.ip.ip
        bin_ip_length = interface.ip.length * 8
        subnet_ip = interface.subnet.ip
        field_ip = np.subtract(ip, subnet_ip)
        field_bin_ip = ip_2_bin_ip(field_ip)
        field_bin_ip = field_bin_ip[bin_ip_length-fields_length:]
        pos = 0
        field_values = {}
        for field_name in fields:
            num_of_bits = fields[field_name]
            field_values[field_name] = int(field_bin_ip[pos:pos+num_of_bits], base=2)
            pos + num_of_bits

        return field_values

    def filter_conv_fields(self, field_values):
        """
        filter filed values and convert them to proper attributes of conv layer

        :param field_values:
        :return: filter_size, mean, stddev, feature_map_size, stride_size
        :rtype: tuple
        """
        filter_size = field_values['filter_size'] + 1
        mean, stddev = self._normalise_mean_stddev(field_values['mean'], field_values['std_dev'])
        feature_map_size = field_values['num_of_feature_maps'] + 1
        stride_size = field_values['stride_size'] + 1
        logging.debug('Filtered Conv field values(filter_size, mean, stddev, feature_map_size, stride_size):%s', str((filter_size, mean, stddev, feature_map_size, stride_size)))

        return filter_size, mean, stddev, feature_map_size, stride_size

    def filter_pooling_fields(self, field_values):
        """
        filter filed values and convert them to proper attributes of pooling layer

        :param field_values:
        :return: kernel_size, stride_size, kernel_type
        :rtype: tuple
        """
        kernel_size = field_values['kernel_size'] + 1
        stride_size = field_values['stride_size'] + 1
        kernel_type = field_values['type']
        logging.debug('Filtered Pooling field values(kernel_size, stride_size, kernel_type):%s', str((kernel_size, stride_size, kernel_type)))
        return kernel_size, stride_size, kernel_type

    def filter_full_fields(self, field_values):
        """
        filter filed values and convert them to proper attributes of fully-connected layer

        :param field_values:
        :return: mean, stddev, hidden_neuron_num
        :rtype: tuple
        """
        mean, stddev = self._normalise_mean_stddev(field_values['mean'], field_values['std_dev'])
        hidden_neuron_num = field_values['num_of_neurons'] + 1
        logging.debug('Filtered Fully Connected field values(mean, stddev, hidden_neuron_num):%s',
                      str((mean, stddev, hidden_neuron_num)))
        return mean, stddev, hidden_neuron_num

    def _normalise_mean_stddev(self, mean, stddev):
        """
        normalise mean and stddev from 512 to decimal value
        :param mean: IP value represent mean
        :type mean: int
        :param stddev: IP value represent stddev
        :type stddev: int
        :return: (mean, stddev) tuple
        :rtype: tuple
        """
        mean = (mean - 255) / 2560
        stddev = (stddev + 1) / 512
        return mean, stddev
