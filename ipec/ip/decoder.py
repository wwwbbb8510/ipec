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

    def __init__(self, mean_centre=None, mean_divisor=None, stddev_divisor=None):
        """
        constructor
        """
        self.mean_centre = 255 if mean_centre is None else mean_centre
        self.mean_divisor = 2560 if mean_divisor is None else mean_divisor
        self.stddev_divisor = 512 if stddev_divisor is None else stddev_divisor

    def decode_2_field_values(self, interface):
        """
        decode an interface into a list of values and their corresponding fields

        :param interface: an interface including ip and subnet
        :type interface: Interface
        :return: (filed, value) dict
        :rtype: dict
        """
        logging.debug('The interface to be decoded: %s', str(interface))
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

        logging.debug('The fields decoded from the interface: %s', str(field_values))

        return field_values

    def filter_conv_fields(self, field_values):
        """
        filter filed values and convert them to proper attributes of conv layer

        :param field_values:
        :return: filter_size, mean, stddev, feature_map_size, stride_size
        :rtype: tuple
        """
        filter_size = field_values['filter_size'] + 1
        mean, stddev = self._normalise_mean_stddev(field_values['mean'], field_values[
            'std_dev']) if 'mean' in field_values.keys() and 'std_dev' in field_values.keys() else (None, None)
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
        mean, stddev = self._normalise_mean_stddev(field_values['mean'], field_values[
            'std_dev']) if 'mean' in field_values.keys() and 'std_dev' in field_values.keys() else (None, None)
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
        mean = (mean - self.mean_centre) / self.mean_divisor
        stddev = (stddev + 1) / self.stddev_divisor
        return mean, stddev
