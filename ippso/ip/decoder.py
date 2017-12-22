from .core import IPStructure
from .core import Subnet
from .core import Interface
from .core import ip_2_bin_ip

import numpy as np

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
        mean = (field_values['mean'] - 256) / 100
        stddev = (field_values['std_dev'] + 1) / 100
        feature_map_size = field_values['num_of_feature_maps'] + 1
        stride_size = field_values['stride_size'] + 1
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
        return kernel_size, stride_size, kernel_type

    def filter_full_fields(self, field_values):
        """
        filter filed values and convert them to proper attributes of fully-connected layer

        :param field_values:
        :return: mean, stddev, hidden_neuron_num
        :rtype: tuple
        """
        mean = (field_values['mean'] - 256) / 100
        stddev = (field_values['std_dev'] + 1) / 100
        hidden_neuron_num = field_values['num_of_neurons'] + 1
        return mean, stddev, hidden_neuron_num

