import numpy as np

from ippso.ip.core import parse_subnet_str
from ippso.ip.core import IPStructure
from ippso.ip.core import Interface
from ippso.ip.encoder import Encoder
from ippso.ip.decoder import Decoder
from ippso.ip.core import max_decimal_value_of_binary

# convolutional layer fields
CONV_FIELDS = {
    'filter_size': 5,
    'num_of_feature_maps': 7,
    'stride_size': 4,
    'mean': 9,
    'std_dev': 9
}

# convolutional layer subnet
CONV_SUBNET = '0.0.0.0.0/6'

# pooling layer fields
POOLING_FIELDS = {
    'kernel_size': 5,
    'stride_size': 4,
    'type': 1
}

# pooling layer subnet
POOLING_SUBNET = '4.32.0.0.0/30'

# fully-connected layer fields
FULLYCONNECTED_FIELDS = {
    'num_of_neurons': 11,
    'mean': 9,
    'std_dev': 9
}

# fully-connected layer subnet
FULLYCONNECTED_SUBNET = '4.0.0.0.0/11'

# disabled layer fields
DISABLED_FIELDS = {
    'disabled': 10,
}

# disabled layer subnet
DISABLED_SUBNET = '4.32.0.4.0/30'


def initialise_cnn_layers_3_bytes():
    # convolutional layer fields
    conv_fields = {
        'filter_size': 3, #8
        'num_of_feature_maps': 7, #128
        'stride_size': 2, #4
        'mean': 4, #(0~15-7)/8
        'std_dev': 4 # 0~16/16
        #total bits: 20
    }

    # convolutional layer subnet
    conv_subnet = '0.0.0/4'

    # pooling layer fields
    pooling_fields = {
        'kernel_size': 2,
        'stride_size': 2,
        'type': 1,
        'placeholder': 14
        # total bits: 19
    }

    # pooling layer subnet
    pooling_subnet = '16.0.0/5'

    # fully-connected layer fields
    fullyconnected_fields = {
        'num_of_neurons': 11,
        'mean': 4,
        'std_dev': 4
        # total bits: 19
    }

    # fully-connected layer subnet
    fullyconnected_subnet = '24.0.0/5'

    # disabled layer fields
    disabled_fields = {
        'disabled': 19,
    }

    # disabled layer subnet
    disabled_subnet = '32.0.0/5'
    return {
        'conv': ConvLayer(conv_subnet,conv_fields),
        'pooling': PoolingLayer(pooling_subnet, pooling_fields),
        'full': FullyConnectedLayer(fullyconnected_subnet, fullyconnected_fields),
        'disabled': DisabledLayer(disabled_subnet, disabled_fields)
    }

class BaseCNNLayer:
    """
    BaseCNNLayer class
    """
    def __init__(self, str_subnet, fields):
        """
        constructor

        :param str_subnet: subnet string, e.g. 127.0.0.1/24
        :type str_subnet: string
        :param fields: a dict of (field_name, num_of_bits) pair
        :type fields: dict
        """
        self.str_subnet = str_subnet
        self.fields = fields
        self.subnet = parse_subnet_str(str_subnet)
        self.ip_structure = IPStructure(fields)
        self.encoder = Encoder(self.ip_structure, self.subnet)
        self.decoder = Decoder()

    def encode_2_interface(self, field_values):
        """
        encode filed values to an IP interface

        :param field_values: field values
        :type field_values: a dict of (field_name, field_value) pairs
        :return: the layer interface
        :rtype: Interface
        """
        interface = self.encoder.encode_2_interface(field_values)
        return interface

    def decode_2_field_values(self, interface):
        """
        decode an IP interface to field values

        :param interface: an IP interface
        :type interface: Interface
        :return: a dict of (field_name, field_value) pairs
        :rtype: dict
        """
        field_values = self.decoder.decode_2_field_values(interface)
        return field_values

    def generate_random_interface(self):
        """
        generate an IP interface with random settings

        :rtype: Interface
        :return: an IP interface
        """
        field_values = {}
        for field_name in self.fields:
            num_of_bits = self.fields[field_name]
            max_value = max_decimal_value_of_binary(num_of_bits)
            rand_value = np.random.randint(0, max_value+1)
            field_values[field_name] = rand_value
        return self.encode_2_interface(field_values)

    def check_interface_in_type(self, interface):
        """
        check whether the interface belongs to this type

        :param interface: an IP interface
        :type interface: Interface
        :return: boolean
        :rtype: bool
        """
        return self.subnet.check_ip_in_subnet(interface.ip)


class ConvLayer(BaseCNNLayer):
    """
    ConvLayer class
    """
    def __init__(self, str_subnet=None, fields=None):
        """
        constructor

        :param str_subnet: subnet string, e.g. 127.0.0.1/24
        :type str_subnet: string
        :param fields: a dict of (field_name, num_of_bits) pair
        :type fields: dict
        """
        if str_subnet is None:
            str_subnet = CONV_SUBNET
        if fields is None:
            fields = CONV_FIELDS
        super(ConvLayer, self).__init__(str_subnet, fields)


class PoolingLayer(BaseCNNLayer):
    """
    PoolingLayer class
    """
    def __init__(self, str_subnet=None, fields=None):
        """
        constructor

        :param str_subnet: subnet string, e.g. 127.0.0.1/24
        :type str_subnet: string
        :param fields: a dict of (field_name, num_of_bits) pair
        :type fields: dict
        """
        if str_subnet is None:
            str_subnet = POOLING_SUBNET
        if fields is None:
            fields = POOLING_FIELDS
        super(PoolingLayer, self).__init__(str_subnet, fields)


class FullyConnectedLayer(BaseCNNLayer):
    """
    FullyConnectedLayer class
    """
    def __init__(self, str_subnet=None, fields=None):
        """
        constructor

        :param str_subnet: subnet string, e.g. 127.0.0.1/24
        :type str_subnet: string
        :param fields: a dict of (field_name, num_of_bits) pair
        :type fields: dict
        """
        if str_subnet is None:
            str_subnet = FULLYCONNECTED_SUBNET
        if fields is None:
            fields = FULLYCONNECTED_FIELDS
        super(FullyConnectedLayer, self).__init__(str_subnet, fields)


class DisabledLayer(BaseCNNLayer):
    """
    DisabledLayer class
    """
    def __init__(self, str_subnet=None, fields=None):
        """
        constructor

        :param str_subnet: subnet string, e.g. 127.0.0.1/24
        :type str_subnet: string
        :param fields: a dict of (field_name, num_of_bits) pair
        :type fields: dict
        """
        if str_subnet is None:
            str_subnet = DISABLED_SUBNET
        if fields is None:
            fields = DISABLED_FIELDS
        super(DisabledLayer, self).__init__(str_subnet, fields)
