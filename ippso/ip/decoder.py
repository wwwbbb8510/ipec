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
        ip = interface.ip.ip
        subnet_ip = interface.subnet.ip
        field_ip = np.subtract(ip, subnet_ip)
        field_bin_ip = ip_2_bin_ip(field_ip)
        pos = 0
        field_values = {}
        for field_name in fields:
            num_of_bits = fields[field_name]
            field_values[field_name] = int(field_bin_ip[pos:pos+num_of_bits], base=2)
            pos + num_of_bits

        return field_values
