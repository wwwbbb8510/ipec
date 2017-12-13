from .core import IPStructure
from .core import Subnet
from .core import Interface
from .core import IPAddress
from .core import bin_add

import numpy as np

class Encoder:
    """
    Encoder class
    """

    def __init__(self, ip_structure, subnet):
        """
        constructor

        :param ip_structure: ip structure containing the fields and their # of bits
        :type ip_structure: IPStructure
        :param subnet: subnet of the specific encoder, e.g. Conv Layer encoder, Pooling Layer encoder
        :type subnet: Subnet
        """
        self.ip_structure = ip_structure
        self.subnet = subnet

    def encode_2_interface(self, field_values):
        """
        encode the values of a list of fields into an interface including the IP and subnet

        :param field_values: (filed, value) dict
        :type field_values: dict
        :return: an interface including the IP and the subnet
        :rtype: Interface
        """
        bin_ip = ''
        fields = self.ip_structure.fields
        for field_name in fields:
            try:
                v = field_values[field_name]
                num_of_bits = fields[field_name]
                v_bin = np.binary_repr(v)
                if len(v_bin) > num_of_bits:
                    raise Exception('field value is out of the allowed bound')
                v_bin = v_bin.zfill(num_of_bits)
                bin_ip += v_bin

            except KeyError as e:
                raise Exception('fields and field_values does not match')
            except Exception as e:
                raise e

        subnet_bin_ip = self.subnet.bin_ip
        bin_ip = bin_add(bin_ip, subnet_bin_ip)
        ip_addr = IPAddress(length=bin_ip/8, bin_ip=bin_ip)
        interface = Interface(ip=ip_addr, subnet=self.subnet, ip_structure=self.ip_structure)
        return interface
