from .core import IPStructure
from .core import Subnet
from .core import Interface

class Decoder:
    """
    Decoder clss
    """

    def __init__(self, ipStructure, subnet):
        """
        constructor

        :param ipStructure: ip structure containing the fields and their # of bits
        :type ipStructure: IPStructure
        :param subnet: subnet of the specific encoder, e.g. Conv Layer encoder, Pooling Layer encoder
        :type subnet: Subnet
        """
        self.ipStruture = ipStructure
        self.subnet = subnet

    def decode(self, interface):
        """
        decode an interface into a list of values and their corresponding fields

        :param interface: an interface including ip and subnet
        :type interface: Interface
        :return: a list of (field, value) pairs
        :type return: list
        """
