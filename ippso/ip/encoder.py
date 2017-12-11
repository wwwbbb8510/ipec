from .core import IPStructure
from .core import Subnet
from .core import Interface

class Encoder:
    """
    Encoder clss
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

    def encode(self, fieldValues):
        """
        encode the values of a list of fields into an interface including the IP and subnet

        :param fieldValues: a list of (filed, value) pairs
        :type fieldValues: list
        :return: an interface including the IP and the subnet
        :type return: Interface
        """
        