import numpy as np


class IPStructure:
    """
    IPStructure class (fields and its storage bits)
    which will be used for encoding and decoding
    """

    def __init__(self, fields):
        """
        constructor

        :param fields: fields encoded in the IP - a list of (name, #bits) pairs
        :type fields: list
        """
        self.fields = fields


class Interface:
    """
    Interface class
    which will carry both IP and subnet information
    """

    def __init__(self, ip, subnet):
        """
        constructor

        :param ip: IP address object
        :type ip: IPAddress
        :param subnet: subnet object
        :type subnet: Subnet
        """
        self.ip = ip
        self.subnet = subnet


class IPAddress:
    """
    IP Address class
    """

    def __init__(self, length=5, ip=None, bin_ip = None):
        """
        constructor

        :param length: IP length(the number of bytes) e.g. the length of 192.168.1.1 is 4
        :type length: int
        :param ip: decimal IP address carried by a numpy array
        :type ip: numpy.array
        :param bin_ip: binary string of the IP
        :type bin_ip: string
        """
        self.length = length
        if ip == None and bin_ip == None:
            self.ip = np.zeros(length, dtype=np.uint8)
            self.bin_ip = self._ip_2_bin_ip()
        elif ip is not None:
            self.ip = ip
            self.bin_ip = self._ip_2_bin_ip()
        elif bin_ip is not None:
            self.bin_ip = bin_ip
            self.ip = self._bin_ip_2_ip()


    def _ip_2_bin_ip(self, ip=None):
        """
        convert IP from decimal IP to binary string

        :param ip: decimal IP in a numpy array
        :type ip: numpy.array
        :return: a binary string of the IP address
        :type return: string
        """
        if ip is None:
            ip = self.ip
        bin_ip = ''
        for single_byte in np.nditer(ip):
            bin_ip += np.binary_repr(single_byte)
        return bin_ip


    def _bin_ip_2_ip(self, bin_ip=None):
        """
        convert a binary string of  IP to decimal IP in a numpy array

        :param bin_ip: binary string of an IP
        :type bin_ip: string
        :return: a decimal IP address in numpy array
        :type return: numpy.array
        """
        if bin_ip is None:
            bin_ip = self.bin_ip
        ip = np.zeros(self.length, dtype=np.uint8)
        for i in range(self.length):
            ip[i] = int(bin_ip[i*8: (i+1)*8], base=2)
        return ip


class SubnetMask:
    """
    Subnet mask class
    """

    def __init__(self, mask_length, ip_length=40):
        """
        constructor

        :param mask_length: the length(number of bits) of the subnet mask, e.g. the subnet length of 255.255.255.0 is 3*8=24
        :type mask_length: int
        :param ip_length: the length(number of bits) of the IP, e.g. the length of 192.168.1.1 is 4*8=32
        """
        self.ip_length = ip_length
        self.mask_length = mask_length
        self.mask_binary = self._mask_binary()
        self.mask_decimal = self.mask_decimal()

    def _mask_binary(self):
        """
        generate the binary mask given the mask length and the ip length

        :return: a binary mask
        :type return: string
        """
        mask_binary = ''
        for i in range(self.ip_length):
            if i < self.mask_length:
                mask_binary += '1'
            else:
                mask_binary += '0'
        return mask_binary

    def _mask_decimal(self, mask_binary=None):
        """
        generate the decimal mask

        :param mask_binary: binary mask
        :type mask_binary: string
        :return: the decimal mask in numpy array
        :type return: numpy.array
        """
        if mask_binary is None:
            mask_binary = self.mask_binary
        decimal_length = self.ip_length / 8
        mask_decimal = np.zeros(decimal_length, dtype=np.uint8)
        for i in range(decimal_length):
            mask_decimal[i] = int(mask_binary[i*8: (i+1)*8], base=2)
        return mask_decimal


class Subnet:
    """
    Subnet class
    """

    def __init__(self, ip, mask):
        """
        constructor
        :param ip: IP address object
        :type ip: IPAddress
        :param mask: subnet mask object
        :type mask: SubnetMask
        """
        self.ip = ip
        self.mask = mask
