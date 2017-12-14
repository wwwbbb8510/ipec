import numpy as np


def bin_add(x, y):
    """
    binary addition x+y

    :param x: binary string x
    :type x: string
    :param y: binary string y
    :type y: string
    :return: added result binary string
    :rtype: string
    """
    maxlen = max(len(x), len(y))

    # Normalize lengths
    x = x.zfill(maxlen)
    y = y.zfill(maxlen)

    result = ''
    carry = 0

    for i in range(maxlen - 1, -1, -1):
        r = carry
        r += 1 if x[i] == '1' else 0
        r += 1 if y[i] == '1' else 0

        # r can be 0,1,2,3 (carry + x[i] + y[i])
        # and among these, for r==1 and r==3 you will have result bit = 1
        # for r==2 and r==3 you will have carry = 1

        result = ('1' if r % 2 == 1 else '0') + result
        carry = 0 if r < 2 else 1

    if carry != 0: result = '1' + result

    return result.zfill(maxlen)


def ip_2_bin_ip(ip):
    """
    convert IP from decimal IP to binary string

    :param ip: decimal IP in a numpy array
    :type ip: np.array
    :return: a binary string of the IP address
    :rtype: string
    """
    bin_ip = ''
    for single_byte in np.nditer(ip):
        bin_ip += np.binary_repr(single_byte)
    return bin_ip


def bin_ip_2_ip(bin_ip):
    """
    convert a binary string of  IP to decimal IP in a numpy array

    :param bin_ip: binary string of an IP
    :type bin_ip: string
    :return: a decimal IP address in numpy array
    :rtype: np.array
    """
    length = len(bin_ip) / 8
    ip = np.zeros(length, dtype=np.uint8)
    for i in range(length):
        ip[i] = int(bin_ip[i*8: (i+1)*8], base=2)
    return ip

def parse_subnet_str(subnet_str):
    """
    parse subnet string and generate subnet object

    :param subnet_str: subnet string
    :type subnet_str: string
    :return: subnet object
    :rtype: Subnet
    """
    arr_subnet = subnet_str.split('/')
    subnet_ip_str = arr_subnet[0]
    subnet_mask_length = arr_subnet[1]
    arr_subnet_ip = subnet_ip_str.split('.')
    subnet_ip = np.asarray(arr_subnet_ip, dtype=np.uint8)
    subnet_mask = SubnetMask(subnet_mask_length, len(subnet_ip) * 8)
    subnet = Subnet(subnet_ip, subnet_mask)
    return subnet

def max_decimal_value_of_binary(num_of_bits):
    """
    get max decimal value of a binary string with a fixed length

    :param num_of_bits: # of bits
    :type num_of_bits: int
    :return: max decimal value
    :rtype: int
    """
    return int('1'*num_of_bits, base=2)

def compare_2_ips(ip_1, ip_2):
    """
    compare two ips

    :param ip_1: the first ip address
    :type ip_1: numpy.array
    :param ip_2: the second ip address
    :type ip_2: numpy.array
    :return: -1: the first ip is greater than the second ip, 0: equal, 2: smaller
    :rtype: int
    """
    bin_ip_1 = ip_2_bin_ip(ip_1)
    bin_ip_2 = ip_2_bin_ip(ip_2)
    return compare_2_bin_ips(bin_ip_1, bin_ip_2)

def compare_2_bin_ips(bin_ip_1, bin_ip_2):
    """
        compare two binary ips

        :param bin_ip_1: the first binary ip address
        :type bin_ip_1: string
        :param bin_ip_2: the second binary ip address
        :type bin_ip_2: string
        :return: -1: the first ip is smaller than the second ip, 0: equal, 1: greater
        :rtype: int
        """
    if bin_ip_1 > bin_ip_2:
        flag = 1
    elif bin_ip_1 < bin_ip_2:
        flag = -1
    else:
        flag = 0
    return flag

class Interface:
    """
    Interface class
    which will carry both IP and subnet information
    """

    def __init__(self, ip, subnet, ip_structure):
        """
        constructor

        :param ip: IP address object
        :type ip: IPAddress
        :param subnet: subnet object
        :type subnet: Subnet
        :param ip_structure: ip structure - the fields that the IP is comprised of
        :type ip_structure: IPStructure
        """
        self.ip = ip
        self.subnet = subnet
        self.ip_structure = ip_structure

    def update_byte(self, pos, value):
        """
        update one byte of the decimal IP

        :param pos: byte position
        :type pos: int
        :param value: value of the byte
        :type value: int
        """
        self.ip.update_byte(pos, value)


class IPStructure:
    """
    IPStructure class (fields and its storage bits)
    which will be used for encoding and decoding
    """

    def __init__(self, fields):
        """
        constructor

        :param fields: fields encoded in the IP - a dict of (name, #bits) pairs
        :type fields: dict
        """
        self.fields = fields


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
        :type ip: np.array
        :param bin_ip: binary string of the IP
        :type bin_ip: string
        """
        self.length = length
        if ip is None and bin_ip is None:
            self.ip = np.zeros(length, dtype=np.uint8)
            self.bin_ip = self._ip_2_bin_ip()
        elif ip is not None:
            self.ip = ip
            self.bin_ip = self._ip_2_bin_ip()
        elif bin_ip is not None:
            self.bin_ip = bin_ip
            self.ip = self._bin_ip_2_ip()

    def update_byte(self, pos, value):
        """
        update one byte of the decimal IP

        :param pos: byte position
        :type pos: int
        :param value: value of the byte
        :type value: int
        """
        self.ip[pos] = value
        self.bin_ip = self._ip_2_bin_ip()

    def _ip_2_bin_ip(self, ip=None):
        """
        convert IP from decimal IP to binary string

        :param ip: decimal IP in a numpy array
        :type ip: np.array
        :return: a binary string of the IP address
        :rtype: string
        """
        if ip is None:
            ip = self.ip
        bin_ip = ip_2_bin_ip(ip)
        return bin_ip


    def _bin_ip_2_ip(self, bin_ip=None):
        """
        convert a binary string of  IP to decimal IP in a numpy array

        :param bin_ip: binary string of an IP
        :type bin_ip: string
        :return: a decimal IP address in numpy array
        :rtype: np.array
        """
        if bin_ip is None:
            bin_ip = self.bin_ip
        ip = bin_ip_2_ip(bin_ip)
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
        :rtype: string
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
        :rtype: np.array
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
        :param ip: IP address
        :type ip: numpy.array
        :param mask: subnet mask object
        :type mask: SubnetMask
        """
        self.ip = ip
        self.bin_ip = self._ip_2_bin_ip()
        self.mask = mask
        self.max_bin_ip = self._max_bin_ip()
        self.max_ip = self.max_ip

    def _ip_2_bin_ip(self, ip=None):
        """
        convert ip to binary ip

        :param ip: ip address where the subnet IP starts from
        :type ip: np.array
        :return:
        """
        if ip is None:
            ip = self.ip
        bin_ip = ip_2_bin_ip(ip)
        return bin_ip

    def _max_bin_ip(self):
        """
        calculate the max binary ip

        :return: max binary ip in the subnet
        :rtype: string
        """
        max_binary = '1' * (self.mask.ip_length - self.mask.mask_length)
        max_binary.zfill(self.mask.ip_length)
        max_bin_ip = bin_add(self.bin_ip, max_binary)
        return max_bin_ip

    def check_ip_in_subnet(self, ip):
        """
        check whether an ip is in the subnet

        :param ip: ip address
        :type ip: IPAddress
        :return: boolean value
        :rtype: bool
        """
        flag = False
        if compare_2_ips(ip.ip, self.ip.ip) >=0 and compare_2_ips(ip.ip, self.max_ip) <= 0:
            flag = True
        return flag
