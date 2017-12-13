import numpy as np
from ippso.ip.core import Interface

class Particle:
    """
    Particle class
    """
    def __init__(self, length, w, c1, c2):
        """
        constructor

        :param length: the length/dimension of the particle
        :type length: int
        :param w: inertia weight
        :type w: float
        :param c1: a list of acceleration co-efficients for pbest
        :type c1: list
        :param c2: a list of acceleration co-efficients for gbest
        :type c2: list
        """
        self.length = length
        self.w = w
        self.c1 = c1
        self.c2 = c2

        # initialise pbest to None
        self.pbest = None

    def update(self, gbest):
        """
        update particle

        :param gbest: global best
        :type gbest: Particle
        """



class CNNParticle(Particle):
    """
    CNN Particle class
    """
    def __init__(self, length, max_fully_connected_length, w, c1, c2, layers):
        """
        constructor

        :param length: the length/dimension of the particle
        :type length: int
        :param max_fully_connected_length: the max length of fully-connected layers
        :type max_fully_connected_length: int
        :param w: inertia weight
        :type w: float
        :param c1: a list of acceleration co-efficients for pbest
        :type c1: list
        :param c2: a list of acceleration co-efficients for gbest
        :type c2: list
        :param layers: a dict of (layer_name, layer) pairs; keys: conv, pooling, full, disabled
        :type layers: dict
        """
        self.max_fully_connected_length = max_fully_connected_length
        self.layers = layers
        super(CNNParticle, self).__init__(length, w, c1, c2)
        self.x = np.empty(self.length, dtype=Interface)
        self.v = None

    def initialise(self):
        """
        initialise a particle with random settings
        """
        for i in range(self.length):
            # initialise the first layer which will always be a conv layer
            # and initialise the velocity with zeros
            if i == 0:
                interface = self.layers['conv'].generate_random_interface()
                self.x[i] = interface
                self.v = np.zeros((self.length, interface.ip.length))
            # initialise the last layer which will always be a fully-connected layer
            elif i == self.length - 1:
                self.x[i] = self.layers['full'].generate_random_interface()
            # initialise the (max_fully_connected_length-1 layers) from the right end of the layers
            # which can be conv, pooling, fully-connected or disabled
            # and initialise in reversed order
            # and remove fully-connected from the above list when one of the initialised layer is not fully-connected
            elif i > self.length - self.max_fully_connected_length:
                available_layers = [self.layers['conv'], self.layers['pooling'], self.layers['full'], self.layers['disabled']]
                interface = self.initialise_by_chance(available_layers)
                offset = self.max_fully_connected_length - (self.length - i + 1)
                self.x[self.length-2-offset] = interface
                if not self.layers['full'].check_interface_in_type(interface):
                    del available_layers[2]
            # initialise the middle part which can be conv, pooling or disabled
            else:
                available_layers = [self.layers['conv'], self.layers['pooling'], self.layers['disabled']]
                self.x[i] = self.initialise_by_chance(available_layers)

    def initialise_by_chance(self, layers):
        """
        initialise a type of layer by chance

        :param layers: a list of possible types of layers
        :type layers: list
        :return: an IP interface
        :rtype: Interface
        """
        num_of_types = len(layers)
        prob = np.random.uniform(0, 1)
        interface = None
        for i in range(num_of_types):
            if prob >= 1/num_of_types * i and prob < 1/num_of_types * (i+1):
                interface = layers[i].generate_random_interface()
                break
        if interface is None:
            interface = layers[0].generate_random_interface()
        return interface