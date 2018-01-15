import numpy as np
import logging
from ipec.ip.core import Interface, InterfaceArray
from ipec.ip.decoder import Decoder
import copy
import pickle

DEFAULT_PARTICLE_PATH = 'log/ga_gbest.pkl'

def save_chromosome(chromosome, path=None):
    """
    save a chromosome using pickle

    :param chromosome: the chromosome to be saved
    :type chromosome: Chromosome
    :param path: path to write the serialised chromosome
    :type path: string
    """
    if path is None:
        path = DEFAULT_PARTICLE_PATH
    with open(path, 'wb') as output:
        pickle.dump(chromosome, output, pickle.HIGHEST_PROTOCOL)
    output.close()

def load_chromosome(path=None):
    """
    load chromosome from persisted pickle file

    :param path: pickle file path
    :type path: string
    :return: loaded chromosome
    :rtype: Chromosome
    """
    if path is None:
        path = DEFAULT_PARTICLE_PATH
    with open(path, 'rb') as input:
        chromosome = pickle.load(input)
    input.close()
    return chromosome


class Chromosome(InterfaceArray):
    """
    Chromosome class
    """
    def __init__(self, id, length, layers=None, generation=0):
        """
        constructor

        :param id: chromosome ID
        :type id: int
        :param length: the length/dimension of the chromosome
        :type length: int
        :param layers: a dict of (layer_name, layer) pairs; keys: conv, pooling, full, disabled
        :type layers: dict
        :param generation: DE generation
        :type generation: int
        """
        super(Chromosome, self).__init__(id)
        self.length = length
        self.layers = layers
        self.generation = generation
        self.decoder = Decoder()

        # initialise fitness
        self.fitness = None

    def compare_with(self, chromosome):
        """
        compare this chromosome to the input chromosome

        :param chromosome: the input chromosome
        :type chromosome: Chromosome
        :return: 0: equal, 1: this chromosome is greater, -1: this chromosome is less
        :rtype: int
        """
        return self.compare_fitness(self.fitness, chromosome.fitness)

    def compare_fitness(self, fitness_1, fitness_2):
        """
        compare fitness of two chromosomes @abstractmethod

        :param fitness_1: fitness of the first chromosome
        :type: tuple
        :param fitness_2: fitness of the second chromosome
        :type tuple
        :return: 0: equal, 1: greater, -1: less
        :rtype: int
        """


class CNNChromosome(Chromosome):
    """
    CNN Chromosome class
    """
    def __init__(self, id, length, max_fully_connected_length, layers, generation=0):
        """
        constructor

        :param id: chromosome ID
        :type id: int
        :param length: the length/dimension of the chromosome
        :type length: int
        :param max_fully_connected_length: the max length of fully-connected layers
        :type max_fully_connected_length: int
        :param layers: a dict of (layer_name, layer) pairs; keys: conv, pooling, full, disabled
        :type layers: dict
        :param generation: DE generation
        :type generation: int
        """
        logging.info('===initialise CNN chromosome with ID: %d===', id)
        self.max_fully_connected_length = max_fully_connected_length
        super(CNNChromosome, self).__init__(id, length, layers, generation)
        self.x = np.empty(self.length, dtype=Interface)

    def check_interface_in_types(self, interface,  layer_names=None):
        """
        check whether the interface is in the types of layers with the given layer names

        :param interface: interface to be checked
        :type interface: Interface
        :param layer_names: layer type names, (conv, full, disabled, pooling)
        :type layer_names: list
        :return: check result (true or false)
        :rtype: bool
        """
        flag = False
        if layer_names is None:
            layer_names = ['conv', 'pooling', 'full', 'disabled']
        for name in layer_names:
            flag = self.layers[name].check_interface_in_type(interface)
            if flag == True:
                break
        return flag

    def fix_invalid_interface(self):
        """
        fix invalid layers
        """
        is_fully_connected_found = False
        logging.debug('===start fixing invalid layers of Chromosome-%d===', self.id)
        for i in range(self.length):
            # fix first layer is not conv layer
            if i == 0:
                if not self.layers['conv'].check_interface_in_type(self.x[i]):
                    logging.debug('fix the first layer')
                    logging.debug('Interface-%d before fixed: %s', i, str(self.x[i]))
                    self.x[i] = self.layers['conv'].generate_random_interface()
                    logging.debug('Interface-%d after fixed: %s', i, str(self.x[i]))
            # fix layer between second layer to (max-max_fully_connected) layer is fully-connected or invalid
            elif i <= self.length - self.max_fully_connected_length:
                if self.check_interface_in_types(self.x[i], ['full']) or not self.check_interface_in_types(self.x[i]):
                    logging.debug('fix middle layers')
                    logging.debug('Interface-%d before fixed: %s', i, str(self.x[i]))
                    available_layers = ['conv', 'pooling', 'disabled']
                    self.x[i] = self.initialise_by_chance(available_layers)
                    logging.debug('Interface-%d after fixed: %s', i, str(self.x[i]))
            # fix non-fully connected layer or invalid layer after fully-connected layer found
            elif i > self.length - self.max_fully_connected_length:
                # fix invalid layer with random layer types
                if not self.check_interface_in_types(self.x[i]):
                    logging.debug('fix invalid layers in the last layers')
                    logging.debug('Interface-%d before fixed: %s', i, str(self.x[i]))
                    self.x[i] = self.initialise_by_chance(['conv', 'pooling', 'full', 'disabled'])
                    logging.debug('Interface-%d after fixed: %s', i, str(self.x[i]))
                # the layer is fully-connected
                if not is_fully_connected_found and self.check_interface_in_types(self.x[i], ['full']):
                    is_fully_connected_found = True
                # after fully-connected layer found, the layer is not fully-connected or disabled or is not valid
                elif is_fully_connected_found and not (self.check_interface_in_types(self.x[i], ['full', 'disabled'])):
                    logging.debug('fix last layers that non-fully-connected layer found after fully-connected layer')
                    logging.debug('Interface-%d before fixed: %s', i, str(self.x[i]))
                    available_layers = ['full', 'disabled']
                    self.x[i] = self.initialise_by_chance(available_layers)
                    logging.debug('Interface-%d after fixed: %s', i, str(self.x[i]))
                # fix no fully-connected layer found until the end
                if i == self.length - 1 and not is_fully_connected_found:
                    self.x[i] = self.initialise_by_chance(['full'])

        logging.debug('===finish fixing invalid layers of Chromosome-%d===', self.id)

    def initialise(self):
        """
        initialise a chromosome with random settings
        """
        logging.info('===start initialising CNN chromosome===')
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
                available_layers = ['conv', 'pooling', 'full', 'disabled']
                interface = self.initialise_by_chance(available_layers)
                offset = self.max_fully_connected_length - (self.length - i + 1)
                self.x[self.length-2-offset] = interface
                if not self.layers['full'].check_interface_in_type(interface):
                    del available_layers[2]
            # initialise the middle part which can be conv, pooling or disabled
            else:
                available_layers = ['conv', 'pooling', 'disabled']
                self.x[i] = self.initialise_by_chance(available_layers)
            logging.info('Interface of Layer-%d: %s', i, str(self.x[i]))
        logging.info('===finish initialising CNN chromosome===')
        return self

    def initialise_by_chance(self, layer_names):
        """
        initialise a type of layer by chance

        :param layer_names: a list names of possible types of layers
        :type layer_names: list
        :return: an IP interface
        :rtype: Interface
        """
        num_of_types = len(layer_names)
        prob = np.random.uniform(0, 1)
        interface = None
        for i in range(num_of_types):
            if prob >= 1/num_of_types * i and prob < 1/num_of_types * (i+1):
                interface = self.layers[layer_names[i]].generate_random_interface()
                break
        if interface is None:
            interface = self.layers[layer_names[0]].generate_random_interface()
        return interface

    def compare_fitness(self, fitness_1, fitness_2):
        """
        compare fitness of two chromosomes @abstractmethod

        :param fitness_1: fitness of the first chromosome
        :type: tuple
        :param fitness_2: fitness of the second chromosome
        :type tuple
        :return: 0: equal, 1: greater, -1: less
        :rtype: int
        """
        flag = None
        for i in range(len(fitness_1)):
            if fitness_1[i] > fitness_2[i]:
                flag = 1
                break
            elif fitness_1[i] < fitness_2[i]:
                flag = -1
                break
            else:
                flag = 0
        return flag