import numpy as np
import logging
from ippso.ip.core import Interface
import copy
import pickle


DEFAULT_PARTICLE_PATH = 'log/gbest.pkl'

def save_particle(particle, path=None):
    """
    save a particle using pickle

    :param particle: the particle to be saved
    :type particle: Particle
    :param path: path to write the serialised particle
    :type path: string
    """
    if path is None:
        path = DEFAULT_PARTICLE_PATH
    with open(path, 'wb') as output:
        pickle.dump(particle, output, pickle.HIGHEST_PROTOCOL)

def load_particle(path=None):
    """
    load a particle from saved path
    :param path: path containing the saved particle
    :type path: string
    :return: the particle instance loaded from the path
    :rtype: Particle
    """
    if path is None:
        path = DEFAULT_PARTICLE_PATH
    with open(path, 'rb') as input:
        particle = pickle.load(input)
    return particle

class Particle:
    """
    Particle class
    """
    def __init__(self, id, length, w, c1, c2):
        """
        constructor

        :param id: particle ID
        :type id: int
        :param length: the length/dimension of the particle
        :type length: int
        :param w: inertia weight
        :type w: float
        :param c1: an array of acceleration co-efficients for pbest
        :type c1: numpy.array
        :param c2: an array of acceleration co-efficients for gbest
        :type c2: numpy.array
        """
        self.id = id
        self.length = length
        self.w = w
        self.c1 = c1
        self.c2 = c2

        # initialise pbest, x, v to None
        self.pbest = None
        self.x = None
        self.v = None
        self.fitness = None

    def update(self, gbest):
        """
        update particle

        :param gbest: global best
        :type gbest: Particle
        """
        # initialise pbest by copying itself
        if self.pbest is None:
            self.pbest = copy.deepcopy(self)
        # update position and velocity
        for i in range(self.length):
            logging.debug('===start updating velocity and position of Interface-%d of Particle-%d===', i, self.id)
            logging.debug('interface before update: %s', str(self.x[i]))
            logging.debug('velocity before update: %s', str(self.v[i, :]))
            logging.debug('interface in pbest: %s', str(self.pbest.x[i]))
            logging.debug('interface in gbest: %s', str(gbest.x[i]))
            interface = self.x[i]
            gbest_interface = gbest.x[i]
            pbest_interface = self.pbest.x[i]
            for j in range(interface.ip.length):
                logging.debug('===start updating bytes-%d of Interface-%d of Particle-%d===', j, i, self.id)
                # calculate the new position and velocity of one byte of the IP address
                v_ij = self.v[i,j]
                x_ij = interface.ip.ip[j]
                gbest_x_ij = gbest_interface.ip.ip[j]
                pbest_x_ij = pbest_interface.ip.ip[j]
                r1 = np.random.uniform(0, 1)
                r2 = np.random.uniform(0, 1)
                new_v_ij = self.w * v_ij + self.c1[j] * r1 * (pbest_x_ij - x_ij) + self.c2[j] * r2 * (gbest_x_ij - x_ij)
                new_x_ij = x_ij + new_v_ij
                new_x_ij if new_x_ij < 256 else new_x_ij - 256
                # update the IP and velocity of the particle
                self.x[i].update_byte(j, new_x_ij)
                self.v[i,j] = new_v_ij
                logging.debug('===finish updating bytes-%d of Interface-%d of Particle-%d===', j, i, self.id)
            logging.debug('interface after update: %s', str(self.x[i]))
            logging.debug('velocity after update: %s', str(self.v[i, :]))
            logging.debug('===finish updating velocity and position of Interface-%d of Particle-%d===', i, self.id)

    def update_pbest(self, fitness):
        """
        update pbest

        :param fitness: fitness tuple
        :type fitness: tuple
        """
        logging.info('===start updating pbest of Particle-%d===', self.id)
        self.fitness = fitness
        # initialise the pbest with the first evaluated particle
        if self.pbest.fitness is None:
            self.pbest = copy.deepcopy(self)
            logging.info('pbest is initialised as the original particle')
        else:
            flag = self._compare_fitness(self.fitness, self.pbest.fitness)
            # particle fitness is greater than the pbest fitness
            if flag > 0:
                pbest_particle = copy.deepcopy(self)
                self.pbest = pbest_particle
                logging.info('pbest is updated by the updated particle')
        logging.info('===finish updating pbest of Particle-%d===', self.id)

    def compare_with(self, particle):
        """
        compare this particle to the input particle

        :param particle: the input particle
        :type particle: Particle
        :return: 0: equal, 1: this particle is greater, -1: this particle is less
        :rtype: int
        """
        return self._compare_fitness(self.fitness, particle.fitness)

    def _compare_fitness(self, fitness_1, fitness_2):
        """
        compare fitness of two particles @abstractmethod

        :param fitness_1: fitness of the first particle
        :type: tuple
        :param fitness_2: fitness of the second particle
        :type tuple
        :return: 0: equal, 1: greater, -1: less
        :rtype: int
        """


class CNNParticle(Particle):
    """
    CNN Particle class
    """
    def __init__(self, id, length, max_fully_connected_length, w, c1, c2, layers):
        """
        constructor

        :param id: particle ID
        :type id: int
        :param length: the length/dimension of the particle
        :type length: int
        :param max_fully_connected_length: the max length of fully-connected layers
        :type max_fully_connected_length: int
        :param w: inertia weight
        :type w: float
        :param c1: an array of acceleration co-efficients for pbest
        :type c1: numpy.array
        :param c2: an array of acceleration co-efficients for gbest
        :type c2: numpy.array
        :param layers: a dict of (layer_name, layer) pairs; keys: conv, pooling, full, disabled
        :type layers: dict
        """
        logging.info('===initialise CNN particle with ID: %d===', id)
        self.max_fully_connected_length = max_fully_connected_length
        self.layers = layers
        super(CNNParticle, self).__init__(id, length, w, c1, c2)
        self.x = np.empty(self.length, dtype=Interface)


    def initialise(self):
        """
        initialise a particle with random settings
        """
        logging.info('===start initialising CNN particle===')
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
            logging.info('Interface of Layer-%d: %s', i, str(self.x[i]))
        logging.info('===finish initialising CNN particle===')
        return self

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

    def _compare_fitness(self, fitness_1, fitness_2):
        """
        compare fitness of two particles @abstractmethod

        :param fitness_1: fitness of the first particle
        :type: tuple
        :param fitness_2: fitness of the second particle
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

