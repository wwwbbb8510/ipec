import numpy as np
from .particle import Particle
from .particle import CNNParticle
from ippso.cnn.layers import ConvLayer
from ippso.cnn.layers import FullyConnectedLayer
from ippso.cnn.layers import DisabledLayer
from ippso.cnn.layers import PoolingLayer
from .evaluator import Evaluator, CNNEvaluator, initialise_cnn_evaluator
import copy

POPULATION_DEFAULT_PARAMS = {
    'pop_size': 5, #50,
    'particle_length': 5, #15,
    'max_full': 2, #5,
    'w': 0.1,
    'c1': np.asarray([0.00001, 0.0001, 0.001, 0.01, 0.1]),
    'c2': np.asarray([0.00001, 0.0001, 0.001, 0.01, 0.1]),
    'layers': {
        'conv': ConvLayer(),
        'pooling': PoolingLayer(),
        'full': FullyConnectedLayer(),
        'disabled': DisabledLayer()
    },
    'max_steps': 5, #50
}


def initialise_cnn_population(pop_size=None, particle_length=None, max_fully_connected_length=None, w=None, c1=None, c2=None, layers=None):
    """
    initialise a cnn population

    :param pop_size: population size
    :type pop_size: int
    :param particle_length: the length/dimension of the particle
    :type particle_length: int
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
    :return: a cnn population
    :rtype: CNNPopulation
    """
    if pop_size is None:
        pop_size = POPULATION_DEFAULT_PARAMS['pop_size']
    if particle_length is None:
        particle_length = POPULATION_DEFAULT_PARAMS['particle_length']
    if max_fully_connected_length is None:
        max_fully_connected_length = POPULATION_DEFAULT_PARAMS['max_full']
    if w is None:
        w = POPULATION_DEFAULT_PARAMS['w']
    if c1 is None:
        c1 = POPULATION_DEFAULT_PARAMS['c1']
    if c2 is None:
        c2 = POPULATION_DEFAULT_PARAMS['c2']
    if layers is None:
        layers = POPULATION_DEFAULT_PARAMS['layers']
    return CNNPopulation(pop_size, particle_length, max_fully_connected_length, w, c1, c2, layers).initialise()


class Population:
    """
    Population class
    """
    def __init__(self, pop_size,  particle_length, w, c1, c2, evaluator=None):
        """
        constructor

        :param pop_size: population size
        :type pop_size: int
        :param particle_length: the length/dimension of the particle
        :type particle_length: int
        :param w: inertia weight
        :type w: float
        :param c1: an array of acceleration co-efficients for pbest
        :type c1: numpy.array
        :param c2: an array of acceleration co-efficients for gbest
        :type c2: numpy.array
        :param evaluator: evaluator to calculate the fitness
        :type evaluator: Evaluator
        """
        self.pop_size = pop_size
        self.pop = np.empty(pop_size, dtype=Particle)
        self.particle_length = particle_length
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.evaluator = evaluator

        # initialise gbest to None
        self.gbest = None

    def fly_a_step(self):
        """
        train the PSO population for one step
        """
        for particle in self.pop:
            particle.update(self.gbest)
            fitness = self.evaluator.eval(particle)
            particle.update_pbest(fitness)
            # gbest has never not been evaluated
            if self.gbest.fitness is None:
                self.gbest = copy.deepcopy(particle.pbest)
            # pbest is greater than gbest, update gbest
            elif particle.pbest.compare_with(self.gbest) > 0:
                self.gbest = copy.deepcopy(particle.pbest)


    def fly_2_end(self, max_steps=None):
        """
        train the PSO population until the termination criteria meet

        :param max_steps: max fly steps
        :type max_steps: int
        """
        if max_steps is None:
            max_steps = POPULATION_DEFAULT_PARAMS['max_steps']
        for i in range(max_steps):
            self.fly_a_step()
        return self.gbest

class CNNPopulation(Population):
    """
    CNNPopulation class
    """
    def __init__(self, pop_size, particle_length, max_fully_connected_length, w, c1, c2, layers, evaluator=None):
        """
        constructor

        :param pop_size: population size
        :type pop_size: int
        :param particle_length: the length/dimension of the particle
        :type particle_length: int
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
        :param evaluator: evaluator to calculate the fitness
        :type evaluator: CNNEvaluator
        """
        self.max_fully_connected_length = max_fully_connected_length
        self.layers = layers
        super(CNNPopulation, self).__init__(pop_size, particle_length, w, c1, c2, evaluator)

    def initialise(self):
        """
        initialise the population
        """
        # set default evaluator
        if self.evaluator is None:
            self.evaluator = initialise_cnn_evaluator()
        for i in range(self.pop_size):
            particle = CNNParticle(i, self.particle_length, self.max_fully_connected_length, self.w, self.c1, self.c2, self.layers).initialise()
            self.pop[i] = particle
        self.gbest = self.pop[0]
        return self
