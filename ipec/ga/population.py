import copy
import logging

import numpy as np

from ipec.cnn.evaluator import Evaluator, CNNEvaluator, initialise_cnn_evaluator
from ipec.cnn.layers import ConvLayer
from ipec.cnn.layers import DisabledLayer
from ipec.cnn.layers import FullyConnectedLayer
from ipec.cnn.layers import PoolingLayer
from ipec.ip.decoder import Decoder
from .chromosome import Chromosome, CNNChromosome

POPULATION_DEFAULT_PARAMS = {
    'pop_size': 3, #50,
    'chromosome_length': 5, #15,
    'max_full': 2, #5,
    'elitism_rate': 0.5,
    'mutation_rate': np.asarray([0.1, 0.2]),
    'layers': {
        'conv': ConvLayer(),
        'pooling': PoolingLayer(),
        'full': FullyConnectedLayer(),
        'disabled': DisabledLayer()
    },
    'max_generation': 3, #50
}


def initialise_cnn_population(pop_size=None, chromosome_length=None, max_fully_connected_length=None, elitism_rate=None, mutation_rate=None, layers=None, evaluator=None, max_generation=None):
    """
    initialise a cnn population

    :param pop_size: population size
    :type pop_size: int
    :param chromosome_length: the length/dimension of the chromosome
    :type chromosome_length: int
    :param max_fully_connected_length: the max length of fully-connected layers
    :type max_fully_connected_length: int
    :param elitism_rate: elitism rate
    :type elitism_rate: float
    :param mutation_rate: mutation rate. [mutation rate for interfaces in a chromosome, mutation rate for bits in an interface]
    :type mutation_rate: numpy.array
    :param layers: a dict of (layer_name, layer) pairs; keys: conv, pooling, full, disabled
    :type layers: dict
    :param max_generation: max DE generation
    :type max_generation: int
    :return: a cnn population
    :rtype: CNNPopulation
    """
    if pop_size is None:
        pop_size = POPULATION_DEFAULT_PARAMS['pop_size']
    if chromosome_length is None:
        chromosome_length = POPULATION_DEFAULT_PARAMS['chromosome_length']
    if max_fully_connected_length is None:
        max_fully_connected_length = POPULATION_DEFAULT_PARAMS['max_full']
    if mutation_rate is None:
        mutation_rate = POPULATION_DEFAULT_PARAMS['mutation_rate']
    if elitism_rate is None:
        elitism_rate = POPULATION_DEFAULT_PARAMS['elitism_rate']
    if max_generation is None:
        max_generation = POPULATION_DEFAULT_PARAMS['max_generation']
    if layers is None:
        layers = POPULATION_DEFAULT_PARAMS['layers']
    logging.info('===initialise the PSO population with the following parameters===')
    logging.info('population size: %d, chromosome length: %d, max fully-connected length: %d, max generation: %d', pop_size, chromosome_length, max_fully_connected_length, max_generation)
    return CNNPopulation(pop_size, chromosome_length, max_fully_connected_length, elitism_rate, mutation_rate, layers, evaluator, max_generation).initialise()


class Population:
    """
    Population class
    """
    def __init__(self, pop_size,  chromosome_length, elitism_rate, mutation_rate, layers, evaluator=None, max_generation=None):
        """
        constructor

        :param pop_size: population size
        :type pop_size: int
        :param chromosome_length: the length/dimension of the chromosome
        :type chromosome_length: int
        :param elitism_rate: elitism rate
        :type elitism_rate: float
        :param mutation_rate: mutation rate. [mutation rate for interfaces in a chromosome, mutation rate for bits in an interface]
        :type mutation_rate: numpy.array
        :param layers: a dict of (layer_name, layer) pairs; keys: conv, pooling, full, disabled
        :type layers: dict
        :param evaluator: evaluator to calculate the fitness
        :type evaluator: Evaluator
        :param max_generation: max generation
        :type max_generation: int
        """
        self.pop_size = pop_size
        self.pop = np.empty(pop_size, dtype=Chromosome)
        self.chromosome_length = chromosome_length
        self.elitism_rate = elitism_rate
        self.mutation_rate = mutation_rate
        self.layers = layers
        self.max_generation = max_generation if max_generation > 0 else POPULATION_DEFAULT_PARAMS['max_generation']
        self.evaluator = evaluator
        self.decoder = Decoder()
        self.best_chromosome = None
        self.roulette_proportions = None

    def evolve(self):
        """
        evolve the population
        """
        for g in range(self.max_generation):
            logging.info('===start updating population at step-%d===', g)
            # evaluate the first generation as the chromosomes are not evaluated during initialisation
            if g == 0:
                for chromosome in self.pop:
                    eval_result = self.evaluator.eval(chromosome)
                    # use minus standard deviation which is the less the better
                    # use minus number of connections which is the less the better
                    chromosome.fitness = (eval_result[0], -eval_result[1], -eval_result[2])

            # generate new pop
            new_pop = np.empty(self.pop_size, dtype=Chromosome)
            new_pop_index = 0
            # add elite chromosomes in the new generation
            elite_chromosomes = self.elitism()
            if elite_chromosomes is not None:
                for chromosome in elite_chromosomes:
                    new_chromosome = copy.deepcopy(chromosome)
                    new_chromosome.id = new_pop_index
                    new_pop[new_pop_index] = new_chromosome
                    new_pop_index = new_pop_index + 1
            # generate children (after doing selection, crossover, mutation) in the population
            while new_pop_index < self.pop_size:
                chromosome_1, chromosome_2 = self.select()
                candidate_chromosome = self.crossover(chromosome_1, chromosome_2)
                candidate_chromosome = self.mutate(candidate_chromosome)
                candidate_chromosome.id = new_pop_index
                eval_result = self.evaluator.eval(chromosome)
                # use minus standard deviation which is the less the better
                # use minus number of connections which is the less the better
                chromosome.fitness = (eval_result[0], -eval_result[1], -eval_result[2])
                # update best chromosome
                if self.best_chromosome is None:
                    self.best_chromosome = copy.deepcopy(self.pop[new_pop_index])
                elif self.best_chromosome.compare_with(self.pop[new_pop_index]) < 0:
                    self.best_chromosome = copy.deepcopy(self.pop[new_pop_index])
                logging.info('===fitness of Chromosome-%d at generation-%d: %s===', new_pop_index, g, str(self.pop[new_pop_index].fitness))
                new_pop[new_pop_index] = candidate_chromosome
                new_pop_index = new_pop_index + 1

            logging.info('===fitness of best chromosome at generation-%d: %s===', g, str(self.best_chromosome.fitness))
            logging.info('===finish updating population at generation-%d===', g)

        return self.best_chromosome

    def elitism(self):
        """
        GA elitism

        :return: elitism array of chromosome
        :type: numpy.array
        """
        elitism_pop = None
        elitism_amount = int(self.elitism_rate * self.pop_size)
        if elitism_amount > 0:
            # construct a sortable array
            dtype = [('chromosome', Chromosome), ('s_0', float), ('s_1', float), ('s_2', float)]
            sortable_pop = np.empty(self.pop_size, dtype=dtype)
            for i in range(self.pop_size):
                fitness = self.pop[i].fitness
                sortable_pop[i] = (self.pop[i], fitness[0], fitness[1], fitness[2])
            sorted_pop = np.sort(sortable_pop, order=['s_0', 's_1', 's_2'])
            elitism_pop = np.empty(elitism_amount, dtype=Chromosome)
            for i in range(self.pop_size-elitism_amount, self.pop_size):
                elitism_pop[i-(self.pop_size-elitism_amount)] = sorted_pop[i][0]

        return elitism_pop


    def select(self):
        """
        select two chromosomes for crossover and mutation

        :return: two unique chromosomes
        :rtype: tuple
        """
        # roulette-select chromosome_1
        c1_index = self.spin_roulette()
        chromosome_1 = self.pop[c1_index]
        # roulette-select chromosome_2
        c2_index = c1_index
        while c1_index == c2_index:
            c2_index = self.spin_roulette()
            chromosome_2 = self.pop[c2_index]

        return (chromosome_1, chromosome_2)


    def spin_roulette(self):
        if self.roulette_proportions is None:
            self.roulette_proportions = self.calculate_roulette_proportions()
        prob = np.random.uniform(0, 1)
        roulette_index = self.pop_size - 1
        for i in range(self.roulette_proportions.shape[0]):
            if prob < self.roulette_proportions[i]:
                roulette_index = i
                break
        return roulette_index

    def calculate_roulette_proportions(self):
        """
        calculate roulette proportions for selection
        :return:
        """
        # calculate the accumulated fitness
        accumulated_fitness = 0
        for chromosome in self.pop:
            accumulated_fitness += chromosome.fitness[0]
        # calculate the proportion
        previous_roulette_point = 0
        self.roulette_proportions = np.zeros(29)
        for i in range(self.pop_size-1):
            new_roulette_point = previous_roulette_point + self.pop[i].fitness[0]/accumulated_fitness
            self.roulette_proportions[i] = new_roulette_point
            previous_roulette_point = new_roulette_point
        return self.roulette_proportions

    def crossover(self, chromosome_1, chromosome_2):
        """
        crossover

        :param chromosome_1: first parent chromosome
        :type chromosome_1: Chromosome
        :param chromosome_2: second parent chromosome
        :type chromosome_2: Chromosome
        :return: candidate chromosome
        :rtype: Chromosome
        """
        candidate_chromosome = copy.deepcopy(chromosome_1)
        start_point = np.random.randint(0, self.chromosome_length)
        mutation_length = np.random.randint(1, self.chromosome_length - start_point+1)
        for i in range(start_point, start_point+mutation_length):
            candidate_chromosome.x[i] = chromosome_2.x[i]

        return candidate_chromosome

    def mutate(self, candidate_chromosome):
        """
        mutation

        :param candidate_chromosome: candidate chromosome after crossover
        :type candidate_chromosome: Chromosome
        :return: candidate chromosome
        :rtype: Chromosome
        """
        for i in range(self.chromosome_length):
            interface = candidate_chromosome.x[i]
            rand = np.random.uniform(0, 1)
            # check whether to mutate the interface
            if rand < self.mutation_rate[0]:
                bin_ip = interface.ip.bin_ip
                for j in range(len(bin_ip)):
                    # check whether to mutate the bit
                    rand = np.random.uniform(0, 1)
                    if rand < self.mutation_rate:
                        bin_ip[j] = '0' if bin_ip[j] == '1' else '1'
                candidate_chromosome.x[i].update_ip_by_binary_string(bin_ip)
                if self.layers is not None:
                    candidate_chromosome.x[i].update_subnet_and_structure(self.layers)
            else:
                continue

        # fix invalid interface after crossover
        candidate_chromosome.fix_invalid_interface()
        return candidate_chromosome


class CNNPopulation(Population):
    """
    CNNPopulation class
    """
    def __init__(self, pop_size, chromosome_length, max_fully_connected_length, elitism_rate, mutation_rate, layers, evaluator=None, max_generation=None):
        """
        constructor

        :param pop_size: population size
        :type pop_size: int
        :param chromosome_length: the length/dimension of the chromosome
        :type chromosome_length: int
        :param max_fully_connected_length: the max length of fully-connected layers
        :type max_fully_connected_length: int
        :param f: F value in the update equation at the mutation step
        :type f: float
        :param cr: crossover rate at the mutation step
        :type cr: float
        :param layers: a dict of (layer_name, layer) pairs; keys: conv, pooling, full, disabled
        :type layers: dict
        :param evaluator: evaluator to calculate the fitness
        :type evaluator: CNNEvaluator
        :param max_generation: max generation
        :type max_generation: int
        """
        self.max_fully_connected_length = max_fully_connected_length
        super(CNNPopulation, self).__init__(pop_size, chromosome_length, elitism_rate, mutation_rate, layers, evaluator, max_generation)

    def initialise(self):
        """
        initialise the population
        """
        # set default evaluator
        if self.evaluator is None:
            self.evaluator = initialise_cnn_evaluator()
        logging.info('===start initialising population')
        for i in range(self.pop_size):
            chromosome = CNNChromosome(i, self.chromosome_length, self.max_fully_connected_length, self.layers).initialise()
            self.pop[i] = chromosome
        logging.info('===finish initialising population')
        return self

