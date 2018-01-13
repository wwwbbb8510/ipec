import copy
import logging

import numpy as np

from ipec.cnn.evaluator import Evaluator, CNNEvaluator, initialise_cnn_evaluator
from ipec.cnn.layers import ConvLayer
from ipec.cnn.layers import DisabledLayer
from ipec.cnn.layers import FullyConnectedLayer
from ipec.cnn.layers import PoolingLayer
from ipec.ip.decoder import Decoder
from .agent import CNNAgent
from .agent import Agent

POPULATION_DEFAULT_PARAMS = {
    'pop_size': 5, #50,
    'agent_length': 5, #15,
    'max_full': 2, #5,
    'f': 0.5,
    'cr': 0.5,
    'layers': {
        'conv': ConvLayer(),
        'pooling': PoolingLayer(),
        'full': FullyConnectedLayer(),
        'disabled': DisabledLayer()
    },
    'max_generation': 3, #50
}


def initialise_cnn_population(pop_size=None, agent_length=None, max_fully_connected_length=None, f=None, cr=None, layers=None, evaluator=None, max_generation=None):
    """
    initialise a cnn population

    :param pop_size: population size
    :type pop_size: int
    :param agent_length: the length/dimension of the agent
    :type agent_length: int
    :param max_fully_connected_length: the max length of fully-connected layers
    :type max_fully_connected_length: int
    :param f: F value in the update equation at the mutation step
    :type f: float
    :param cr: crossover rate at the mutation step
    :type cr: float
    :param layers: a dict of (layer_name, layer) pairs; keys: conv, pooling, full, disabled
    :type layers: dict
    :param max_generation: max DE generation
    :type max_generation: int
    :return: a cnn population
    :rtype: CNNPopulation
    """
    if pop_size is None:
        pop_size = POPULATION_DEFAULT_PARAMS['pop_size']
    if agent_length is None:
        agent_length = POPULATION_DEFAULT_PARAMS['agent_length']
    if max_fully_connected_length is None:
        max_fully_connected_length = POPULATION_DEFAULT_PARAMS['max_full']
    if f is None:
        f = POPULATION_DEFAULT_PARAMS['f']
    if cr is None:
        cr = POPULATION_DEFAULT_PARAMS['cr']
    if max_generation is None:
        max_generation = POPULATION_DEFAULT_PARAMS['max_generation']
    if layers is None:
        layers = POPULATION_DEFAULT_PARAMS['layers']
    logging.info('===initialise the PSO population with the following parameters===')
    logging.info('population size: %d, agent length: %d, max fully-connected length: %d, max generation: %d', pop_size, agent_length, max_fully_connected_length, max_generation)
    return CNNPopulation(pop_size, agent_length, max_fully_connected_length, f, cr, layers, evaluator, max_generation).initialise()


class Population:
    """
    Population class
    """
    def __init__(self, pop_size,  agent_length, f, cr, evaluator=None, max_generation=None):
        """
        constructor

        :param pop_size: population size
        :type pop_size: int
        :param agent_length: the length/dimension of the agent
        :type agent_length: int
        :param f: F value in the update equation at the mutation step
        :type f: float
        :param cr: crossover rate at the mutation step
        :type cr: float
        :param evaluator: evaluator to calculate the fitness
        :type evaluator: Evaluator
        :param max_generation: max generation
        :type max_generation: int
        """
        self.pop_size = pop_size
        self.pop = np.empty(pop_size, dtype=Agent)
        self.agent_length = agent_length
        self.f = f
        self.cr = cr
        self.max_generation = max_generation if max_generation > 0 else POPULATION_DEFAULT_PARAMS['max_generation']
        self.evaluator = evaluator
        self.decoder = Decoder()
        self.best_agent = None

    def evolve(self):
        """
        evolve the population
        """
        for g in range(self.max_generation):
            logging.info('===start updating population at step-%d===', g)
            i = 0
            for agent in self.pop:
                if g > 0:
                    agent_r1, agent_r2, agent_r3 = self.select(i)
                    agent_trial = self.mutate(agent, agent_r1, agent_r2, agent_r3)
                    agent_candidate = self.crossover(agent, agent_trial)
                    eval_result_candidate = self.evaluator.eval(agent)
                    agent_candidate.fitness = (eval_result_candidate[0], -eval_result_candidate[1], -eval_result_candidate[2])
                    if agent.compare_with(agent_candidate) < 0:
                        self.pop[i] = agent_candidate
                else:
                    eval_result = self.evaluator.eval(agent)
                    # use minus standard deviation which is the less the better
                    # use minus number of connections which is the less the better
                    agent.fitness = (eval_result[0], -eval_result[1], -eval_result[2])
                # update best agent
                if self.best_agent is None:
                    self.best_agent = copy.deepcopy(self.pop[i])
                elif self.best_agent.compare_with(self.pop[i]) < 0:
                    self.best_agent = copy.deepcopy(self.pop[i])
                logging.info('===fitness of Agent-%d at generation-%d: %s===', i, g, str(self.pop[i].fitness))
                i = i + 1
            logging.info('===fitness of best agent at generation-%d: %s===', g, str(self.best_agent.fitness))
            logging.info('===finish updating population at generation-%d===', g)

        return self.best_agent

    def select(self, target_index):
        """
        select three unique agents from the population exclusive of the target agent

        :param target_index: target agent index
        :type target_index: int
        :return: three unique agents
        :rtype: tuple
        """
        # randomly select agent_r1
        r1 = target_index
        while r1 == target_index:
            r1 = np.random.randint(0, self.pop_size)
            agent_r1 = self.pop[r1]

        # randomly select agent_r2, but not the same as agent_r1
        r2 = r1
        while r2 == r1 or r2 == target_index:
            r2 = np.random.randint(0, self.pop_size)
            agent_r2 = self.pop[r2]
        r3 = r2

        # randomly select agent_r3, but not the same as agent_r1 or agent_r2
        while r3 == r2 or r3 == r1 or r3 == target_index:
            r3 = np.random.randint(0, self.pop_size)
            agent_r3 = self.pop[r3]

        return (agent_r1, agent_r2, agent_r3)

    def mutate(self, target, agent_r1, agent_r2, agent_r3):
        """
        mutation

        :param target: target agent
        :type target: Agent
        :param agent_r1: first random agent
        :type agent_r1: Agent
        :param agent_r2: second random agent
        :type agent_r2: Agent
        :param agent_r3: third random agent
        :type agent_r3: Agent
        :return: difference agent
        :rtype: Agent
        """
        agent_trial = copy.deepcopy(target)
        agent_trial.id = self.pop_size+1
        for i in range(self.agent_length):
            interface = agent_trial.x[i]
            for j in range(interface.ip.length):
                rand = np.random.uniform(0, 1)
                if rand < self.cr:
                    x_r1 = agent_r1.x[i].ip.ip[j]
                    x_r2 = agent_r2.x[i].ip.ip[j]
                    x_r3 = agent_r3.x[i].ip.ip[j]
                    new_x_ij = x_r1 + self.f * (x_r2 - x_r3)
                    agent_trial.x[i].update_byte(j, new_x_ij)
        return agent_trial

    def crossover(self, target, agent_trial):
        """
        crossover

        :param target: target agent
        :type target: Agent
        :param agent_trial: difference agent
        :type agent_trial: Agent
        :return: candidate agent
        :rtype: Agent
        """
        candidate_agent = copy.deepcopy(target)
        start_point = np.random.randint(0, self.agent_length)
        mutation_length = np.random.randint(1, self.agent_length - start_point+1)
        for i in range(start_point, start_point+mutation_length):
            candidate_agent.x[i] = agent_trial.x[i]

        # fix invalid interface after crossover
        candidate_agent.fix_invalid_interface()

        return candidate_agent


class CNNPopulation(Population):
    """
    CNNPopulation class
    """
    def __init__(self, pop_size, agent_length, max_fully_connected_length, f, cr, layers, evaluator=None, max_generation=None):
        """
        constructor

        :param pop_size: population size
        :type pop_size: int
        :param agent_length: the length/dimension of the agent
        :type agent_length: int
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
        self.layers = layers
        super(CNNPopulation, self).__init__(pop_size, agent_length, f, cr, evaluator, max_generation)

    def initialise(self):
        """
        initialise the population
        """
        # set default evaluator
        if self.evaluator is None:
            self.evaluator = initialise_cnn_evaluator()
        logging.info('===start initialising population')
        for i in range(self.pop_size):
            agent = CNNAgent(i, self.agent_length, self.max_fully_connected_length, self.layers).initialise()
            self.pop[i] = agent
        logging.info('===finish initialising population')
        return self

