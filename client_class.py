import copy
import numpy as np
import sys
import random
import config
import threading
from warnings import simplefilter
from datetime import datetime
from sklearn import metrics

from agent_class import Agent
from message import Message
from utils.dp_mechanisms import laplace
import utils.diffie_hellman as dh

from sklearn.linear_model import SGDClassifier

class ClientAgent(Agent):
    def __init__(self, agent_number, train_datasets, evaluator, active_clients):

        super(ClientAgent, self).__init__(agent_number=agent_number, agent_type="client_agent")

        self.train_datasets = train_datasets
        self.evaluator = evaluator
        self.active_clients = active_clients

        self.directory = None
        self.pubkeyList = None
        self.seckeyList = None
        self.otherkeyList = None
        self.commonkeyList = None
        self.seeds = None
        self.deltas = None

        self.computation_times = {}

        self.personal_weights = {}  # personal weights. Maps iteration (int) to weights (numpy array)
        self.personal_bias = {}

        self.weights_dp_noise = {}  # keyed by iteration; noise added at each iteration
        self.bias_dp_noise = {}

        self.federated_weights = {}  # averaged weights
        self.federated_bias = {}

        self.personal_accuracy = {}
        self.federated_accuracy = {}
        
    def add_noise(self, weights, bias, iteration):
        # :return: weights, bias
        # :rtype: numpy arrays

        # Lấy shape weights
        weights_shape = weights.shape
        weights_dp_noise = np.zeros(weights_shape)

        
        bias_shape = bias.shape
        bias_dp_noise = np.zeros(bias_shape)

        # generate DP parameters
        active_clients_lens = [config.LENS_PER_ITERATION[client_name] for client_name in self.active_clients]

        smallest_dataset = min(active_clients_lens)
        if config.USING_CUMULATIVE:
            smallest_dataset *= iteration

        sensitivity = 2 / (
                len(self.active_clients) * smallest_dataset * config.alpha)
        epsilon = config.EPSILONS[self.name]

        random.seed(config.RANDOM_SEEDS[self.name][iteration])
        # adding differentially private noise
        for i in range(weights_shape[0]):  # weights_modified is 2-D
            for j in range(weights_shape[1]):
                if config.DP_ALGORITHM == 'Laplace':
                    dp_noise = laplace(mean=config.mean, sensitivity=sensitivity, epsilon=epsilon)
                elif config.DP_ALGORITHM == 'Gamma':
                    scale = sensitivity / epsilon
                    num_clients = len(self.directory.clients)
                    dp_noise = random.gammavariate(1 / num_clients, scale) - random.gammavariate(1 / num_clients,
                                                                                                scale)
                else:
                    raise AssertionError('Need to specify config.DP_ALGORITHM as Laplace or Gamma')
                weights_dp_noise[i][j] = dp_noise

        if config.bias_DP_NOISE:
            for i in range(bias_shape[0]):
                if config.DP_ALGORITHM == 'Laplace':
                    dp_noise = laplace(mean=config.mean, sensitivity=sensitivity, epsilon=epsilon)
                elif config.DP_ALGORITHM == 'Gamma':
                    scale = sensitivity / epsilon
                    num_clients = len(self.directory.clients)
                    dp_noise = random.gammavariate(1 / num_clients, scale) - random.gammavariate(1 / num_clients, scale)
                else:
                    raise AssertionError('Need to specify config.DP_ALGORITHM as Laplace or Gamma')
                bias_dp_noise[i] = dp_noise

        weights_with_noise = copy.deepcopy(weights)  # make a copy to not mutate weights
        bias_with_noise = copy.deepcopy(bias)

        self.weights_dp_noise[iteration] = weights_dp_noise
        weights_with_noise += weights_dp_noise
        self.bias_dp_noise[iteration] = bias_dp_noise
        bias_with_noise += bias_dp_noise
        return weights_with_noise, bias_with_noise
    
    def HE(weights_with_noise, bias_with_noise):
        
        return 1