"""Imports"""
from datetime import timedelta
import random

"""Simulation Parameters"""

# NUM_CLIENTS: Number of client agents
NUM_CLIENTS = 3
client_names = ['client_agent_' + str(i) for i in range(NUM_CLIENTS)]
# NUM_SERVERS: More than one server will require additional coding to specify each server's behavior in the simulation
NUM_SERVERS = 1

# ITERATIONS: Số vòng lặp chạy
ITERATIONS = 10

# LEN_PER_ITERATION: How many datapoints each client gets per iteration (starts at 0). On iteration i, each client has (i+1) * LEN_PER_ITERATION samples
len_per_iteration = 50  # using equal size datasets for each client in this example
LENS_PER_ITERATION = {client_name: len_per_iteration for client_name in client_names}

# LEN_TEST: Length of test dataset. Note whole dataset length is 1797
LEN_TEST = 300

VERBOSITY = 1 # verbose result

"""Dataset Formatting"""
# When set to True, simulation uses cumulative datasets (i.e., each iteration includes the last), which also affects the way
# training is done. With cumulative datasets, each iteration each client trains on its dataset from scratch.
USING_CUMULATIVE = True


"""Security"""
# USE_SECURITY: Implements Diffie-Helman key exchange for added security. Slows runtime slightly, but no effect on performance
USE_SECURITY = True

"""Differential Privacy"""
# USE_DP_PRIVACY: Whether to implement differential privacy functionality. Defaults to laplacian noise.
USE_DP_PRIVACY = False
# SUBTRACT_DP_NOISE: Use more advanced version of protocol which has each client subtract the DP noise it added from the federated model it receives
SUBTRACT_DP_NOISE = False  # Subtract your own DP noise from federated model to increase accuracy
assert (SUBTRACT_DP_NOISE == False or (
            SUBTRACT_DP_NOISE == True and USE_DP_PRIVACY == True))  # Only subtract DP Noise if adding it to begin with

INTERCEPTS_DP_NOISE = False  # Add DP noise to intercepts (for Logistic Regression example)
# DP_ALGORITHM: either Laplace or Gamma. Can easily add more in client agent's code

DP_ALGORITHM = "Gamma"

# DP Privacy Parameters
epsilon = 1.0  # smaller epsilon --> more noise/less accuracy
# can make each client's epsilon different if desired
EPSILONS = {client_name: epsilon for client_name in client_names}
alpha = 1
mean = 0

"""Client Dropout"""
# CLIENT_DROPOUT: When TRUE, clients drop out of simulation when personal weights are within tolerance of federated weights
CLIENT_DROPOUT = False
tolerance = 10.0  # note this value should change depending on whether you are normalizing

"""Latency"""
SIMULATE_LATENCIES = True
# Define any agent-agent communication latencies here. If none is provided, defaults to zero.


LATENCY_DICT = {}
# # fill in the rest with zeros:
if 'server_agent_0' not in LATENCY_DICT.keys():
    LATENCY_DICT['server_agent_0'] = {}

for client_name in client_names:
    if client_name not in LATENCY_DICT.keys():
        LATENCY_DICT[client_name] = {client_name2: timedelta(seconds=0.1) for client_name2 in client_names}
    LATENCY_DICT[client_name]['server_agent_0'] = timedelta(seconds=0.1)
    LATENCY_DICT['server_agent_0'][client_name] = timedelta(seconds=0.1)

LATENCY_DICT['client_agent_1'] = {client_name: timedelta(seconds=2.0) for client_name in client_names}
LATENCY_DICT['client_agent_1']['server_agent_0'] = timedelta(seconds=2.0)
LATENCY_DICT['server_agent_0']['client_agent_1'] = timedelta(seconds=2.0)

LATENCY_DICT['client_agent_0']['server_agent_0'] = timedelta(seconds=0.3)
LATENCY_DICT['server_agent_0']['client_agent_0'] = timedelta(seconds=0.3)

"""
Additional Constants: likely won't need modification
"""
# LOG_MAX_ITER: max iterations for the logistic regression
LOG_MAX_ITER = 10
random.seed(0)
# RANDOM_SEEDS: required for reproducibility of simulation. Seeds every iteration of the training for each client
RANDOM_SEEDS = {client_name: list(random.sample(range(0, 1000000), 100)) for client_name in client_names}
print(RANDOM_SEEDS)