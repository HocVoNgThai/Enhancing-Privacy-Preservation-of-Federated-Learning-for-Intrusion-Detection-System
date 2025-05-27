from numbers import Real
import random
import numpy as np

def laplace(mean, sensitivity, epsilon): # mean : value to be randomized (mean)
        scale = sensitivity / epsilon
        rand = random.uniform(0,1) - 0.5 # rand : uniform random variable
        return mean - scale * np.sign(rand) * np.log(1 - 2 * np.abs(rand))