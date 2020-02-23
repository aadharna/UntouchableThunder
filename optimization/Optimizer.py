import torch
import ctypes
import numpy as np
from functools import reduce
from collections import OrderedDict

class PyTorchObjective():
    """PyTorch objective function, wrapped to be called by scipy.optimize."""

    def __init__(self, agent, popsize=100, minimize=True):
        self.agent = agent
        # make an x0 from the parameters in this module
        parameters = OrderedDict(agent.nn.named_parameters())
        self.param_shapes = {n: parameters[n].size() for n in parameters}
        # ravel and concatenate all parameters to make x0
        self.x0 = np.concatenate([parameters[n].data.numpy().ravel()
                                  for n in parameters])
        
        self.eval_fn = agent.evaluate # produces a scalar loss when run that evalutes the nn
                                 # NEGATIVE VALUES ARE GOOD HERE. We're trying to minimize the loss surface.

        self.minimize = minimize
        self.c = 0
        self.popsize = popsize
        
        self.best_score = np.inf if minimize else -np.inf
        self.best_individual = self.x0
        
        _min = -1.0  # min
        _max = 1.0  # max
        dimension = self.x0.shape[0]        
        
        self.init_fitnesses = (_max - _min) * np.random.uniform(size=popsize) + _min
        self.init_population = (
            _max - _min) * np.random.uniform(size=(self.popsize, dimension)) + _min
        
        self.out_population = np.zeros((popsize, dimension), dtype='float64')
        self.out_fitnesses  = np.zeros((popsize, 1), dtype='float64')
        

        
          ## NOTE, we might want to change init_pop back, in some manner. 
#         # get noise
#         noise = np.random.randn(self.popsize, self.x0.shape[0])
#         # add noise to theta for each member of the population
#         pop = self.x0 + noise

        self.init_population[0] = self.x0
        
        
        self.watching = []
        

    def unpack_parameters(self, x):
        """will be supplied a 1D array, chop it up for each parameter."""
        i = 0
        named_parameters = OrderedDict()
        for n in self.param_shapes:
            param_len = reduce(lambda x, y: x * y, self.param_shapes[n])
            # slice out a section of this length
            param = x[i:i + param_len]
            # reshape according to this size, and cast to torch
            param = param.reshape(*self.param_shapes[n])
            named_parameters[n] = torch.from_numpy(param)
            # update index
            i += param_len
        return named_parameters

    # def is_new(self, x):
    #     # if this is the first thing we've seen
    #     if not hasattr(self, 'cached_x'):
    #         return True
    #     else:
    #         # compare x to cached_x to determine if we've been given a new input
    #         x, self.cached_x = np.array(x), np.array(self.cached_x)
    #         error = np.abs(x - self.cached_x)
    #         return error.max() > 1e-8

    def cache(self, x):
        # unpack x and load into module
        state_dict = self.unpack_parameters(x)
        self.agent.nn.load_state_dict(state_dict)
        # store the raw array as well
        self.cached_x = x
        # calculate the objective using x
        score = self.eval_fn()
        self.cached_score = score

        if self.minimize:
            self.cached_score = -1 * self.cached_score
        

    def obj_fun(self, x, dimension):
        self.c += 1
        self.cache(z) 
        return self.cached_score
    
    def obj_fun_c(self, x, dimension):
        z = np.zeros(dimension, dtype=np.float64)
        for i in range(dimension):
            z[i] = np.float64(x[i])
            
        score = self.fun(z, dimension)
        
        # add global best variables
        if (score < self.best_score and self.minimize) or \
           (score > self.best_score and not self.minimize):
            self.best_score = score
            self.best_individual = self.cached_x
            
        return ctypes.c_double(self.cached_score)
        
        
    def update_nn(self, weights):
        state_dict = self.unpack_parameters(weights)
        self.agent.nn.load_state_dict(state_dict)
    
    def results_callback(self, population, fitness_values, population_size, problem_size):

        # Store results to python memory containers
        # Store population
        for i in range(0, population_size * problem_size):
            row = i // problem_size
            col = i % problem_size
            self.out_population[row][col] = np.float64(population[i])

        # Store fitness values
        for j in range(0, population_size):
            f = fitness_values[j]
            self.out_fitnesses[j] = np.float64(f)
        return

