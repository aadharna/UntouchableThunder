import torch
import ctypes
import numpy as np
from functools import reduce
from collections import OrderedDict
from utils.diff_evo import differential_evolution
# from devo import *

class PyTorchObjective():
    """PyTorch objective function, wrapped to be called by scipy.optimize."""

    def __init__(self, agent, bound_limits=(-5, 5), popsize=100):
        self.agent = agent
        # make an x0 from the parameters in this module
        parameters = OrderedDict(agent.nn.named_parameters())
        self.param_shapes = {n: parameters[n].size() for n in parameters}
        # ravel and concatenate all parameters to make x0
        self.x0 = np.concatenate([parameters[n].data.numpy().ravel()
                                  for n in parameters])
        
        self.eval_fn = agent.evaluate # produces a scalar loss when run that evalutes the nn
                                 # NEGATIVE VALUES ARE GOOD HERE. We're trying to minimize the loss surface.
            
        self.id = agent.name
        self.c = 0
        self.bounds = [bound_limits]*self.x0.shape[0]
        self.popsize = popsize
        
        
        
        _min = -1.0  # min
        _max = 1.0  # max
        dimension = self.x0.shape[0]
        
        
        self.init_fitnesses = (_max - _min) * np.random.uniform(size=popsize) + _min
        
        self.out_population = np.zeros((popsize, dimension), dtype='float32')
        self.out_fitnesses  = np.zeros((popsize, 1), dtype='float32')
        
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

    def pack_grads(self):
        """pack all the gradients from the parameters in the module into a
        numpy array."""
        grads = []
        for p in self.nn.parameters():
            grad = p.grad.data.numpy()
            grads.append(grad.ravel())
        return np.concatenate(grads)

    def is_new(self, x):
        # if this is the first thing we've seen
        if not hasattr(self, 'cached_x'):
            return True
        else:
            # compare x to cached_x to determine if we've been given a new input
            x, self.cached_x = np.array(x), np.array(self.cached_x)
            error = np.abs(x - self.cached_x)
            return error.max() > 1e-8

    def cache(self, x):
        # unpack x and load into module
        state_dict = self.unpack_parameters(x)
        self.agent.nn.load_state_dict(state_dict)
        # store the raw array as well
        self.cached_x = x
        # calculate the objective using x
        score = self.eval_fn()
        self.cached_score = -1 * score
        
        if self.c % 50:
            self.watching.append(self.cached_score)
        

    def fun(self, x, dimension):
        self.c += 1
        z = np.zeros(dimension, dtype=np.float32)
        for i in range(dimension):
            z[i] = x[i] + 0.
        if self.is_new(z):
            self.cache(z)
            
        return self.cached_score
    
    def fun_c(self, x, dimension):
        
        self.fun(x, dimension)
        return ctypes.c_double(self.cached_score)
        
        
    def update_nn(self, answer):
        state_dict = self.unpack_parameters(answer.x)
        self.agent.nn.load_state_dict(state_dict)

    def create_population(self):
        ## TALK WITH TAE JONG ABOUT A GOOD WAY TO ADD NOISE.
        _max, _min = 1., -1.
        self.init_population = (
            _max - _min) * np.random.uniform(size=(self.popsize, self.x0.shape[0])) + _min
        
          ## NOTE, we might want to change this back, in some manner. 
#         # get noise
#         noise = np.random.randn(self.popsize, self.x0.shape[0])
#         # add noise to theta for each member of the population
#         pop = [self.x0 + noise[i] for i in range(self.popsize)]

        self.init_population[0] = self.x0

        return self.init_population
    
    def results_callback(self, population, fitness_values, population_size, problem_size):
        # Store results to python memory containers
        # Store population
        for i in range(0, population_size * problem_size):
            row = int(i / problem_size)
            col = i % problem_size
            self.out_population[row][col] = population[i]

        # Store fitness values
        for j in range(0, population_size):
            f = fitness_values[j]
            self.out_fitnesses[j] = f
        return
        

    
    
def run_TJ_DE(_de, pair, n=3, popsize=99):
    """run Tae Jong's DE
    _de:  which DE algorithm do we want to use?
    pair: PyTorchObjective wrapper of paired NNAgent_Environment object
    n: number of function evaluations
    """
    return _de.run(
                    n,
                    pair.popsize,
                    0.5,
                    0.9,
                    pair.fun,
                    pair.x0.shape[0],
                    -5.0,
                    5.0,
                    pair.create_population().ctypes.data_as(c.POINTER(c.c_double)),
                    pair.init_fitnesses.ctypes.data_as(c.POINTER(c.c_double)),
                    pair.results_callback
                    )
    
def run_opt_n_steps(pair, n=3, popsize=99, strategy='rand1bin'):
    """run n steps on the evolution optimizer

    :param pair: PyTorchObjective wrapper of paired NNAgent_Environment object
    :param n: number of generations
    :param popsize: size of population - 1
    :param strategy: weight update strat
    :return: 'best' weights found
    """
    return differential_evolution(pair.fun, pair.bounds,
                                 strategy=strategy,
                                 popsize=popsize,
                                 maxiter=n,
                                 polish=False,
                                 x0=pair.x0)
