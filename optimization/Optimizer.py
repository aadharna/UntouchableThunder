import torch
import numpy as np
from functools import reduce
from collections import OrderedDict
from utils.diff_evo import differential_evolution

class PyTorchObjective(object):
    """PyTorch objective function, wrapped to be called by scipy.optimize."""

    def __init__(self, agent, bound_limits=(-5, 5)):
        self.f = agent.nn  # some pytorch module
        # make an x0 from the parameters in this module
        parameters = OrderedDict(agent.nn.named_parameters())
        self.param_shapes = {n: parameters[n].size() for n in parameters}
        # ravel and concatenate all parameters to make x0
        self.x0 = np.concatenate([parameters[n].data.numpy().ravel()
                                  for n in parameters])

        self.eval_fn = agent.evaluate  # produces a scalar loss when run that evalutes the nn
        self.id = agent.name
        self.c = 0
        self.bounds = [bound_limits]*self.x0.shape[0]

    def unpack_parameters(self, x):
        """optimize.minimize will supply 1D array, chop it up for each parameter."""
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
        for p in self.f.parameters():
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
        self.f.load_state_dict(state_dict)
        # store the raw array as well
        self.cached_x = x
        # calculate the objective using x
        score = self.eval_fn()
        self.cached_score = score

    def fun(self, x):
        self.c += 1
        if self.is_new(x):
            self.cache(x)
        if self.c % 150 == 0:
            print(f"achieved score of: {self.cached_score} on {self.c}")
        return self.cached_score

    def update_nn(self, answer):
        state_dict = self.unpack_parameters(answer.x)
        self.f.load_state_dict(state_dict)


def run_opt_n_steps(pair, n=3, popsize=99, strategy='rand1bin'):
    """run n steps on the evolution optimizer

    :param pair: NNAgent object
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
