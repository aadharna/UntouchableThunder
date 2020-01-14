from utils.diff_evo import differential_evolution
from utils.ppo import ppo
from devo import *

#-------------------------OPTIMIZATION RUNNER FUNCTIONS---------------------------------#
    
def run_ppo(policy_agent, env_fn, path, n_concurrent_games=1, frames=100000):
    """Run Proximal Policy optimization:
       Adapted from: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py
       
       :param policy_network: Pass the NNagent directly
       :param env: pass in the outer GridGame env.make fn
       :param frames: number of frames you want ppo to run for. 
       
       return: optimized neural network
    """
    return ppo(pg            = policy_agent, 
               env_fn        = env_fn(), # this removes the other function from GridGameObject.make()
               num_envs      = n_concurrent_games,
               path_to_runs  = path, 
               total_frames  = frames)
    
    
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
                    0.1,
                    pair.fun,
                    pair.x0.shape[0],
                    -5.0,
                    5.0,
                    pair.create_population().ctypes.data_as(c.POINTER(c.c_double)),
                    pair.init_fitnesses.ctypes.data_as(c.POINTER(c.c_double)),
                    pair.results_callback
                    )