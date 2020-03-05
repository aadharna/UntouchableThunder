import os
import gc
import ctypes as c
import pandas
from utils.ppo import ppo
from utils.CoDE import CoDE
from utils.DE import DE
#import devo
#import devo.DE
#import devo.SHADE
#import devo.JADE
#import devo.jDE
#import devo.CoDE

#-------------------------OPTIMIZATION RUNNER FUNCTIONS---------------------------------#


def run_ppo(policy_agent, env_fn, path, 
            pair_id, 
            outer_poet_loop_count, 
            n_concurrent_games=1, 
            frames=100000):
    """Run Proximal Policy optimization:
       Adapted from: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py
       
       :param policy_agent: Pass the NNagent directly
       :param env_fn: pass in the outer GridGame env.make fn
       :param path: path to director to store ppo graphs in. ./runs
       :param frames: number of frames you want ppo to run for.
       :param outer_poet_loop_count: poet number loop
       :param n_concurrent_games: number of vectorized games


       
       return: optimized neural network
    """
    return ppo(pg            = policy_agent, 
               env_fn        = env_fn(), # this removes the outer function from GridGameObject.make()
               num_envs      = n_concurrent_games,
               path_to_runs  = path, 
               total_frames  = frames,
               pair_id       = pair_id,
               outer_poet_loop_count = outer_poet_loop_count)

# from memory_profiler import profile
# @profile
def run_CoDE(AE_pair,
             results_prefix, unique_run_id, pair_id, poet_loop_counter,
             generation_max=5,
             scaling_factor=0.5, #list of three numbers
             crossover_rate=0.5, #list of three numbers
             lower_bound=-5,
             upper_bound=5):
    
    scores = {}
    ans =  CoDE(pair=AE_pair,
                init_population=AE_pair.init_population,
                problem_size=AE_pair.x0.shape[0],
                scaling_factor=scaling_factor,
                crossover_rate=crossover_rate,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                generation_max=generation_max,
                scores=scores)

    destination = os.path.join(f'{results_prefix}',
                               f'{pair_id}',
                               f'poet{poet_loop_counter}_{generation_max}gens_{AE_pair.popsize}pop_scores.csv')
    
    df = pandas.DataFrame.from_dict(scores)    
    df.to_csv(destination)
    return ans

def run_DE(AE_pair,
             results_prefix, unique_run_id, pair_id, poet_loop_counter,
             generation_max=5,
             scaling_factor=0.5, #list of three numbers
             crossover_rate=0.5, #list of three numbers
             lower_bound=-5,
             upper_bound=5):
    
    scores = {}
    ans =  DE(pair=AE_pair,
                init_population=AE_pair.init_population,
                problem_size=AE_pair.x0.shape[0],
                scaling_factor=scaling_factor,
                crossover_rate=crossover_rate,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                generation_max=generation_max,
                scores=scores)

    destination = os.path.join(f'{results_prefix}',
                               f'{pair_id}',
                               f'poet{poet_loop_counter}_{generation_max}gens_{AE_pair.popsize}pop_scores.csv')
    
    df = pandas.DataFrame.from_dict(scores)    
    df.to_csv(destination)
    return ans


# #from memory_profiler import profile
# #@profile
def run_TJ_DE(opt_name, pair, n, pair_id, poet_loop_counter,
              results_prefix,
              unique_run_id,
              scaling_factor=0.5,
              crossover_rate=0.1,
              min_weight=-5,
              max_weight=5,
              ):
#     """run Tae Jong's DE
#     opt_name:  which DE algorithm do we want to use?. string. e.g. 'jDE'
#     pair: PyTorchObjective wrapper of paired NNAgent_Environment object
#     n: number of function evaluations
#     """
#
#     _de = getattr(devo, opt_name)
#
#     generations = n // pair.popsize
#     x_c = pair.init_population.ctypes.data_as(c.POINTER(c.c_double))
#     y_c = pair.init_fitnesses.ctypes.data_as(c.POINTER(c.c_double))
#     scores = {}
#     for g in range(generations):
#         # # Using Adaptive-DEs
#         _de.run(
#             pair.popsize,
#             pair.popsize,  # population size
#             scaling_factor,  # scaling factor
#             crossover_rate,  # crossover rate
#             pair.obj_fun_c,     # objective function wrapper; sends back C-intelligible results
#             pair.x0.shape[0],  # problem size
#             min_weight,  # unused value
#             max_weight,  # unused value
#             x_c,
#             y_c,
#             pair.results_callback  # no results callback needed
#         )
#
#         scores[g] = -pair.out_fitnesses.squeeze() if pair.minimize else pair.out_fitnesses.squeeze()
#
#         x_c = pair.out_population.ctypes.data_as(
#             c.POINTER(c.c_double))
#         y_c = pair.out_fitnesses.ctypes.data_as(
#             c.POINTER(c.c_double))
#
#     # save scores
#     df = pandas.DataFrame.from_dict(scores)
#
#     destination = os.path.join(f'{results_prefix}',
#                                f'{pair_id}',
#                                f'poet{poet_loop_counter}_{generations}gens_{pair.popsize}pop_scores.csv')
#
#     df.to_csv(destination)
#
#     del scores, df, x_c, y_c
#     gc.collect()
#
    return
