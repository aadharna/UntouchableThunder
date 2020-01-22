import ctypes as c
from utils.ppo import ppo
import devo.DE
import devo.SHADE
import devo.JADE
import devo.jDE
import devo.CoDE

#-------------------------OPTIMIZATION RUNNER FUNCTIONS---------------------------------#


def run_ppo(policy_agent, env_fn, path, 
            pair_id, 
            outer_poet_loop_count, 
            n_concurrent_games=1, 
            frames=100000):
    """Run Proximal Policy optimization:
       Adapted from: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py
       
       :param policy_network: Pass the NNagent directly
       :param env: pass in the outer GridGame env.make fn
       :param frames: number of frames you want ppo to run for. 
       
       return: optimized neural network
    """
    return ppo(pg            = policy_agent, 
               env_fn        = env_fn(), # this removes the outer function from GridGameObject.make()
               num_envs      = n_concurrent_games,
               path_to_runs  = path, 
               total_frames  = frames,
               pair_id       = pair_id,
               outer_poet_loop_count = outer_poet_loop_count)
    
    
def run_TJ_DE(opt_name, pair, n, pair_id,
              scaling_factor=0.5,
              crossover_rate=0.1,
              min_weight=-5,
              max_weight=5,
              ):
    """run Tae Jong's DE
    opt_name:  which DE algorithm do we want to use?. string. e.g. 'jDE'
    pair: PyTorchObjective wrapper of paired NNAgent_Environment object
    n: number of function evaluations
    """

    import devo
    _de = getattr(devo, opt_name)

    generations = n // pair.popsize
    x_c = pair.create_population().ctypes.data_as(c.POINTER(c.c_double))
    y_c = pair.init_fitnesses.ctypes.data_as(c.POINTER(c.c_double))
    scores = {}
    for g in range(generations):
        # # Using Adaptive-DEs
        _de.run(
            pair.popsize,
            pair.popsize,  # population size
            scaling_factor,  # scaling factor
            crossover_rate,  # crossover rate
            pair.fun_c,
            pair.x0.shape[0],  # problem size
            min_weight,  # unused value
            max_weight,  # unused value
            x_c,
            y_c,
            pair.results_callback  # no results callback needed
        )

        scores[g] = -pair.out_fitnesses.squeeze() if pair.minimize else pair.out_fitnesses.squeeze()

        x_c = pair.out_population.ctypes.data_as(
            c.POINTER(c.c_double))
        y_c = pair.out_fitnesses.ctypes.data_as(
            c.POINTER(c.c_double))

    # save scores
    import pandas
    df = pandas.DataFrame.from_dict(scores)
    df.to_csv(f'./results/{pair_id}/{opt_name}_{generations}gens_{pair.popsize}pop_scores.csv')

    return
