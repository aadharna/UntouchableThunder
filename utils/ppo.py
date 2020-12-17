import os
import random
import time

import gym
from gym.wrappers import TimeLimit, Monitor
from gym.spaces import Discrete, Box, MultiBinary, MultiDiscrete, Space
# import gvgai


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

import numpy as np
from utils.utils import zelda_spaces

from generator.env_gen_wrapper import GridGame
from agent.NNagent import NNagent
from agent.models import Value

# from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

def preprocess_obs_space(obs_space: Space, device: str):
    """
    The `preprocess_obs_fn` receives the observation `x` in the shape of
    `(batch_num,) + obs_space.shape`.
    1) If the `obs_space` is `Discrete`, `preprocess_obs_fn` outputs a
    preprocessed obs in the shape of
    `(batch_num, obs_space.n)`.
    2) If the `obs_space` is `Box`, `preprocess_obs_fn` outputs a
    preprocessed obs in the shape of
    `(batch_num,) + obs_space.shape`.
    In addition, the preprocessed obs will be sent to `device` (either
    `cpu` or `cuda`)
    """
    if isinstance(obs_space, Discrete):
        def preprocess_obs_fn(x):
            return F.one_hot(torch.LongTensor(x), obs_space.n).float().to(device)
        return (obs_space.n, preprocess_obs_fn)

    elif isinstance(obs_space, Box):
        def preprocess_obs_fn(x):
            return torch.Tensor(x).float().view(torch.Tensor(x).shape[0], -1).to(device)
        return (np.array(obs_space.shape).prod(), preprocess_obs_fn)

    else:
        raise NotImplementedError("Error: the model does not support input space of type {}".format(
            type(obs_space).__name__))

def preprocess_ac_space(ac_space: Space):
    if isinstance(ac_space, Discrete):
        return ac_space.n

    elif isinstance(ac_space, MultiDiscrete):
        return ac_space.nvec.sum()

    elif isinstance(ac_space, Box):
        return np.prod(ac_space.shape)

    else:
        raise NotImplementedError("Error: the model does not support output space of type {}".format(
            type(ac_space).__name__))
        
        
        
def ppo(pg,              # NNagent! This is for the rl_get_action fn.
        env_fn,          # gvgai-gym env fn, specifically a GridGame obj.
        path_to_runs,    # path to store run information
        pair_id,
        outer_poet_loop_count,
        num_envs=1,
        cuda=False,
        lr=4e-4, 
        seed=42, 
        episode_len=250, 
        total_frames=100000, 
        gamma=0.99, 
        value_coeff=0.25, 
        grad_norm=0.5,
        ent_coeff=0.01,
        clip=0.2, 
        epoch_update=3):
    """
    """
    if num_envs > 1:
        raise NotImplementedError("Using vectorized envs is not yet implemented")
            # for vectorized version, use this.
#         gyms  = [env_fn for _ in tqdm(range(num_envs))]                   #GridGame envs
#         objs  = [NNagent(GG=None, parent=pg.nn) 
#                  for _ in tqdm(range(solver.popsize))]                    #Objective(NNagents)

#         env = SubprocVecEnv(gyms, 
#                              spaces=zelda_spaces, 
#                              context='fork')
          
    else:    
        env = env_fn() #This calls the inner function that actually makes the env!
    
    # set up 
    experiment_name = f"{env.game}__pair{pair_id}__loop{outer_poet_loop_count}__{int(time.time())}"
    writer = SummaryWriter(os.path.join(path_to_runs, experiment_name))

    if torch.cuda.is_available():
        place = 'cuda' 
    else: 
        place = 'cpu'
        
    device = torch.device(place)
                       
    # set seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # set up value network
    vf = Value(env.depth)
    vf.double()
    optimizer = optim.Adam(list(pg.nn.parameters()) + list(vf.parameters()), lr=lr)
    loss_fn = nn.MSELoss()
    
    # TRY NOT TO MODIFY: start the game
    global_step = 0
    while global_step < total_frames:
        next_obs = np.array(env.reset())
        actions = np.empty((episode_len,), dtype=object)
        rewards, dones = np.zeros((2, episode_len))
        obs = np.empty((episode_len,) + zelda_spaces[0].shape)
        compass_info = np.empty((episode_len, 4))

        # ALGO LOGIC: put other storage logic here
        values = torch.zeros((episode_len), device=device)
        neglogprobs = torch.zeros((episode_len,), device=device)
        entropys = torch.zeros((episode_len,), device=device)

        # TRY NOT TO MODIFY: prepare the execution of the game.
        for step in range(episode_len):
            global_step += 1
            obs[step] = next_obs.copy()

            compass_info[step] = pg.compass_info

            # ALGO LOGIC: put action logic here
            values[step] = vf.forward(torch.DoubleTensor(obs[step:step+1]), 
                                      torch.DoubleTensor(compass_info[step:step+1]))

            # ALGO LOGIC: `env.action_space` specific logic
            action, neglogprob, entropy = pg.rl_get_action(torch.DoubleTensor(obs[step:step+1]), 
                                                           torch.DoubleTensor(compass_info[step:step+1]))

            actions[step], neglogprobs[step], entropys[step] = action.item(), neglogprob, entropy

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, rewards[step], dones[step], _ = env.step(actions[step])
            next_obs = np.array(next_obs)
            if dones[step]:
                break

        # ALGO LOGIC: training.
        # calculate the discounted rewards, or namely, returns
        returns = np.zeros_like(rewards)
        for t in reversed(range(rewards.shape[0]-1)):
            returns[t] = rewards[t] + gamma * returns[t+1] * (1-dones[t])
        # advantages are returns - baseline, value estimates in our case
        advantages = returns - values.detach().cpu().numpy()

        neglogprobs = neglogprobs.detach()
        non_empty_idx = np.argmax(dones) + 1
        for _ in range(epoch_update):
            # ALGO LOGIC: `env.action_space` specific logic
            _, new_neglogprobs, _ = pg.rl_get_action(torch.DoubleTensor(obs[:non_empty_idx]), 
                                                     torch.DoubleTensor(compass_info[:non_empty_idx]))

            ratio = torch.exp(neglogprobs[:non_empty_idx] - new_neglogprobs)
            surrogate1 = ratio * torch.Tensor(advantages)[:non_empty_idx]
            surrogate2 = torch.clamp(ratio, 1-clip, 1+clip) * torch.Tensor(advantages)[:non_empty_idx]
            clip_value = torch.min(surrogate1, surrogate2)
            vf_loss = loss_fn(torch.Tensor(returns), values) * value_coeff
            loss = vf_loss - (clip_value + entropys[:non_empty_idx] * ent_coeff).mean()

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(list(pg.nn.parameters()) + list(vf.parameters()), grad_norm)
            optimizer.step()

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/episode_reward", rewards.sum(), global_step)
        writer.add_scalar("losses/value_loss", vf_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropys[:step].mean().item(), global_step)
    
    writer.close()
    env.close()
    
    return 