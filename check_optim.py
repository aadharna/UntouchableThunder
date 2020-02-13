import torch
import torch.nn.functional as F
import numpy as np
from optimization.Optimizer import PyTorchObjective

from copy import deepcopy

import os
import time
import gym
import gvgai
from generator.levels.base import Generator

from agent.NNagent import NNagent
from agent.base import Agent
from agent.models import Net

from generator.env_gen_wrapper import GridGame

from utils.ADPParent import ADPParent
from utils.ADPTASK_ENUM import ADPTASK

from torch import save as torch_save
from torch import load as torch_load

######################## HELPERS ########################

def callOut(parent):
    print("calling out")
    children = []
    while len(children) < 1:
        try:
            time.sleep(5)
        except KeyboardInterrupt as e:
            print(e)
            import sys
            sys.exit(0)
        children = parent.pickupChildren()
    return children

def flatten(answer_list):
    f_dict = {}
    for dicts in answer_list:
        for k, v in dicts.items():
            for experiment in v:
                f_dict[(experiment['chromosome_id'], experiment['env_id'])] = experiment
    return f_dict


def waitForAndCollectAnswers(parent, children):
    print('waiting for answers')

    while not parent.checkChildResponseStatus(children):
        time.sleep(5)

    answer_pointers = os.listdir(os.path.join(
        parent.root,
        parent.subfolders['sent_by_child']
    ))

    answers = [parent.readChildAnswer(answer) for answer in answer_pointers]

    flat_answers = flatten(answers)

    print('collected answers')
    return flat_answers


def divideWorkBetweenChildren(agents, envs, children, transfer_eval=False):
    # private function to implement circular queue for assigning tasks
    def dispenseChild(children):
        num_children = len(children)
        for i in range(1000000):
            yield children[i % num_children]

    dispenser = dispenseChild(children)
    tasks = {}
    for _ in range(len(children)):
        id = next(dispenser)
        tasks[id] = {}
        tasks[id]['nn'] = []
        tasks[id]['env'] = []
        tasks[id]['nn_id'] = []
        tasks[id]['env_id'] = []

                                                                    # itertools product
    agent_env_work_pair = zip(agents, envs) if not transfer_eval else product(agents, envs)

    for agent, env in agent_env_work_pair:
        id = next(dispenser)
        tasks[id]['env'].append(str(env.generator))
        tasks[id]['nn'].append(agent.nn.state_dict())
        tasks[id]['nn_id'].append(agent.id)
        tasks[id]['env_id'].append(env.id)

    return tasks

def dieAndKillChildren(parent, pairs):

    [pair.env.close() for pair in pairs]

    path = os.path.join(parent.root,
                        parent.subfolders['alive_signals'])

    alive = os.listdir(path)

    for a in alive:
        os.remove(os.path.join(path, a))


        
        
############# ARGUMENTS ###############

import argparse
from utils.loader import load_from_yaml
parser = argparse.ArgumentParser()
parser.add_argument("--args_file", type=str, default='./args.yml', help='path to args file')
parser.add_argument("--exp_name", type=str, default='de_exp', help='set exp name')

_args = parser.parse_args()
args = load_from_yaml(_args.args_file)


################## CODE FOR TEST ##########################


if __name__ == "__main__":
    # parent handles communcation between itself and workers
    parent = ADPParent()
    # get experiment name
    unique_run_id = _args.exp_name
    
    # load starter net
    net = Net(6, 13)
    net.load_state_dict(torch_load('./start.pt'))
    
    
    # build simulations!
    pairs = [NNagent(time_stamp=unique_run_id,
                     prefix=args.result_prefix,
                     GG=GridGame(game=args.game,
                                play_length=args.game_len,
                                path='./levels',
                                lvl_name=args.init_lvl,
                                mechanics=['+', 'g', '1', '2', '3', 'w'],
                                # monsters, key, door, wall
                                ),
                     parent=net
                   ), 
             
             NNagent(time_stamp=unique_run_id,
                     prefix=args.result_prefix,
                     GG=GridGame(game=args.game,
                                play_length=args.game_len,
                                path='./levels',
                                lvl_name=args.init_lvl,
                                mechanics=['+', 'g', '1', '2', '3', 'w'],
                                # monsters, key, door, wall
                                ),
                     parent=net
                   )
            ]

    done = False
    i = 0
    chkpt = f"{args.result_prefix}/results_{unique_run_id}/POET_CHKPT"
    if not os.path.exists(chkpt):
        os.mkdir(chkpt)

    while not done:
        try:
            tdir = os.path.join(chkpt, str(i))
            if not os.path.exists(tdir):
                os.mkdir(tdir)
            
            # check if children are alive
            children = callOut(parent)
            print(children)

            # get available children
            availableChildren = parent.isChildAvailable(children)

            # if list is empty, wait and check again
            while not bool(availableChildren):
                time.sleep(5)
                availableChildren = parent.isChildAvailable(children)

            distributed_work = divideWorkBetweenChildren(pairs,  #  agents. We're not going to use the paired envs
                                                         [pairs[i].env for i in range(len(pairs))],
                                                         availableChildren)

            print("optimizing with DE")
            for worker_id in distributed_work:

                parent.createChildTask(run_id=unique_run_id,
                                       work_dict=distributed_work[worker_id],
                                       worker_id=worker_id,
                                       task_id=ADPTASK.OPTIMIZE,
                                       poet_loop_counter=i,
                                       rl=False,
                                       algo=args.DE_algo,
                                       ngames=5000)

            # get answers from children
            eval_answers = waitForAndCollectAnswers(parent, availableChildren)
            
            updatePairs(pairs, eval_answers, ADPTASK.OPTIMIZE)
            
            done = True
        
        except KeyboardInterrupt as e:
            print(e)
            dieAndKillChildren(parent, pairs)
            import sys
            sys.exit(0)
            
    dieAndKillChildren(parent, pairs)

