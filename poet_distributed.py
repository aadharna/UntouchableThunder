import os
import time
import gvgai
import numpy as np
from itertools import product

from utils.call_java_competition_agent import runJavaAgent
from utils.ADPParent import ADPParent
from utils.ADPTASK_ENUM import ADPTASK

from agent.models import Net
# from generator.env_gen_wrapper import GridGame
from generator.levels.base import Generator, _initialize
from agent.minimalPair import MinimalPair

from torch import save as torch_save
from torch import load as torch_load

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


def waitForAndCollectAnswers(parent, children, distributed_work, unique_run_id, poet_loop_counter, task):
    print('waiting for answers')
    resend = []
    answers_list = []
    time.sleep(10)
    while not parent.checkChildResponseStatus(children, resend):
        if resend:
            time.sleep(5)
            print(f"resending work {resend}")
            # save completed work so that child who gets second task
            # does not overwrite the first task.
            for (reassigned_from, reassigned_to) in resend:
                if not reassigned_from == reassigned_to:
                    answers_list.append(parent.readChildAnswer(f'answer{reassigned_to}.pkl'))
                    
            send_work({k[1]:distributed_work[k[0]] for k in resend}, task, parent, unique_run_id, poet_loop_counter)
            resend = []
        time.sleep(5)

    answer_pointers = os.listdir(os.path.join(
        parent.root,
        parent.subfolders['sent_by_child']
    ))

    answers_list.extend([parent.readChildAnswer(answer) for answer in answer_pointers])
    flat_answers = flatten(answers_list)

    print('collected answers')
    return flat_answers


def divideWorkBetweenChildren(agents, envs, children, transfer_eval=False):
    # private function to implement circular queue for assigning tasks
    def dispenseChild(children):
        num_children = len(children)
        for i in range(np.random.randint(0, num_children), 1000000):
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

    for agent, generator in agent_env_work_pair:
        id = next(dispenser)
        tasks[id]['env'].append(str(generator))
        tasks[id]['nn'].append(agent.nn.state_dict())
        tasks[id]['nn_id'].append(agent.id)
        tasks[id]['env_id'].append(generator.id)

    return tasks


def updatePairs(pairs, answers, task_type):
    """

    :param pairs: list of active NN-Env pairs
    :param answers: flattened by chromosome_id and env_id children_response dicts
    :param task_type: ADPTASK ID
    :return:
    """
    print("updating")
    # do something with the answers.
    # for each dict from the children

    for (xsome_id, env_id) in answers:
        # print(xsome_id)
        for each_pair in pairs:
            if xsome_id == each_pair.id:
                # print("found matching nn")
                each_pair.score = answers[(xsome_id, env_id)]['score']
                if task_type == ADPTASK.OPTIMIZE:
                    nn = answers[(xsome_id, env_id)]['nn']  # this is a state_dict
                    each_pair.nn.load_state_dict(nn)


def dieAndKillChildren(parent, pairs):

    # [pair.env.close() for pair in pairs]

    path = os.path.join(parent.root,
                        parent.subfolders['alive_signals'])

    alive = os.listdir(path)

    for a in alive:
        os.remove(os.path.join(path, a))


def perform_transfer(pairs, answers, poet_loop_counter, unique_run_id):
    """
    find the network which performed best in each env.
    Move that best-network into that env.

    Eval agent j in env k.
    Find best agent, a for each env
    Move agent a into env k

    :param pairs: agent-env pairs
    :param answers: flattened answers index by (agent.id, env.id)
    :param poet_loop_counter: int counter
    :return:
    """

    for k, fixed_env_pair in enumerate(pairs):
        current_score = answers[(fixed_env_pair.id, fixed_env_pair.generator.id)]['score']
        current_net = fixed_env_pair.nn.state_dict()
        transferred_id = fixed_env_pair.id
        # for every other network, evaluate environment k in agent j
        for j, changing_agent_pair in enumerate(pairs):
            if k == j:
                continue
            else:
                j_score = answers[(changing_agent_pair.id, fixed_env_pair.generator.id)]['score']

                if args.transfer_mc:
                    if not answers[(changing_agent_pair.id,
                                    changing_agent_pair.generator.id)]['won']:
                        continue

                # todo talk about <=?
                if current_score < j_score:
                    # updated network
                    print(f"update network {fixed_env_pair.id} to {changing_agent_pair.id}")
                    current_score = j_score
                    current_net   = changing_agent_pair.nn.state_dict()
                    transferred_id = changing_agent_pair.id

        #transfer into environment, k, the agent, j, who performed the best.
        if not fixed_env_pair.id == transferred_id:
            fixed_env_pair.nn.load_state_dict(current_net)

            # todo talk with lisa about if
            # fixed_env_pair.id = transferred_id ? It's not clear to me.

            with open(os.path.join(f'{args.result_prefix}/results_{unique_run_id}/{fixed_env_pair.id}',
                                   f'poet{poet_loop_counter}_network_{transferred_id}_transferred_here.txt'),
                      'w+') as fname:
                pass

def pass_mc(gridGame):
    print("testing MC")
    
    path_to_game = f'./ext/GVGAI_GYM/games/{args.game}_v0/{args.game}.txt'
    print("running mcts agent")
    # if you LOSE with a tree-serach agent, it's too hard.
    wonGameMCTS = runJavaAgent('runGVGAI.jar', 
                               path_to_game,
                               gridGame.path_to_file,
                               args.comp_agent,
                               str(args.game_len),
                               )

    print("running random agent")
    # if you WIN playing randomly, the level is too easy.
    wonGameRandomly = runJavaAgent('runGVGAI.jar',
                                   path_to_game,
                                   gridGame.path_to_file,
                                   'random',
                                   str(args.game_len))

    # if not too easy and not too hard:
    if not wonGameRandomly and wonGameMCTS:
        return True

    return False

def get_child_list(parent_list, max_children, unique_run_id):
    child_list = []

    mutation_trial = 0
    while mutation_trial < max_children:
        print(f"mutation_trial {mutation_trial + 1}/{max_children}")
        parent = np.random.choice(parent_list)

        new_gen = parent.mutate(mutationRate=args.mutation_rate,
                                minimal=args.minimal_mutation,
                                r=args.mutation_radius)

        mutation_trial += 1

        if pass_mc(new_gen):
            child_list.append(MinimalPair(unique_run_id=unique_run_id,
                                          generator=new_gen,
                                          prefix=args.result_prefix,
                                          parent=parent.nn))

            tag = os.path.join(f'{args.result_prefix}',
                               f'results_{unique_run_id}',
                               f'{child_list[-1].id}/parent_is_{parent.id}.txt')
            with open(tag, 'w+') as fname:
                pass

    # speciation or novelty goes here
    #
    return child_list

def send_work(distributed_work, task, parent, unique_run_id, poet_loop_counter):
    
    for worker_id in distributed_work:

        parent.createChildTask(run_id=unique_run_id,
                               work_dict=distributed_work[worker_id],
                               worker_id=worker_id,
                               task_id=task,
                               poet_loop_counter=i,
                               rl=args.rl,
                               algo=args.DE_algo,
                               ngames=args.n_games,
                               popsize=args.popsize)



####################### HELPER FUNCTIONS ##########################

# ARGUMENTS TO THE SCRIPT

import argparse
from utils.loader import load_from_yaml
parser = argparse.ArgumentParser()
parser.add_argument("--args_file", type=str, default='./args.yml', help='path to args file')
parser.add_argument("--exp_name", type=str, default='exp1', help='exp name')
# parser.add_argument("--game", type=str, default='dzelda', help='set gvgai game')
# parser.add_argument("--lvl_dir", type=str, default='./levels', help='path to lvl dir')
# parser.add_argument("--init_lvl", type=str, default='start.txt', help='level from ./levels folder')
# parser.add_argument("--game_len", type=int, default=250, help='game length')
# parser.add_argument("--n_games", type=int, default=1000, help='opt length in num games')
# parser.add_argument("--rl", type=bool, default=False, help='use RL?')
# parser.add_argument("--DE_algo", type=str, default='CoDE', help='which DE algo to use if rl is False?')
# parser.add_argument("--mutation_timer", type=int, default=5, help='steps until mutation')
# parser.add_argument("--mutation_rate", type=float, default=0.75, help='change of mutation')
# parser.add_argument("--transfer_timer", type=int, default=15, help='steps until transfer')
# parser.add_argument("--max_children", type=int, default=8, help='number of children to add each transfer step')
# parser.add_argument("--max_envs", type=int, default=50, help='max number of GVGAI-gym envs allowed at any one time')
# parser.add_argument("--comp_agent", type=str, default="mcts", help="what gvgai comp should be used for MC?")
# parser.add_argument("--num_poet_loops", type=int, default=10, help="How many POET loops to run")
# parser.add_argument("--result_prefix", type=str, default='.', help="prefix of where to place results folder")
# parser.add_argument("--start_fresh", type=bool, default=True, help="start from scratch or pick up from previous session")
#
_args = parser.parse_args()
args = load_from_yaml(_args.args_file)
print(args)
print(__name__)

############### POET ###############

if __name__ == "__main__":

    parent = ADPParent()
    unique_run_id = _args.exp_name
    net = Net(6, 13)
    net.load_state_dict(torch_load('./start.pt'))

    lvl = _initialize(os.path.join(args.lvl_dir, f"{args.game}_{args.init_lvl}"), d=args.shape0)
    lvl_shape = lvl.shape
    generator = Generator(game=args.game, 
                          tile_world=lvl,
                          shape=lvl.shape,
                          path=args.lvl_dir,
                          mechanics=args.mechanics,
                          generation=0)
    generator.to_file(0, args.game)
    
    pairs = [MinimalPair(unique_run_id=unique_run_id,
                         game=args.game,
                         generator=generator,
                         parent=net,
                         prefix=args.result_prefix)
            ]

    done = False
    i = 0
    chkpt = f"{args.result_prefix}/results_{unique_run_id}/POET_CHKPT"
    if not os.path.exists(chkpt):
        os.mkdir(chkpt)
    
    time.sleep(20)
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
                                                         [pairs[i].generator for i in range(len(pairs))],
                                                         availableChildren)

            print("evaluating")
            send_work(distributed_work, ADPTASK.EVALUATE, parent, unique_run_id, i)
            
            # get answers from children
            eval_answers = waitForAndCollectAnswers(parent, availableChildren, distributed_work, unique_run_id, i,  ADPTASK.EVALUATE)

            updatePairs(pairs, eval_answers, ADPTASK.EVALUATE)

            # Add in new children
            #
            new_envs = []
            print("mutation?")
            if i % args.mutation_timer == 0:
                print("yes")
                new_envs = get_child_list(pairs, args.max_children, unique_run_id)

            pairs.extend(new_envs)
            del new_envs # this does not delete the children that have now been placed in pairs.
            # print(len(pairs))

            # kill extra population.
            #
            if len(pairs) > args.max_envs:
                aged_pairs = sorted(pairs, key=lambda x: x.id, reverse=True)
                pairs = aged_pairs[:args.max_envs]
                del aged_pairs
            
            # Optimizations step
            #
            print("optimizing")
            distributed_work = divideWorkBetweenChildren(pairs,
                                                         [pairs[i].generator for i in range(len(pairs))],
                                                         availableChildren)

            send_work(distributed_work, ADPTASK.OPTIMIZE, parent, unique_run_id, i)

            # get answers from children
            opt_answers = waitForAndCollectAnswers(parent, availableChildren, distributed_work, unique_run_id, i, ADPTASK.OPTIMIZE)

            updatePairs(pairs, opt_answers, ADPTASK.OPTIMIZE)

            # TRANSFER NNs between ENVS,
            # EVALUATE each NN with each ENV.
            #
            if (i + 1) % args.transfer_timer == 0:
                print("transferring")
                distributed_work = divideWorkBetweenChildren(pairs,
                                                             [pairs[i].generator for i in range(len(pairs))],
                                                             availableChildren,
                                                             transfer_eval=True)

                send_work(distributed_work, ADPTASK.EVALUATE, parent, unique_run_id, i)

                # get answers from children
                transfer_eval_answers = waitForAndCollectAnswers(parent, availableChildren, distributed_work, unique_run_id, i, ADPTASK.EVALUATE)

                # use information to determine if NN i should migrate to env j.
                perform_transfer(pairs, transfer_eval_answers, i, unique_run_id)

            # save checkpoints of networks into POET folder
            #
            for pair in pairs:
                torch_save(pair.nn.state_dict(), os.path.join(chkpt,
                                                              str(i),
                                                              f'network{pair.id}.pt'))

            i += 1
            if i >= args.num_poet_loops:
                done = True

        except KeyboardInterrupt as e:
            print(e)
            dieAndKillChildren(parent, pairs)
            import sys
            sys.exit(0)

    dieAndKillChildren(parent, pairs)
