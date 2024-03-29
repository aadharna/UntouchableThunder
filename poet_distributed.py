import os
import time
import numpy as np
from itertools import product

from utils.call_java_competition_agent import runJavaAgent
from utils.dZeldaCharacterization import getdZeldaLvlCharacterization
from utils.ADPParent import ADPParent
from utils.ADPTASK_ENUM import ADPTASK

from agent.models import Net
from generator.levels.EvolutionaryGenerator import EvolutionaryGenerator
from generator.levels.IlluminatingGenerator import IlluminatingGenerator
from generator.levels.base import _initialize
from agent.minimalPair import MinimalPair

from torch import save as torch_save
from torch import load as torch_load

from tqdm import tqdm

def callOut(parent):
    # print("calling out")
    children = []
    while len(children) < 1:
        try:
            time.sleep(2)
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
    # print('waiting for answers')
    resend = []
    answers_list = []
    time.sleep(2)
    while not parent.checkChildResponseStatus(children, resend):
        if resend:
            time.sleep(2)
            # print(f"resending work {resend}")
            # save completed work so that child who gets second task
            # does not overwrite the first task.
            for (reassigned_from, reassigned_to) in resend:
                if not reassigned_from == reassigned_to:
                    answers_list.append(parent.readChildAnswer(f'answer{reassigned_to}.pkl'))
                    
            send_work({k[1]:distributed_work[k[0]] for k in resend}, task, parent, unique_run_id, poet_loop_counter)
            resend = []
        time.sleep(2)

    answer_pointers = os.listdir(os.path.join(
        parent.root,
        parent.subfolders['sent_by_child']
    ))

    answers_list.extend([parent.readChildAnswer(answer) for answer in answer_pointers])
    flat_answers = flatten(answers_list)

    # print('collected answers')
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
        tasks[id]['diff'] = []

                                                                    # itertools product
    agent_env_work_pair = zip(agents, envs) if not transfer_eval else product(agents, envs)

    for agent, generator in agent_env_work_pair:
        id = next(dispenser)
        lvl = str(generator) if args.generatorType == 'evolutionary' else None
        tasks[id]['env'].append(lvl)
        tasks[id]['nn'].append(agent.nn.state_dict())
        tasks[id]['nn_id'].append(agent.id)
        tasks[id]['env_id'].append(generator.id)
        tasks[id]['diff'].append(agent.generator.diff)

    return tasks


def updatePairs(pairs, answers, task_type, poet_loop_id, stats):
    """

    :param pairs: list of active NN-Env pairs
    :param answers: flattened by chromosome_id and env_id children_response dicts
    :param task_type: ADPTASK ID
    :return:
    """
    # print("updating")
    # do something with the answers.
    # for each dict from the children

    for (xsome_id, env_id) in answers:
        # print(xsome_id)
        for each_pair in pairs:
            if xsome_id == each_pair.id:
                # print("found matching nn")
                each_pair.score = answers[(xsome_id, env_id)]['score']
                try:
                    stats['lvls'][each_pair.id]['won'] = answers[(xsome_id, env_id)]['won'] if answers[(xsome_id, env_id)]['won'] else False
                except KeyError:
                    import pdb; pdb.set_trace()
                if task_type == ADPTASK.OPTIMIZE:
                    nn = answers[(xsome_id, env_id)]['nn']  # this is a state_dict
                    each_pair.nn.load_state_dict(nn)


def cycleWorkers(parent):
    path = os.path.join(parent.root,
                        parent.subfolders['alive_signals'])

    alive = os.listdir(path)

    for a in alive:
        parent.placeChildFlag(os.path.join(path, a) + '.cycle')

def dieAndKillChildren(parent, pairs, stats):

    from utils.utils import save_obj
    save_obj(stats,
             os.path.join(f"{args.result_prefix}", f"results_{unique_run_id}"),
             f"pinsky_stats")

    # [pair.env.close() for pair in pairs]
    path = os.path.join(parent.root,
                        parent.subfolders['alive_signals'])

    alive = os.listdir(path)

    for a in alive:
        os.remove(os.path.join(path, a))
        # create #.txt.done files. 
        parent.placeChildFlag(os.path.join(path, a) + '.done')

def perform_transfer(pairs, answers, poet_loop_counter, unique_run_id, stats):
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

    new_weights = {}
    stats['transfers'][poet_loop_counter] = []

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
                    # print(f"update network {fixed_env_pair.id} to {changing_agent_pair.id}")
                    current_score = j_score
                    current_net   = changing_agent_pair.nn.state_dict()
                    transferred_id = changing_agent_pair.id
                    new_weights[fixed_env_pair.id] = (current_net, transferred_id)

    for k, fixed_env_pair in enumerate(pairs):
        if fixed_env_pair.id in new_weights:
            state_dict, new_agent_id = new_weights[fixed_env_pair.id]
            fixed_env_pair.nn.load_state_dict(state_dict)
            stats['transfers'][poet_loop_counter].append((new_agent_id, fixed_env_pair.id))

            with open(os.path.join(f'{args.result_prefix}', f'results_{unique_run_id}', f'{fixed_env_pair.id}',
                                   f'poet{poet_loop_counter}_network_{new_agent_id}_transferred_here.txt'),
                      'w+') as fname:
                pass

def pass_mc(new_generator, unique_run_id, poet_loop_counter):

    # print("testing MC")
    
    path_to_game = os.path.join('levels', f'gvgai_{args.game}_def.txt')
    # print("running mcts agent")
    # if you LOSE with a tree-serach agent, it's too hard.
    # print(f"path: {gridGame.path_to_file}, {type(gridGame.path_to_file)}")
    wonGameMCTS = runJavaAgent('runGVGAI.jar',
                               path_to_game,
                               new_generator.path_to_file,
                               args.comp_agent,
                               str(args.game_len))

    # print("running random agent")
    # if you WIN playing randomly, the level is too easy.
    wonGameRandomly = runJavaAgent('runGVGAI.jar',
                                   path_to_game,
                                   new_generator.path_to_file,
                                   'random',
                                   str(args.game_len))

    # if not too easy and not too hard:
    if not wonGameRandomly and wonGameMCTS:
        return True

    difficulty = ''
    if wonGameRandomly:
        difficulty += '.easy'
    if not wonGameMCTS:
        difficulty += '.hard'

    level = os.path.join(args.result_prefix, f'results_{unique_run_id}', 'rejected',
                         f'poet{poet_loop_counter}_lvl{new_generator.id}{difficulty}.txt')

    with open(level, 'w+') as fname:
        fname.write(new_generator.string)

    return False

def get_child_list(parent_list, max_children, unique_run_id, stats, poet_loop_counter):
    child_list = []
    passed = 0
    mutation_trial = 0

    while mutation_trial < max_children:
        # print(f"mutation_trial {mutation_trial + 1}/{max_children}")
        parent = np.random.choice(parent_list)

        if args.generatorType == "evolutionary":
            mu = args.mutation_rate
            minimal = args.minimal_mutation
            r = args.mutation_radius
        else:
            mu = None
            minimal = True,
            r = parent.generator.diff

        new_gen = parent.mutate(mutationRate=mu,
                                minimal=minimal,
                                r=r)

        mutation_trial += 1

        if pass_mc(new_gen, unique_run_id, poet_loop_counter):
            passed += 1
            child_list.append(MinimalPair(unique_run_id=unique_run_id,
                                          generatorType=args.generatorType,
                                          generator=new_gen,
                                          prefix=args.result_prefix,
                                          parent=parent.nn,
                                          game=parent.game))

            stats['lvls'][child_list[-1].id] = {}
            stats['lvls'][child_list[-1].id]['lvl'] = str(new_gen)
            stats['lvls'][child_list[-1].id]['lvlChar'] = getdZeldaLvlCharacterization(child_list[-1].fileName,
                                                                                       args.lvl_dir,
                                                                                       _args.args_file)
            stats['lvls'][child_list[-1].id]['won'] = False
            stats['lineage'].append((parent.id, child_list[-1].id))

            tag = os.path.join(f'{args.result_prefix}',
                               f'results_{unique_run_id}',
                               f'{child_list[-1].id}', f'parent_is_{parent.id}.txt')
            with open(tag, 'w+') as fname:
                pass
    stats[poet_loop_counter]['viable'] = passed / max_children
    # speciation or novelty goes here
    #

    return child_list

def send_work(distributed_work, task, parent, unique_run_id, poet_loop_counter):
    
    for worker_id in distributed_work:

        parent.createChildTask(run_id=unique_run_id,
                               work_dict=distributed_work[worker_id],
                               worker_id=worker_id,
                               task_id=task,
                               poet_loop_counter=poet_loop_counter,
                               rl=args.rl,
                               algo=args.DE_algo,
                               ngames=args.n_games,
                               popsize=args.popsize)


def getChildren(parent):

    children = callOut(parent)
    # print(children)

    # get available children
    availableChildren = parent.isChildAvailable(children)

    # if list is empty, wait and check again
    while not bool(availableChildren):
        time.sleep(2)
        availableChildren = parent.isChildAvailable(children)

    return availableChildren


####################### HELPER FUNCTIONS ##########################

# ARGUMENTS TO THE SCRIPT

import argparse
from utils.loader import load_from_yaml
parser = argparse.ArgumentParser()
parser.add_argument("--exp_name", type=str, help='exp name')
parser.add_argument("--args_file", type=str, default='./args.yml', help='path to args file')
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

    unique_run_id = _args.exp_name
    parent = ADPParent(prefix=os.path.join(f"{args.result_prefix}", f"results_{unique_run_id}"))
    net = Net(args.action, args.depth)
    # if args.game == 'dzelda':
    #     net.load_state_dict(torch_load(f'./start.pt'))

    lvl = _initialize(os.path.join(args.lvl_dir, f"{args.game}_{args.init_lvl}"))
    lvl_shape = lvl.shape

    genArgs = {'game':args.game,
               'args_file':_args.args_file,
               'tile_world':lvl,
               'run_folder': os.path.join(f"{args.result_prefix}", f"results_{unique_run_id}"),
               'shape':lvl.shape,
               'path':args.lvl_dir,
               'diff':0.05,
               'mechanics':args.mechanics,
               'env_id':0,
               'generation':0}

    print("Create generator")
    Generator = EvolutionaryGenerator if args.generatorType == "evolutionary" else IlluminatingGenerator
    generator = Generator(**genArgs)

    archive = []

    print("Create minPair")
    pairs = [MinimalPair(unique_run_id=unique_run_id,
                         game=args.game,
                         generatorType=args.generatorType,
                         generator=generator,
                         parent=net,
                         prefix=args.result_prefix)
            ]

    generator.to_file(0, args.game)

    done = False
    i = 0
    chkpt = os.path.join(f"{args.result_prefix}", f"results_{unique_run_id}", f"POET_CHKPT")
    reject = os.path.join(f"{args.result_prefix}", f"results_{unique_run_id}", f"rejected")
    if not os.path.exists(chkpt):
        os.mkdir(chkpt)
    if not os.path.exists(reject):
        os.mkdir(reject)
    
    stats = {}
    stats['lineage'] = []
    stats['transfers'] = {}
    stats['lvls'] = {}

    stats['lvls'][pairs[0].id] = {}
    stats['lvls'][pairs[0].id]['lvl'] = str(pairs[0].generator)
    stats['lvls'][pairs[0].id]['lvlChar'] = getdZeldaLvlCharacterization(pairs[0].fileName,
                                                                            args.lvl_dir,
                                                                            _args.args_file)

    time.sleep(20)

    pbar = tqdm(total=args.num_poet_loops)
    

    while not done:
        try:
            stats[i] = {}
            # print(f"poet loop {i}")

            if (i + 1) % args.refresh == 0:
                # print("refreshing workers")
                cycleWorkers(parent)
                time.sleep(20)

            tdir = os.path.join(chkpt, str(i))
            if not os.path.exists(tdir):
                os.mkdir(tdir)
                # check if children are alive
            availableChildren = getChildren(parent)

            distributed_work = divideWorkBetweenChildren(pairs,  #  agents. We're not going to use the paired envs
                                                         [pairs[j].generator for j in range(len(pairs))],
                                                         availableChildren)

            # print("evaluating")
            send_work(distributed_work, ADPTASK.EVALUATE, parent, unique_run_id, i)
            
            # get answers from children
            eval_answers = waitForAndCollectAnswers(parent, availableChildren, distributed_work, unique_run_id, i,  ADPTASK.EVALUATE)

            updatePairs(pairs, eval_answers, ADPTASK.EVALUATE, i, stats)

            # Add in new children
            #
            new_envs = []
            # print("mutation?")
            if i % args.mutation_timer == 0:
                # print("yes")
                new_envs = get_child_list(pairs, args.max_children, unique_run_id, stats, i)

            pairs.extend(new_envs)
            #archive.extend(new_envs)
            del new_envs # this does not delete the children that have now been placed in pairs.
            # print(len(pairs))

            # kill extra population.
            #
            if len(pairs) > args.max_envs:
                aged_pairs = sorted(pairs, key=lambda x: x.id, reverse=True)
                pairs = aged_pairs[:args.max_envs]
                archive.extend(aged_pairs[args.max_envs:])
                del aged_pairs
            
            # Optimizations step
            #
            availableChildren = getChildren(parent)
            # print("optimizing")
            distributed_work = divideWorkBetweenChildren(pairs,
                                                         [pairs[j].generator for j in range(len(pairs))],
                                                         availableChildren)

            send_work(distributed_work, ADPTASK.OPTIMIZE, parent, unique_run_id, i)

            # get answers from children
            opt_answers = waitForAndCollectAnswers(parent, availableChildren, distributed_work, unique_run_id, i, ADPTASK.OPTIMIZE)

            updatePairs(pairs, opt_answers, ADPTASK.OPTIMIZE, i, stats)

            # TRANSFER NNs between ENVS,
            # EVALUATE each NN with each ENV.
            #
            if (i + 1) % args.transfer_timer == 0:
                # print("transferring")
                availableChildren = getChildren(parent)
                distributed_work = divideWorkBetweenChildren(pairs,
                                                             [pairs[j].generator for j in range(len(pairs))],
                                                             availableChildren,
                                                             transfer_eval=True)

                send_work(distributed_work, ADPTASK.EVALUATE, parent, unique_run_id, i)

                # get answers from children
                transfer_eval_answers = waitForAndCollectAnswers(parent, availableChildren, distributed_work, unique_run_id, i, ADPTASK.EVALUATE)

                # use information to determine if NN i should migrate to env j.
                perform_transfer(pairs, transfer_eval_answers, i, unique_run_id, stats)

            # save checkpoints of networks into POET folder
            #
            for pair in pairs:
                torch_save(pair.nn.state_dict(), os.path.join(tdir,
                                                              f'network{pair.id}.pt'))
                with open(os.path.join(tdir,
                                       f'lvl{pair.id}.txt'), 'w+') as fname:
                    fname.write(str(pair.generator))

            i += 1
            pbar.update(1)
            if i >= args.num_poet_loops:
                done = True

        except KeyboardInterrupt as e:
            print(e)
            pbar.close()
            dieAndKillChildren(parent, pairs)
            import sys
            sys.exit(0)

    print("dying")
    pbar.close()
    dieAndKillChildren(parent, pairs, stats)

