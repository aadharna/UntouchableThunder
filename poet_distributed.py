import os
import time
from itertools import product

from utils.ADPParent import ADPParent
from utils.ADPTASK_ENUM import ADPTASK

from agent.NNagent import NNagent
from generator.env_gen_wrapper import GridGame

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

    flat_answers = flatten(answers, children)

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

    [pair.env.close() for pair in pairs]

    path = os.path.join(parent.root,
                        parent.subfolders['alive_signals'])

    alive = os.listdir(path)

    for a in alive:
        os.remove(os.path.join(path, a))

####################### HELPER FUNCTIONS ##########################

# ARGUMENTS TO THE SCRIPT

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--game", type=str, default='dzelda', help='set gvgai game')
parser.add_argument("--init_lvl", type=str, default='start.txt', help='level from ./levels folder')
parser.add_argument("--game_len", type=int, default=250, help='game length')
parser.add_argument("--n_games", type=int, default=1000, help='opt length in num games')
parser.add_argument("--rl", type=bool, default=False, help='use RL?')
parser.add_argument("--DE_algo", type=str, default='jDE', help='which DE algo to use if rl is False?')
parser.add_argument("--mutation_timer", type=int, default=5, help='steps until mutation')
parser.add_argument("--mutation_rate", type=float, default=0.5, help='change of mutation')
parser.add_argument("--transfer_timer", type=int, default=15, help='steps until transfer')
parser.add_argument("--max_children", type=int, default=8, help='number of children to add each transfer step')
parser.add_argument("--max_envs", type=int, default=50, help='max number of GVGAI-gym envs allowed at any one time')


args = parser.parse_args()

print(args)

############### POET ###############

if __name__ == "__main__":

    parent = ADPParent()

    pairs = [NNagent(GridGame(game=args.game,
                            play_length=args.game_len,
                            path='./levels',
                            lvl_name=args.init_lvl,
                            mechanics=['+', 'g'],
                            # monsters, key, door, wall
                            )
                   )
             ]

    done = False
    i = 0
    while not done:
        try:
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

            print("evaluating")
            for worker_id in distributed_work:

                parent.createChildTask(work_dict=distributed_work[worker_id],
                                       worker_id=worker_id,
                                       task_id=ADPTASK.EVALUATE,
                                       poet_loop_counter=i,
                                       rl=args.rl,
                                       algo=args.DE_algo,
                                       ngames=args.n_games)

            # get answers from children
            eval_answers = waitForAndCollectAnswers(parent, availableChildren)

            updatePairs(pairs, eval_answers, ADPTASK.EVALUATE)

            # mutate environment
            # POET VERSION
            # def get_child_list(self, parent_list, max_children):
            #     child_list = []
            #
            #     mutation_trial = 0
            #     while mutation_trial < max_children:
            #         new_env_config, seed, parent_optim_id = self.get_new_env(parent_list)
            #         mutation_trial += 1
            #         if self.pass_dedup(new_env_config):
            #             o = self.create_optimizer(new_env_config, seed, is_candidate=True)
            #             score = o.evaluate_theta(self.optimizers[parent_optim_id].theta)
            #             del o
            #             if self.pass_mc(score):
            #                 novelty_score = compute_novelty_vs_archive(self.env_archive, new_env_config, k=5)
            #                 logger.debug("{} passed mc, novelty score {}".format(score, novelty_score))
            #                 child_list.append((new_env_config, seed, parent_optim_id, novelty_score))
            #
            #     #sort child list according to novelty for high to low
            #     child_list = sorted(child_list,key=lambda x: x[3], reverse=True)
            #     return child_list

            new_envs = []
            if (i+1) % args.mutation_timer == 0:
                for pair in pairs:
                    print(f"mutating {pair.id}")
                    new_envs.append(pair.mutate(args.mutation_rate))

                    # CODE TO TEST NEW ENV GOES HERE.

                    with open(f'./results/{new_envs[-1].id}/parent_is_{pair.id}.txt', 'w+') as fname:
                        pass

            pairs.extend(new_envs)
            del new_envs # this does not delete the pairs that have now been placed in pairs.
            print(len(pairs))

            # kill extra population.
            #
            if len(pairs) > args.max_envs:
                aged_pairs = sorted(pairs, key=lambda x: x.id, reverse=True)
                for extra_env_ids in range(args.max_envs, len(aged_pairs)):
                    aged_pairs[extra_env_ids].env.close()  # close the java envs. delete them from memory.
                                                           # zombie processes will be cleaned up upon exit of main.
                pairs = aged_pairs[:args.max_envs]
                del aged_pairs
            
            # Optimizations step
            #
            print("optimizing")
            distributed_work = divideWorkBetweenChildren(pairs,
                                                         [pairs[i].env for i in range(len(pairs))],
                                                         availableChildren)

            for worker_id in distributed_work:

                parent.createChildTask(work_dict=distributed_work[worker_id],
                                       worker_id=worker_id,
                                       task_id=ADPTASK.OPTIMIZE,
                                       poet_loop_counter=i,
                                       rl=args.rl,
                                       algo=args.DE_algo,
                                       ngames=args.n_games)


            # get answers from children
            opt_answers = waitForAndCollectAnswers(parent, availableChildren)

            updatePairs(pairs, opt_answers, ADPTASK.OPTIMIZE)

            # TRANSFER NNs between ENVS,
            # EVALUATE each NN with each ENV.
            #
            if (i + 1) % args.transfer_timer == 0:
                print("transferring")
                distributed_work = divideWorkBetweenChildren(pairs,
                                                             [pairs[i].env for i in range(len(pairs))],
                                                             availableChildren,
                                                             transfer_eval=True)

                for worker_id in distributed_work:

                    parent.createChildTask(work_dict=distributed_work[worker_id],
                                           worker_id=worker_id,
                                           task_id=ADPTASK.EVALUATE,
                                           poet_loop_counter=i,
                                           rl=args.rl,
                                           algo=args.DE_algo,
                                           ngames=args.n_games)


                # get answers from children
                transfer_eval_answers = waitForAndCollectAnswers(parent, availableChildren)

                # use information to determine if NN i should migrate to env j.




            # save checkpoints of networks into POET folder
            #
            # CODE GOES HERE.

            i += 1
            if i >= 4:
                done = True

        except KeyboardInterrupt as e:
            print(e)
            dieAndKillChildren(parent, pairs)
            import sys
            sys.exit(0)

    dieAndKillChildren(parent, pairs)
