import os
import time

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


def waitForAndCollectAnswers(parent, children):
    print('waiting for answers')

    while not parent.checkChildResponseStatus(children):
        time.sleep(5)

    answer_pointers = os.listdir(os.path.join(
        parent.root,
        parent.subfolders['sent_by_child']
    ))

    answers = [parent.readChildAnswer(answer) for answer in answer_pointers]
    
    print('collected answers')
    return answers

def divideWorkBetweenChildren(pairs, children):
    
    # private function to implement circular queue for assigning tasks
    def dispenseChild(children):
        num_children = len(children)
        for i in range(1000000):
            yield children[i % num_children]
    
    dispensor = dispenseChild(children)
    tasks = {next(dispensor): [] for _ in range(len(children))} 
    for job in pairs:
        tasks[next(dispensor)].append(job)
    
    return tasks

def prepareWorkForChild(pairs, ADPTASK_TYPE):
    
    nns = []
    envs = []
    task_types = []
    chromosome_ids = []
    rl = []
    algo = []
    ngames = []

    for task in pairs:
        nns.append(task.nn)
        envs.append(task.env)
        task_types.append(ADPTASK_TYPE)
        chromosome_ids.append(task.id)
        rl.append(args.rl)
        algo.append(args.DE_algo)
        ngames.append(args.n_games)
    
    return nns, envs, task_types, chromosome_ids, rl, algo, ngames

####################### HELPER FUNCTIONS ##########################

# ARGUMENTS TO THE SCRIPT

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--game", type=str, default='dzelda', help='set gvgai game')
parser.add_argument("--init_lvl", type=str, default='start.txt', help='level from ./levels folder')
parser.add_argument("--game_len", type=int, default=250, help='game length')
parser.add_argument("--n_games", type=int, default=1000, help='num games')
parser.add_argument("--rl", type=bool, default=False, help='use RL?')
parser.add_argument("--DE_algo", type=str, default='jDE', help='which DE algo to use?')
parser.add_argument("--mutation_timer", type=int, default=5, help='steps until mutation')
parser.add_argument("--mutation_rate", type=float, default=0.5, help='change of mutation')
parser.add_argument("--transfer_timer", type=int, default=15, help='steps until transfer')


args = parser.parse_args()

print(args)

if __name__ == "__main__":

    parent = ADPParent()

    pairs = [NNagent(GridGame(game='dzelda',
                            play_length=250,
                            path='./levels',
                            lvl_name='start.txt',
                            mechanics=['+', 'g'],
                            # monsters, key, door, wall
                            )
                   )
             ]

    done = False
    i = 0
    while not done:
        try:
            # mutate environment
            if (i+1) % args.mutation_timer == 0:
                for pair in pairs:
                    pairs.append(pair.mutate(args.mutation_rate))


            # check if children are alive
            children = callOut(parent)
            print(children)

            # get available children
            availableChildren = parent.isChildAvailable(children)

            # if list is empty, wait and check again
            while not bool(availableChildren):
                time.sleep(5)
                availableChildren = parent.isChildAvailable(children)
            
            distributed_work = divideWorkBetweenChildren(pairs, availableChildren)
            
            for worker_id in distributed_work:
                (nns, envs, task_types, chromosome_ids, rl, algo, ngames) = prepareWorkForChild(distributed_work[worker_id], ADPTASK.EVALUATE)
                    
                parent.createChildTask(nns            = nns,
                                       envs           = envs,
                                       task_types     = task_types,
                                       chromosome_ids = chromosome_ids,
                                       child_id       = int(worker_id),
                                       rl             = rl,
                                       algo           = algo,
                                       ngames         = ngames)
                                       
            
            eval_answers = waitForAndCollectAnswers(parent, children)
            

#             parent.createChildTask(nns           = [pairs[0].nn, pairs[0].nn],
#                                    envs          = [pairs[0].env, pairs[0].env],
#                                    task_types    = [ADPTASK.OPTIMIZE, ADPTASK.EVALUATE],
#                                    chromosome_ids= [pairs[0].id, 2],
#                                    child_id      = int(child),  # str 4 --> int 4
#                                    rl            = [True, False],
#                                    algo          = [None, 'jDE'],
#                                    ngames        = [1000, 400])

#             optAnswers = waitForAndCollectAnswers(parent, children)

            # print(optAnswers)

            i += 1
            if i >= 3:
                done = True

        except KeyboardInterrupt as e:
            print(e)
            [pair.env.close() for pair in pairs]
            import sys
            sys.exit(0)

    [pair.env.close() for pair in pairs]


    
