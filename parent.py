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
    
    print(tasks)
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


def updatePairs(pairs, answers, task_type):
    print("updating")
    # do something with the answers.
    # for each dict from the children
    for each_answer, flag in answers:
        # for each answer in the top dict from each child
        if flag:
            for chromosome_ids in each_answer:
                score = each_answer[chromosome_ids]['score']
                _id = each_answer[chromosome_ids]['chromosome_id']

                for each_pair in pairs:
                    if _id == each_pair.id:
                        each_pair.score = score
                        if task_type == ADPTASK.OPTIMIZE:
                            nn = each_answer[_id]['nn'] # this is a state_dict
                            each_pair.nn.load_state_dict(nn)

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
            
            print("evaluating")
            for worker_id in distributed_work:
                (nns, envs, task_types,
                 chromosome_ids, rl, algo, ngames) = prepareWorkForChild(distributed_work[worker_id],
                                                                         ADPTASK.EVALUATE)

                parent.createChildTask(nns            = nns,
                                       envs           = envs,
                                       task_types     = task_types,
                                       chromosome_ids = chromosome_ids,
                                       child_id       = int(worker_id),
                                       rl             = rl,
                                       algo           = algo,
                                       ngames         = ngames)

            # get answers from children
            eval_answers = waitForAndCollectAnswers(parent, availableChildren)

            updatePairs(pairs, eval_answers, ADPTASK.EVALUATE)

            # mutate environment
            new_envs = []
            if (i+1) % args.mutation_timer == 0:
                for pair in pairs:
                    print(f"mutating {pair.id}")
                    new_envs.append(pair.mutate(args.mutation_rate))
            
            pairs.extend(new_envs)
            del new_envs # this does not delete the pairs that have now been placed in pairs.
            print(len(pairs))
            
            # kill extra population.
            #
            # CODE GOES HERE ?
            #
            
            print("optimizing")
            distributed_work = divideWorkBetweenChildren(pairs, availableChildren)

            for worker_id in distributed_work:
                # create work for everyone, even if there is no work to do
                
                (nns, envs, task_types,
                 chromosome_ids, rl, algo, ngames) = prepareWorkForChild(distributed_work[worker_id],
                                                                         ADPTASK.OPTIMIZE)

                parent.createChildTask(nns=nns,
                                       envs=envs,
                                       task_types=task_types,
                                       chromosome_ids=chromosome_ids,
                                       child_id=int(worker_id),
                                       rl=rl,
                                       algo=algo,
                                       ngames=ngames)


            # get answers from children
            opt_answers = waitForAndCollectAnswers(parent, availableChildren)

            updatePairs(pairs, opt_answers, ADPTASK.OPTIMIZE)

            # TRANSFER NNs between ENVS,
            # EVALUATE each NN with each ENV.
            #
            # CODE GOES HERE
            #

            # print(optAnswers)

            i += 1
            if i >= 4:
                done = True

        except KeyboardInterrupt as e:
            print(e)
            [pair.env.close() for pair in pairs]
            import sys
            sys.exit(0)

    [pair.env.close() for pair in pairs]
    
    path = os.path.join(parent.root,
                        parent.subfolders['alive_signals'])

    alive = os.listdir(path)
    
    for a in alive:
        os.remove(os.path.join(path, a))


