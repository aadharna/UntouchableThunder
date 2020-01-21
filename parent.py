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
        print("calling out again")
        try:
            time.sleep(5)
        except KeyboardInterrupt as e:
            print('dying')
            print(e)
            import sys
            sys.exit(0)
        children = parent.pickupChildren()
    print(f'children alive: {len(children)}')
    return children


def waitForAndCollectAnswers(parent, children):

    while not parent.checkChildResponseStatus(children):
        time.sleep(5)

    answer_pointers = os.listdir(os.path.join(
        parent.root,
        parent.subfolders['sent_by_child']
    ))

    answers = [parent.readChildAnswer(answer) for answer in answer_pointers]

    return answers


if __name__ == "__main__":

    parent = ADPParent()

    pairs = [NNagent(GridGame(game='zelda',
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
            if i == 2:
                pairs.append(pairs[0].mutate(1))

            # check if children are alive
            children = callOut(parent)
            print(children)

            # get available children
            availableChildren = parent.isChildAvailable(children)

            # if list is empty, wait and check again
            while not bool(availableChildren):
                time.sleep(5)
                availableChildren = parent.isChildAvailable(children)

            # pick from available chidren
            child = parent.selectAvailableChild(availableChildren)

            # parent.createChildTask(nns           = [pairs[0].nn],
            #                        envs          = [pairs[0].env],
            #                        task_types    = [ADPTASK.EVALUATE],
            #                        chromosome_ids= [pairs[0].id],
            #                        child_id     = int(child),  # str 4 --> int 4
            #                        rl           = [True])
            #
            # evalAnswers = waitForAndCollectAnswers(parent, children)
            #
            # print(evalAnswers)

            parent.createChildTask(nns           = [pairs[0].nn, pairs[0].nn],
                                   envs          = [pairs[0].env, pairs[0].env],
                                   task_types    = [ADPTASK.OPTIMIZE, ADPTASK.EVALUATE],
                                   chromosome_ids= [pairs[0].id, 2],
                                   child_id      = int(child),  # str 4 --> int 4
                                   rl            = [True, False],
                                   algo          = [None, 'jDE'],
                                   ngames        = [1000, 400])

            optAnswers = waitForAndCollectAnswers(parent, children)

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




