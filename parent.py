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






if __name__ == "__main__":

    parent = ADPParent()


    pair = NNagent(GridGame(game='zelda',
                            play_length=250,
                            path='./levels',
                            lvl_name='start.txt',
                            mechanics=['+', 'g'],
                            # monsters, key, door, wall
                            )
                   )

    done = False
    i = 0
    while not done:
        try:
            children = callOut(parent)
            print(children)
            parent.createChildTask(pair.nn, pair.env, ADPTASK.EVALUATE, pair.id, children[0], rl=True)

            while not parent.checkChildResponseStatus(children):
                time.sleep(5)

            answer_pointers = os.listdir(os.path.join(
                parent.root,
                parent.subfolders['sent_by_child']
            ))
            answers = [parent.readChildAnswer(answer) for answer in answer_pointers]

            print(answers)

            i += 1
            if i >= 3:
                done = True

        except KeyboardInterrupt as e:
            print(e)
            pair.env.close()
            import sys
            sys.exit(0)

    pair.env.close()




