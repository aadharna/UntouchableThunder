import os
import time
from utils.ADPChild import ADPChild


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--id", type=int, help='set child id')
parser.add_argument("--game", type=str, default='dzelda', help='what game will POET use?')
parser.add_argument("--game_len", type=int, default=250, help='how many actions can the agent take?')
parser.add_argument("--lvl_dir", type=str, default='./levels', help='path to lvl dir')
args = parser.parse_args()

print(args)

if __name__ == "__main__":

    child = ADPChild(args.id, game=args.game, length=args.game_len, lvl_dir=args.lvl_dir)
    done = False
    while not done:
        try:
            while not child.hasTask():
                # if you're waiting for a task and 
                # your alive flag is removed by the parent
                # kill yourself. 
                if not os.path.exists(child.alive):
                    done = True
                    break
                    
                time.sleep(5)
            
            if not done:
                print("found task")
                child.recieveTaskAndReturnAnswer()
            
            

        except KeyboardInterrupt as e:
            print(e)
            child.pair.env.close()
            import sys
            sys.exit(0)

        # done = True

    child.pair.env.close()
