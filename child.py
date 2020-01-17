import time
from utils.ADPChild import ADPChild


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--id", type=int, help='set child id')
args = parser.parse_args()

if __name__ == "__main__":

    child = ADPChild(args.id)
    done = False
    while not done:
        try:
            while not child.hasTask():
                time.sleep(5)
            print("found task")
            child.recieveTaskAndReturnAnswer()

        except KeyboardInterrupt as e:
            print(e)
            child.pair.env.close()
            import sys
            sys.exit(0)

        done = True

    child.pair.env.close()
