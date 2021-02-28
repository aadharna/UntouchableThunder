import numpy as np
import time

class BaseGenerator:
    def __init__(self):
        self.locations = {}
        pass

    # for compatability reasons
    def update_from_lvl_string(self, new_lvl):
        raise NotImplementedError

    def generate(self):
        '''
        :return: the id and path of the next level to be evaluated.
        '''
        raise NotImplementedError

    def mutate(self, **kwargs):
        """
        Return information to make a new Generator object
        """
        raise NotImplementedError


def _initialize(path, d=9):
    """build numpy array of level from txt file

    :param path: path to txt file representation of level
    :return:
    """
    failing = True
    while failing:

        fpointer = open(path, 'r')
        try:
            f = fpointer.read()
            t = f[-1]
            failing = False
        except IndexError:
            time.sleep(0.1)
        finally:
            fpointer.close()
    # time.sleep(0.05)
    # with open(path, 'r') as file:
    #     # print(path)
    #     f = file.read()

    if f[-1] != "\n":
        f += "\n"
    f = f.split('\n')[:-1] #remove blank line.
    d = len(f)
    tile = [list(row) for row in f]
    
    npa = np.array(tile).reshape((d, -1))  # make into numpy array 9x13
    return npa
