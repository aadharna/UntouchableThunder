import numpy as np

class BaseGenerator:
    def __init__(self):
        super().__init__()


        pass

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
    with open(path, 'r') as file:
        f = file.read()
        if f[-1] != "\n":
            f += "\n"
        f = f.split('\n')[:-1] #remove blank line.
        d = len(f)
        tile = [list(row) for row in f]
    
    npa = np.array(tile).reshape((d, -1))  # make into numpy array 9x13
    return npa
