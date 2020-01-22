import os
import time
import shutil
import numpy as np
from utils.utils import save_obj, load_obj


class ADPParent:
    """
    ADAPTIVE DYNAMIC PROCESS PARENT CLASS
    This will launch out commands into the void
    for APDChild processes to pick up and execute.
    """
    def __init__(self):
        self.root = os.path.join('.', 'communication')
        if not os.path.exists(self.root):
            os.mkdir(self.root)
        
        self.subfolders = {
            'send_to_child': 'outgoing',
            'sent_by_child': 'incoming',
            'alive_signals': 'alive',
            'available_signals': 'available'
        }
        self.createFolders()
        self.resetFolders()

    def createFolders(self):
        for f in self.subfolders.keys():
            path = os.path.join(self.root, self.subfolders[f])
            if not os.path.exists(path):
                os.mkdir(path)

    def resetFolders(self):
        for (_, dirs, _) in os.walk(self.root):
            for d in dirs:
                path  = os.path.join(self.root, d)
                files = os.listdir(path)
                for f in files:
                    os.remove(os.path.join(path, f))

    def checkChildResponseStatus(self, allChildren):        
        response_folder = os.path.join(self.root,
                                       self.subfolders['sent_by_child'])
        
        available_signals = os.listdir(os.path.join(self.root,
                                   self.subfolders['available_signals']))
        
        dones = [False]*len(allChildren)

        for i, c in enumerate(allChildren):
            if os.path.exists(os.path.join(response_folder, f'answer{c}') + '.pkl') and \
               f'{c}.txt' in available_signals:
                dones[i] = True
        
        return np.all(dones)

    def selectAvailableChild(self, availableChildren):
        return np.random.choice(availableChildren)

    def isChildAvailable(self, children):
        availableChildren = []
        alive_signals = [c.split('.')[0] for c in os.listdir(
            os.path.join(self.root, self.subfolders['alive_signals'])
        )]

        available_signals = [c.split('.')[0] for c in os.listdir(
            os.path.join(self.root, self.subfolders['available_signals'])
        )]

        for child in children:
            # if child is alive and available
            if child in alive_signals and child in available_signals:
                availableChildren.append(child)

        return availableChildren

    def readChildAnswer(self, response_file):
        folder = os.path.join(self.root,
                              self.subfolders['sent_by_child'])
        answer = load_obj(folder, response_file)
        # remove answer from folder
        os.remove(os.path.join(folder, response_file))
        return answer, bool(answer)

    def pickupChildren(self):
        """
        Picks up all of the alive children.
        :return:
        """
        alive_signals = os.listdir(
            os.path.join(self.root, self.subfolders['alive_signals'])
        )
        children = []
        for child in alive_signals:
            id = child.split('.')[0] #child names are 1.txt, etc
            children.append(id)

        return sorted(children)

    def createChildTask(self, nns, envs, task_types, chromosome_ids, child_id, poet_loop_counter, **kwargs):
        """

        :param nns:        list of PyTorch NN
        :param envs:       list of GridGame env
        :param task_types: list of ADPTASK ID
        :param chromosome_ids: chromosome_id (list of ints)
        :param child_id:  child id (int)
        :param poet_loop_counter: poet number loop
        :return:
        """

        work = {
            'nns': [nn.state_dict() for nn in nns],
            'lvls': [str(env.generator) for env in envs],
            'task_ids': [task_type for task_type in task_types],
            'chromosome_ids': [chromosome_id for chromosome_id in chromosome_ids],
            'poet': poet_loop_counter,
            'kwargs': kwargs
        }

        save_obj(work,
                 os.path.join(self.root, self.subfolders['send_to_child']),
                 f'child{child_id}')

        available = os.path.join(self.root,
                                 self.subfolders['available_signals'],
                                 f'{child_id}.txt')
        os.remove(available)
        
        time.sleep(3)
