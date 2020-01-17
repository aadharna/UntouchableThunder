import os
import shutil
import numpy as np
from utils.utils import save_obj


class ADPParent:
    """
    ADAPTIVE DYNAMIC PROCESS PARENT CLASS
    This will launch out commands into the void
    for APDChild processes to pick up and execute.
    """
    def __init__(self):
        self.root = os.path.join('.', 'communication')
        self.subfolders = {
            'send_to_child': 'outgoing',
            'sent_by_child': 'incoming',
            'alive_signals': 'alive',
            'busy_signals': 'busy'
        }
        self.createFolders()

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

    def checkChildResponseStatus(self, children):
        folder = os.path.join(self.root,
                              self.subfolders['sent_by_child'])

        for c in children:
            if not os.path.exists(os.path.join(folder, c) + '.pkl'):
                return False
        return True

    def readChildAnswer(self, response_file):
        pass

    def pickupChildren(self):
        alive_signals = os.listdir(
            os.path.join(self.root, self.subfolders['alive_signals'])
        )
        children = []
        for child in alive_signals:
            id = child.split('.')[0] #child names are 1.txt, etc
            children.append(id)

        return sorted(children)


    def pick_available_child(self):
        alive = os.listdir(os.path.join(self.root,
                                        self.subfolders['alive_signals']))
        busy = os.listdir(os.path.join(self.root,
                                       self.subfolders['busy_signals']))

        for c in self.child_ids:
            path = f'{c}.txt'
            if path in alive and path not in busy:
                return c

    def createChildTask(self, nn, env, task_type, child_id, **kwargs):
        """

        :param nn:        PyTorch NN
        :param env:       GridGame env
        :param task_type: ADPTASK ID
        :param id:        child id
        :return:
        """

        sample = {
            'nn': nn.state_dict(),
            'lvl': str(env.generator),
            'task_id': task_type,
            'chromosome_id': id,
            'kwargs': kwargs
        }

        #child_id = self.pick_available_child()

        save_obj(sample,
                 os.path.join(self.root, self.subfolders['send_to_child']),
                 f'child{child_id}')
