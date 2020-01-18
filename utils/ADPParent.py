import os
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
        self.subfolders = {
            'send_to_child': 'outgoing',
            'sent_by_child': 'incoming',
            'alive_signals': 'alive',
            'busy_signals': 'busy'
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

    def checkChildResponseStatus(self, children):
        folder = os.path.join(self.root,
                              self.subfolders['sent_by_child'])

        for c in children:
            if not os.path.exists(os.path.join(folder, f'answer{c}') + '.pkl'):
                return False
        return True

    def selectAvailableChild(self, availableChildren):
        return np.random.choice(availableChildren)

    def isChildAvailable(self, children):
        availableChildren = []
        alive_signals = [c.split('.')[0] for c in os.listdir(
            os.path.join(self.root, self.subfolders['alive_signals'])
        )]

        busy_signals = [c.split('.')[0] for c in os.listdir(
            os.path.join(self.root, self.subfolders['busy_signals'])
        )]

        for child in children:
            if child in alive_signals and child not in busy_signals:
                availableChildren.append(child)
        return np.all(availableChildren), availableChildren

    def readChildAnswer(self, response_file):
        folder = os.path.join(self.root,
                              self.subfolders['sent_by_child'])
        answer = load_obj(folder, response_file)
        # remove answer from folder
        os.remove(os.path.join(folder, response_file))
        return answer

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

    def createChildTask(self, nn, env, task_type, chromosome_id, child_id, **kwargs):
        """

        :param nn:        PyTorch NN
        :param env:       GridGame env
        :param task_type: ADPTASK ID
        :param chromosome_id: chromosome_id (int)
        :param child_id:  child id (int)
        :return:
        """

        sample = {
            'nn': nn.state_dict(),
            'lvl': str(env.generator),
            'task_id': task_type,
            'chromosome_id': chromosome_id,
            'kwargs': kwargs
        }

        save_obj(sample,
                 os.path.join(self.root, self.subfolders['send_to_child']),
                 f'child{child_id}')
