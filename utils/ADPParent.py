import os
import shutil
from utils.utils import save_obj


class ADPParent:
    """
    ADAPTIVE DYNAMIC PROCESS PARENT CLASS
    This will launch out commands into the void
    for APDChildRunner processes to pick up and execute.
    """
    def __init__(self, num_children):
        self.root = os.path.join('.', 'communication')
        self.subfolders = {
            'send_to_child': 'outgoing',
            'sent_by_child': 'incoming',
            'alive_signals': 'alive',
            'busy_signals': 'busy'
        }
        self.createFolders()

        self.child_ids = range(num_children)

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

    def readChildAnswer(self, response_file):
        pass

    def registerChildren(self):
        pass

    def pick_available_child(self):
        alive = os.listdir(os.path.join(self.root,
                                        self.subfolders['alive_signals']))
        busy = os.listdir(os.path.join(self.root,
                                       self.subfolders['busy_signals']))

        for c in self.child_ids:
            path = f'{c}.txt'
            if path in alive and path not in busy:
                return c

    def createChildTask(self, nn, env, task_type, id, **kwargs):
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

        child_id = self.pick_available_child()

        save_obj(sample,
                 os.path.join(self.root, self.subfolders['send_to_child']),
                 f'child{child_id}')
