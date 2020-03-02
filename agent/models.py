import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions.categorical import Categorical

class Value(nn.Module):
    def __init__(self, depth):
        super(Value, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=depth, out_channels=8, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=32, kernel_size=3)
        self.fc = nn.Linear((32 * 3) + 4, 1)

    def forward(self, x, compass):
        x = self.pool(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = x.view(len(x), 32 * 3)
        x = torch.cat([x, compass], dim=1)
        x = self.fc(x)
        return x

# THIS CLASS SHOULD BE CALLED ACTOR
# but I originally had it as Net. 
class Net(nn.Module): 
    def __init__(self, n_actions, depth):
        super(Net, self).__init__()
        self.resizecoeff = 3 if depth == 13 else 4
        self.conv1 = nn.Conv2d(in_channels=depth, out_channels=8, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=32, kernel_size=3)
        self.fc1 = nn.Linear((32 * self.resizecoeff) + 4, 48)
                        # neurons from conv layer + neurons to enter compass info
        self.fc2 = nn.Linear(48, 24)
        self.fc3 = nn.Linear(24, n_actions)

    def forward(self, x, compass):
        x = self.pool(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = x.view(len(x), 32 * self.resizecoeff)
        x = torch.cat([x, compass], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1) #dim = 0 since array was flattened
        return x
    
class ActorCritic(nn.Module):
    def __init__(self, n_actions, depth):
        super(ActorCritic, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=depth, out_channels=8, kernel_size=3)
        self.pool  = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=32, kernel_size=3)
        
        # actor
        self.fc1   = nn.Linear((32 * 3) + 4, 48)
                        # neurons from conv layer + neurons to enter compass info
        self.fc2   = nn.Linear(48, 24)
        self.fc3   = nn.Linear(24, n_actions)
        
        #critic
        self.fc4   = nn.Linear((32 * 3) + 4, 1)

    def forward(self, x, compass):
        # feature_extraction
        x = self.pool(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = x.view(len(x), 32 * 3)
        feat = torch.cat([x, compass], dim=1)
        
        # actor
        a = F.relu(self.fc1(feat))
        a = F.relu(self.fc2(a))
        # policy output
        a = F.softmax(self.fc3(a), dim=1) #dim = 0 since array was flattened
        
        # critic
        # critic regression output
        c = self.fc4(feat)
        return a, c
    
    