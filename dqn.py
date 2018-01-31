import torch.nn as nn

from phi import Phi

class DQN(nn.Module):
    """
    DQN baseline model used in the paper being reproduced
    """
    def __init__(self, n_actions):
        super(DQN, self).__init__()
        self.phi = Phi()
        self.w_dqn = nn.Linear(64, n_actions)

    def forward(self, input):
        h = self.phi(input)
        q = self.w_dqn(h)
        return q
