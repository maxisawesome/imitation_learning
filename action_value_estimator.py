import torch.nn as nn

from phi import Phi

class ActionValueEstimatorNet(nn.Module):
    """
    Action value estimator component of the imitation learning network proposed
    in the paper being reproduced.
    """
    def __init__(self, n_actions, input_shape=(1, 80, 80), value_dim=8):
        super(ActionValueEstimatorNet, self).__init__()
        self.phi = Phi()
        self.n_actions = n_actions

        self.w_a = nn.Bilinear(value_dim, 64, n_actions)

    def get_phi_state_dict(self):
        return self.phi.state_dict()

    def forward(self, input):
        state, subgoal = input
        h = self.phi(state)
        subgoal = subgoal.squeeze(1)
        q = self.w_a(subgoal, h)
        return q      
