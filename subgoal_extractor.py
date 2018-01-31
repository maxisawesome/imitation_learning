import torch
import sys
import torch.nn as nn
from torch.autograd import Variable

from sklearn.neighbors import KDTree
from phi import Phi

class SubgoalExtractorNet(nn.Module):
    """
    Subgoal extractor network proposed in the paper being reproduced. Uses
    expert state sequences to learn to imitate the expert's policy.
    """
    def __init__(self, expert_experiences, k=5, key_dim=16, value_dim=8, input_shape=(1, 80, 80)):
        super(SubgoalExtractorNet, self).__init__()
        
        self.phi = Phi()
        self.expert_experiences = expert_experiences
        self.k = k
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.w_k = nn.Linear(64, key_dim)
        self.w_v = nn.Linear(64, value_dim)
        
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()
  
        self.cossim = nn.CosineSimilarity()

    def update_phi(self, phi_state_dict):
        self.phi.load_state_dict(phi_state_dict)
        if torch.cuda.is_available():
            self.phi = self.phi.cuda()
        for param in self.phi.parameters():
            param.requires_grad = False
        for param in self.w_k.parameters():
            param.requires_grad = False
        for param in self.w_v.parameters():
            param.requires_grad = False
        self.memory_net = []
        for i in range(len(self.expert_experiences)):
            state, next_state = self.expert_experiences[i]
            key = self.w_k(self.phi(state))
            #value = self.w_v(self.phi(state) - self.phi(next_state))
            #self.memory_net.append((key, value, state, next_state))
            self.memory_net.append((key, state, next_state))
        if torch.cuda.is_available():
            keys = [exp[0].squeeze().cpu().data.numpy() for exp in self.memory_net]
        else:
            keys = [exp[0].squeeze().data.numpy() for exp in self.memory_net]
        
        for param in self.w_k.parameters():
            param.requires_grad = True
        for param in self.w_v.parameters():
            param.requires_grad = True
        self.KDTree = KDTree(keys)

    def forward(self, input):
        last_state, state, subgoal = input
        #for param in self.phi.parameters():
        #    param.detach()
        x = self.w_k(self.phi(state))
        if torch.cuda.is_available():
            state_key = x.cpu().data.numpy()
        else:
            state_key = x.data.numpy()

        knn = self.KDTree.query(state_key, self.k)[1][0]
        k_is = []
        v_is = []
        for i_sample in range(state.size()[0]):
            k_is.append([])
            v_is.append([])
            for i_knn in range(len(knn)):
                k_is[-1].append(self.w_k(self.phi(self.memory_net[knn[i_knn]][1])))
                v_is[-1].append(self.w_v(self.phi(self.memory_net[knn[i_knn]][1]) - self.phi(self.memory_net[knn[i_knn]][2])))
            k_is[-1] = torch.cat(k_is[-1]).unsqueeze(0)
            v_is[-1] = torch.cat(v_is[-1]).unsqueeze(0)
 
        k_is = torch.cat(k_is)
        v_is = torch.cat(v_is)
        
        x = x.unsqueeze(1)
        alphas = torch.bmm(x, k_is.permute(0,2,1))
        
        g_hat = self.relu(torch.bmm(alphas, v_is))
        
        g = g_hat / torch.norm(g_hat, p=2)
        guidance_reward = self.cossim(self.w_v(self.phi(last_state) - self.phi(state)), subgoal)
        return g, guidance_reward
