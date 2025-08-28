import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from copy import deepcopy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def hard_sample(logits, dim=-1):
    y_soft = F.softmax(logits, dim=-1)
    index = y_soft.max(dim, keepdim=True)[1]
    y_hard = torch.zeros_like(y_soft).scatter_(dim, index, 1.0)
    ret = y_hard - y_soft.detach() + y_soft
    return ret, index.squeeze(1)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var=256):
        super().__init__()
        self.obs_layer = nn.Linear(state_dim, n_latent_var)
        self.actor_layer = nn.Sequential(
            nn.Linear(n_latent_var, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, action_dim)
        )

    def forward(self, state, action_mask):
        hidden_state = self.obs_layer(state)
        logits = self.actor_layer(hidden_state)
        inf_mask = torch.clamp(torch.log(action_mask.float()),
                               min=torch.finfo(torch.float32).min)
        logits = logits + inf_mask
        train_mask, actions = hard_sample(logits)
        return train_mask, actions


class StraightThrough:
    def __init__(self, state_dim, action_dim, lr, betas):
        self.lr = lr
        self.betas = betas
        self.policy = Actor(state_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=lr, betas=betas)

    def update(self, loss):
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()

def main():
    pass


if __name__ == '__main__':
    main()
