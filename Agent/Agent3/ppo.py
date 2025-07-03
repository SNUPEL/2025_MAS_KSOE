import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from torch.optim.lr_scheduler import StepLR
from torch_geometric.data import Batch
from Agent.Agent3.network import BPScheduler


class Agent3:
    def __init__(self, meta_data,
                 state_size,
                 num_nodes,
                 embed_dim,
                 num_heads,
                 num_HGT_layers,
                 num_actor_layers,
                 num_critic_layers,
                 lr,
                 lr_decay,
                 lr_step,
                 gamma,
                 lmbda,
                 eps_clip,
                 K_epoch,
                 P_coeff,
                 V_coeff,
                 E_coeff,
                 use_value_clipping=True,
                 use_parameter_sharing=True,
                 use_communication=True,
                 device="cpu"):
        pass

    def put_sample(self, state, action, reward, done, log_prob, value):
        pass

    def get_action(self, state):
        pass

    def train(self, last_value):
        pass


    def save_network(self, episode, file_dir):
        pass