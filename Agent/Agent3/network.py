
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import HGTConv
from torch.distributions import Categorical

class BPScheduler(nn.Module):
    def __init__(self,
                 meta_data,
                 state_size,
                 num_nodes,
                 embed_dim,
                 num_heads,
                 num_HGT_layers,
                 num_actor_layers,
                 num_critic_layers,
                 use_parameter_sharing=True):

        super(BPScheduler, self).__init__()
        self.meta_data = meta_data
        self.state_size = state_size
        self.num_nodes = num_nodes
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_HGT_layers = num_HGT_layers                # HGT Layer 수
        self.num_actor_layers = num_actor_layers            # Actor의 층 개수
        self.num_critic_layers = num_critic_layers          # Critic의 층 개수
        self.use_parameter_sharing = use_parameter_sharing  # 파라미터를 누구와 공유하는 것인지?

        self.conv = nn.ModuleList()
        for i in range(self.num_HGT_layers):
            if i == 0:
                self.conv.append(HGTConv(self.state_size, embed_dim, meta_data, heads=num_heads))
            else:
                self.conv.append(HGTConv(embed_dim, embed_dim, meta_data, heads=num_heads))

        self.actor = nn.ModuleList()
        for i in range(self.num_actor_layers):
            if i == 0:
                self.actor.append(nn.Linear(embed_dim * 2, embed_dim))
            elif 0 < i < self.num_actor_layers - 1:
                self.actor.append(nn.Linear(embed_dim, embed_dim))
            else:
                self.actor.append(nn.Linear(embed_dim, 1))

        if use_parameter_sharing:
            self.critic = nn.ModuleList()
            for i in range(self.num_critic_layers):
                if i == 0:
                    self.critic.append(nn.Linear(embed_dim * 2, embed_dim))
                elif i < num_critic_layers - 1:
                    self.critic.append(nn.Linear(embed_dim, embed_dim))
                else:
                    self.critic.append(nn.Linear(embed_dim, 1))

    def act(self,
            graph_feature=None,
            mask=None,
            greedy=False):
        pass

    def evaluate(self,
                 batch_graph_feature=None,
                 batch_action=None,
                 batch_mask=None):
        pass



















