import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import HGTConv
from torch.distributions import Categorical


class BAScheduler(nn.Module):
    def __init__(self,
                 meta_data,
                 state_size,
                 num_nodes,
                 embed_dim,
                 num_heads,
                 num_HGT_layers,
                 num_actor_layers,
                 num_critic_layers,
                 use_parameter_sharing=True,
                 use_communication=True):

        super(BAScheduler, self).__init__()
        self.meta_data = meta_data
        self.state_size = state_size
        self.num_nodes = num_nodes
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_HGT_layers = num_HGT_layers
        self.num_actor_layers = num_actor_layers
        self.num_critic_layers = num_critic_layers
        self.use_parameter_sharing = use_parameter_sharing
        self.use_communication = use_communication

        self.num_bays = self.num_nodes["bay"]

        self.conv = nn.ModuleList()
        for i in range(self.num_HGT_layers):
            if i == 0:
                self.conv.append(HGTConv(self.state_size, embed_dim, meta_data, heads=num_heads))
            else:
                self.conv.append(HGTConv(embed_dim, embed_dim, meta_data, heads=num_heads))

        if self.use_communication:
            self.fc = nn.ModuleList()
            self.fc.append(nn.Linear(2, embed_dim))
            self.fc.append(nn.Linear(embed_dim, embed_dim))

        self.actor = nn.ModuleList()
        for i in range(num_actor_layers):
            if i == 0:
                if self.use_communication:
                    self.actor.append(nn.Linear(embed_dim * 3, embed_dim))
                else:
                    self.actor.append(nn.Linear(embed_dim, embed_dim))
            elif 0 < i < num_actor_layers - 1:
                self.actor.append(nn.Linear(embed_dim, embed_dim))
            else:
                self.actor.append(nn.Linear(embed_dim, 1))

        if use_parameter_sharing:
            self.critic = nn.ModuleList()
            for i in range(num_critic_layers):
                if i == 0:
                    self.critic.append(nn.Linear(embed_dim * 2, embed_dim))
                elif i < num_critic_layers - 1:
                    self.critic.append(nn.Linear(embed_dim, embed_dim))
                else:
                    self.critic.append(nn.Linear(embed_dim, 1))

    def act(self,
            graph_feature=None,
            pairwise_feature=None,
            mask=None,
            greedy=False):

        x_dict, edge_index_dict = graph_feature.x_dict, graph_feature.edge_index_dict

        for i in range(self.num_HGT_layers):
            x_dict = self.conv[i](x_dict, edge_index_dict)
            x_dict = {key: F.elu(x) for key, x in x_dict.items()}

        h_blocks = x_dict["block"]
        h_bays = x_dict["bay"]

        h_blocks_pooled = h_blocks.mean(dim=-2)
        h_bays_pooled = h_bays.mean(dim=-2)

        # h_blocks_pooled_padding = h_blocks_pooled.unsqueeze(-2).expand(h_bays.shape[0], -1)
        # h_actions = torch.cat((h_bays, h_blocks_pooled_padding), dim=-1)

        if self.use_communication:
            h_added = pairwise_feature.squeeze(0)
            for i in range(self.num_HGT_layers):
                h_added = self.fc[i](h_added)
                h_added = F.elu(h_added)

            h_blocks_padding = h_blocks.expand(self.num_bays, -1)
            h_actions = torch.cat((h_bays, h_blocks_padding, h_added), dim=-1)

        else:
            h_actions = h_bays

        for i in range(self.num_actor_layers):
            if i < len(self.actor) - 1:
                h_actions = self.actor[i](h_actions)
                h_actions = F.elu(h_actions)
            else:
                logits = self.actor[i](h_actions).flatten()

        logits[~mask] = float('-inf')
        probs = F.softmax(logits, dim=-1)

        dist = Categorical(probs)

        if greedy:
            action = torch.argmax(probs)
            action_logprob = dist.log_prob(action)
        else:
            action = dist.sample()
            action_logprob = dist.log_prob(action)
            while action_logprob < -15:
                action = dist.sample()
                action_logprob = dist.log_prob(action)

        if self.use_parameter_sharing:
            h_pooled = torch.cat((h_bays_pooled, h_blocks_pooled), dim=-1)

            for i in range(self.num_critic_layers):
                if i < len(self.critic) - 1:
                    h_pooled = self.critic[i](h_pooled)
                    h_pooled = F.elu(h_pooled)
                else:
                    state_value = self.critic[i](h_pooled)

        if self.use_parameter_sharing:
            return action.item(), action_logprob.item(), state_value.squeeze().item(), probs
        else:
            return action.item(), action_logprob.item()

    def evaluate(self,
                 batch_graph_feature=None,
                 batch_pairwise_feature=None,
                 batch_action=None,
                 batch_mask=None):

        batch_size = batch_graph_feature.num_graphs
        x_dict, edge_index_dict = batch_graph_feature.x_dict, batch_graph_feature.edge_index_dict

        for i in range(self.num_HGT_layers):
            x_dict = self.conv[i](x_dict, edge_index_dict)
            x_dict = {key: F.elu(x) for key, x in x_dict.items()}

        h_blocks = x_dict["block"].unsqueeze(0).reshape(batch_size, -1, self.embed_dim)
        h_bays = x_dict["bay"].unsqueeze(0).reshape(batch_size, -1, self.embed_dim)

        h_blocks_pooled = h_blocks.mean(dim=-2)
        h_bays_pooled = h_bays.mean(dim=-2)

        # h_blocks_pooled_padding = h_blocks_pooled.unsqueeze(-2).expand(-1, h_bays.shape[0], -1)
        # h_actions = torch.cat((h_blocks, h_blocks_pooled_padding), dim=-1)

        if self.use_communication:
            h_added = batch_pairwise_feature.squeeze(1)
            for i in range(self.num_HGT_layers):
                h_added = self.fc[i](h_added)
                h_added = F.elu(h_added)

            h_blocks_padding = h_blocks.expand(-1, self.num_bays, -1)
            h_actions = torch.cat((h_bays, h_blocks_padding, h_added), dim=-1)

        else:
            h_actions = h_bays

        for i in range(self.num_actor_layers):
            if i < len(self.actor) - 1:
                h_actions = self.actor[i](h_actions)
                h_actions = F.elu(h_actions)
            else:
                batch_logits = self.actor[i](h_actions).flatten(1)

        batch_mask = batch_mask
        batch_logits[~batch_mask] = float('-inf')
        batch_probs = F.softmax(batch_logits, dim=1)
        batch_dist = Categorical(batch_probs)
        batch_action_logprobs = batch_dist.log_prob(batch_action.squeeze()).unsqueeze(-1)

        if self.use_parameter_sharing:
            h_pooled = torch.cat((h_bays_pooled, h_blocks_pooled), dim=-1)

            for i in range(self.num_critic_layers):
                if i < len(self.critic) - 1:
                    h_pooled = self.critic[i](h_pooled)
                    h_pooled = F.elu(h_pooled)
                else:
                    batch_state_values = self.critic[i](h_pooled)

        batch_dist_entropys = batch_dist.entropy().unsqueeze(-1)

        if self.use_parameter_sharing:
            return batch_action_logprobs, batch_state_values, batch_dist_entropys
        else:
            return batch_action_logprobs, batch_dist_entropys