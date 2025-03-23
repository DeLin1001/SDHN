# code adapted from https://github.com/AnujMahajanOxf/MAVEN

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from components.HGCN_util import HGCN_EMBEDDING  

class CentralVCriticHg(nn.Module):
    def __init__(self, scheme, args):
        super(CentralVCriticHg, self).__init__()

        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.n_hyper_edges = args.n_hyper_edges  

        org_input_shape = self._get_input_shape(scheme)
        self.hgcn_input_shape = args.hgcn_input_shape #every single agent
        self.output_type = "v"


        
        self.hgcn_embedding = HGCN_EMBEDDING(
            in_feature_dim=self.hgcn_input_shape,  
            embedding_dim=args.hgcn_embedding_dim,
            n_hyper_edges=args.n_hyper_edges,
            n_agents=self.n_agents,
            args=args
        )

        if self.args.concate_gcn:
            # Set up network layers
            self.fc1 = nn.Linear(org_input_shape + args.hgcn_embedding_dim, args.hidden_dim)
            self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
            self.fc3 = nn.Linear(args.hidden_dim, 1)
        else:
            # Set up network layers
            self.fc1 = nn.Linear(args.hgcn_embedding_dim, args.hidden_dim)
            self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
            self.fc3 = nn.Linear(args.hidden_dim, 1)

        

       
        
        self.hgcn_hidden = None
        self.last_hyper_graph = None

    def forward(self, batch, t=None):
        inputs, bs, max_t = self._build_inputs(batch, t=t) #[batch_size,t,n_agents,input_shape]

        
        batch_size = inputs.size(0)
        hgcn_embedding=[]
        hyper_graphs=[]
        for t in range(batch.max_seq_length):
            hgcn_input = batch["obs"][:, t].view(batch_size, self.n_agents, -1)[:, :, :self.hgcn_input_shape]  
            hgcn_embedding_t, self.hgcn_hidden, current_hyper_graph = self.hgcn_embedding(
                hgcn_input, self.hgcn_hidden
                )
            hgcn_embedding.append(hgcn_embedding_t)
            hyper_graphs.append(current_hyper_graph)  # [batch_size, n_agents, n_hyper_edges]
        hgcn_embedding=th.stack(hgcn_embedding, dim=1)
        
        
        

        if self.args.concate_gcn:
            
            inputs = th.cat([inputs, hgcn_embedding], dim=-1)
        else:
            inputs = hgcn_embedding.view(batch_size, -1)

        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q, hyper_graphs #hyper_graphs:list,max_seq_length*[batch_size, n_agents, n_hyper_edges]

    def init_hidden(self, batch_size):
        
        self.hgcn_hidden = self.hgcn_embedding.init_hidden().unsqueeze(0).expand(batch_size,  -1)  # 新增


    def _build_inputs(self, batch, t=None):
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t+1)
        inputs = []
        # state
        inputs.append(batch["state"][:, ts].unsqueeze(2).repeat(1, 1, self.n_agents, 1))

        # observations
        if self.args.obs_individual_obs:
            inputs.append(batch["obs"][:, ts].view(bs, max_t, -1).unsqueeze(2).repeat(1, 1, self.n_agents, 1))

        # last actions
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, 0:1]).view(bs, max_t, 1, -1))
            elif isinstance(t, int):
                inputs.append(batch["actions_onehot"][:, slice(t-1, t)].view(bs, max_t, 1, -1))
            else:
                last_actions = th.cat([th.zeros_like(batch["actions_onehot"][:, 0:1]), batch["actions_onehot"][:, :-1]], dim=1)
                last_actions = last_actions.view(bs, max_t, 1, -1).repeat(1, 1, self.n_agents, 1)
                inputs.append(last_actions)

        inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).unsqueeze(0).expand(bs, max_t, -1, -1))

        inputs = th.cat(inputs, dim=-1)
        return inputs, bs, max_t

    def _get_input_shape(self, scheme):
        # state
        input_shape = scheme["state"]["vshape"]
        # observations
        if self.args.obs_individual_obs:
            input_shape += scheme["obs"]["vshape"] * self.n_agents
        # last actions
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0] * self.n_agents
        input_shape += self.n_agents
        return input_shape
