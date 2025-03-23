import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import itertools
from ..basic_controller import BasicMAC
from src.components.HGCN_util_train_hg import HGCN_EMBEDDING  # 修改部分：引入 HGCN_EMBEDDING 模块

class HGCNMAC(BasicMAC):

    def __init__(self, scheme, groups, args):
        super().__init__(scheme, groups, args)
        

        self.args = args
        self.n_agents = args.n_agents

        self.n_hyper_edges = args.n_hyper_edges  


        org_input_shape = self._get_input_shape(scheme)


        self.hgcn_embedding = HGCN_EMBEDDING(
            in_feature_dim=org_input_shape,        
            embedding_dim=args.hgcn_embedding_dim,  
            n_hyper_edges=self.n_hyper_edges,     
            n_agents=self.n_agents                
        )

        

        if self.args.concate_gcn:
            self._build_agents(org_input_shape + args.hgcn_embedding_dim)
        else:
            self._build_agents(args.hgcn_embedding_dim)
        self.hidden_states = None

    def forward(self, ep_batch, t, test_mode=False):

        batch_size = ep_batch.batch_size

        agent_inputs = self._build_inputs(ep_batch, t)


        hgcn_embedding = self.hgcn_embedding(agent_inputs.view(batch_size, self.n_agents, -1))  


        if self.args.concate_gcn:
            agent_inputs = th.cat([agent_inputs, hgcn_embedding.view(batch_size * self.n_agents, -1)], dim=1)
        else:
            agent_inputs=hgcn_embedding

        agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)


        if self.agent_output_type == "pi_logits":
            avail_actions = ep_batch["avail_actions"][:, t]
            reshaped_avail_actions = avail_actions.reshape(batch_size * self.n_agents, -1)
            if getattr(self.args, "mask_before_softmax", True):
                agent_outs[reshaped_avail_actions == 0] = -1e10
            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)


        return agent_outs.view(batch_size, self.n_agents, -1)   

    def init_hidden(self, batch_size):

        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)

    def parameters(self):

        return itertools.chain(
            self.agent.parameters(),
            self.hgcn_embedding.parameters() 
        )

    def cuda(self):
    
        self.agent.cuda()
        self.hgcn_embedding.cuda()  

    def save_models(self, path):
   
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))
        th.save(self.hgcn_embedding.state_dict(), "{}/hgcn_embedding.th".format(path))  

    def load_models(self, path):

        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))
        self.hgcn_embedding.load_state_dict(th.load("{}/hgcn_embedding.th".format(path), map_location=lambda storage, loc: storage))  # 修改部分：加载 HGCN_EMBEDDING 的状态

    def _build_inputs(self, batch, t):

        bs = batch.batch_size
        inputs = []

        # 当前观测
        inputs.append(batch["obs"][:, t])

        # 上一步动作（如果需要）
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t - 1])

        # 智能体 ID（如果需要）
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        # 扁平化和拼接
        inputs = th.cat([x.reshape(bs * self.n_agents, -1) for x in inputs], dim=1)
        return inputs

    def _get_input_shape(self, scheme):

        input_shape = scheme["obs"]["vshape"]

        # 上一步动作
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]

        # 智能体 ID
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape