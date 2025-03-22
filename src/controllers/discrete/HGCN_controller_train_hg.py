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
        """
        HGCNMAC 初始化方法
        :param scheme: 数据的定义结构（例如观测、动作）。
        :param groups: 群组信息。
        :param args: 参数定义。
        """
        super().__init__(scheme, groups, args)
        
        # 保存参数
        self.args = args
        self.n_agents = args.n_agents
        # self.hidden_state_size = args.hidden_state_size  # 隐藏状态维度
        self.n_hyper_edges = args.n_hyper_edges  # 超图的超边数量

        # 获取输入特征维度
        org_input_shape = self._get_input_shape(scheme)

        # 初始化 HGCN_EMBEDDING
        self.hgcn_embedding = HGCN_EMBEDDING(
            in_feature_dim=org_input_shape,        # 输入特征维度
            embedding_dim=args.hgcn_embedding_dim,  # 嵌入维度
            n_hyper_edges=self.n_hyper_edges,     # 超边数量
            n_agents=self.n_agents                # 智能体数量
        )

        
        # 使用 RNN 或 MLP 作为 agent 网络
        # self._build_agents(org_input_shape + args.hgcn_embedding_dim)  # 输入维度加上 HGCN 的嵌入维度
        if self.args.concate_gcn:
            self._build_agents(org_input_shape + args.hgcn_embedding_dim)
        else:
            self._build_agents(args.hgcn_embedding_dim)
        self.hidden_states = None

    def forward(self, ep_batch, t, test_mode=False):
        """
        前向传播方法
        :param ep_batch: 批量数据。
        :param t: 当前时间步。
        :param test_mode: 是否为测试模式。
        :return: agent 输出、超图嵌入。
        """
        batch_size = ep_batch.batch_size

        # 构造智能体输入
        agent_inputs = self._build_inputs(ep_batch, t)

        # 通过 HGCN_EMBEDDING 生成超图嵌入
        hgcn_embedding = self.hgcn_embedding(agent_inputs.view(batch_size, self.n_agents, -1))  

        # 将智能体输入和 HGCN 嵌入拼接
        if self.args.concate_gcn:
            agent_inputs = th.cat([agent_inputs, hgcn_embedding.view(batch_size * self.n_agents, -1)], dim=1)
        else:
            agent_inputs=hgcn_embedding
        # 通过 agent 网络
        agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)

        # 如果输出是策略 logits，应用 softmax
        if self.agent_output_type == "pi_logits":
            avail_actions = ep_batch["avail_actions"][:, t]
            reshaped_avail_actions = avail_actions.reshape(batch_size * self.n_agents, -1)
            if getattr(self.args, "mask_before_softmax", True):
                agent_outs[reshaped_avail_actions == 0] = -1e10
            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)

        # 返回智能体输出
        return agent_outs.view(batch_size, self.n_agents, -1)   #, hgcn_embedding

    def init_hidden(self, batch_size):
        """
        初始化隐藏状态
        :param batch_size: 批量大小。
        """
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)

    def parameters(self):
        """
        获取所有可优化参数
        """
        return itertools.chain(
            self.agent.parameters(),
            self.hgcn_embedding.parameters()  # 修改部分：确保 HGCN_EMBEDDING 的参数被优化
        )

    def cuda(self):
        """
        将模型移动到 GPU
        """
        self.agent.cuda()
        self.hgcn_embedding.cuda()  # 修改部分：确保 HGCN_EMBEDDING 被移动到 GPU

    def save_models(self, path):
        """
        保存模型
        :param path: 保存路径。
        """
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))
        th.save(self.hgcn_embedding.state_dict(), "{}/hgcn_embedding.th".format(path))  # 修改部分：保存 HGCN_EMBEDDING 的状态

    def load_models(self, path):
        """
        加载模型
        :param path: 加载路径。
        """
        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))
        self.hgcn_embedding.load_state_dict(th.load("{}/hgcn_embedding.th".format(path), map_location=lambda storage, loc: storage))  # 修改部分：加载 HGCN_EMBEDDING 的状态

    def _build_inputs(self, batch, t):
        """
        构造智能体的输入
        :param batch: 批量数据。
        :param t: 当前时间步。
        :return: 拼接后的输入。
        """
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
        """
        获取输入特征维度
        :param scheme: 数据结构。
        :return: 输入特征维度。
        """
        input_shape = scheme["obs"]["vshape"]

        # 上一步动作
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]

        # 智能体 ID
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape