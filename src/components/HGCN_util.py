import torch
# from torch._C import INSERT_FOLD_PREPACK_OPS, Node
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import itertools

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter

from torch_geometric.experimental import disable_dynamic_shapes
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import scatter, softmax
import os
class HypergraphConv(MessagePassing):
    r"""The hypergraph convolutional operator from the `"Hypergraph Convolution
    and Hypergraph Attention" <https://arxiv.org/abs/1901.08150>`_ paper.

    .. math::
        \mathbf{X}^{\prime} = \mathbf{D}^{-1} \mathbf{H} \mathbf{W}
        \mathbf{B}^{-1} \mathbf{H}^{\top} \mathbf{X} \mathbf{\Theta}

    where :math:`\mathbf{H} \in {\{ 0, 1 \}}^{N \times M}` is the incidence
    matrix, :math:`\mathbf{W} \in \mathbb{R}^M` is the diagonal hyperedge
    weight matrix, and
    :math:`\mathbf{D}` and :math:`\mathbf{B}` are the corresponding degree
    matrices.

    For example, in the hypergraph scenario
    :math:`\mathcal{G} = (\mathcal{V}, \mathcal{E})` with
    :math:`\mathcal{V} = \{ 0, 1, 2, 3 \}` and
    :math:`\mathcal{E} = \{ \{ 0, 1, 2 \}, \{ 1, 2, 3 \} \}`, the
    :obj:`hyperedge_index` is represented as:

    .. code-block:: python

        hyperedge_index = torch.tensor([
            [0, 1, 2, 1, 2, 3],
            [0, 0, 0, 1, 1, 1],
        ])

    Args:
        in_channels (int): Size of each input sample, or :obj:`-1` to derive
            the size from the first input(s) to the forward method.
        out_channels (int): Size of each output sample.
        use_attention (bool, optional): If set to :obj:`True`, attention
            will be added to this layer. (default: :obj:`False`)
        attention_mode (str, optional): The mode on how to compute attention.
            If set to :obj:`"node"`, will compute attention scores of nodes
            within all nodes belonging to the same hyperedge.
            If set to :obj:`"edge"`, will compute attention scores of nodes
            across all edges holding this node belongs to.
            (default: :obj:`"node"`)
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the multi-head
            attentions are averaged instead of concatenated.
            (default: :obj:`True`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})`,
          hyperedge indices :math:`(|\mathcal{V}|, |\mathcal{E}|)`,
          hyperedge weights :math:`(|\mathcal{E}|)` *(optional)*
          hyperedge features :math:`(|\mathcal{E}|, D)` *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, F_{out})`
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_attention: bool = False,
        attention_mode: str = 'node',
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0,
        bias: bool = True,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(flow='source_to_target', node_dim=0, **kwargs)

        assert attention_mode in ['node', 'edge']

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_attention = use_attention
        self.attention_mode = attention_mode

        if self.use_attention:
            self.heads = heads
            self.concat = concat
            self.negative_slope = negative_slope
            self.dropout = dropout
            self.lin = Linear(in_channels, heads * out_channels, bias=False,
                              weight_initializer='glorot')
            self.att = Parameter(torch.empty(1, heads, 2 * out_channels))
        else:
            self.heads = 1
            self.concat = True
            self.lin = Linear(in_channels, out_channels, bias=False,
                              weight_initializer='glorot')

        if bias and concat:
            self.bias = Parameter(torch.empty(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin.reset_parameters()
        if self.use_attention:
            glorot(self.att)
        zeros(self.bias)

    @disable_dynamic_shapes(required_args=['num_edges'])
    def forward(self, x: Tensor, hyperedge_index: Tensor,
                hyperedge_weight: Optional[Tensor] = None,
                hyperedge_attr: Optional[Tensor] = None,
                num_edges: Optional[int] = None) -> Tensor:
        r"""Runs the forward pass of the module with batch support.

        Args:
            x (torch.Tensor): Batched node feature matrix
                :math:`\mathbf{X} \in \mathbb{R}^{B \times N \times F}`.
            hyperedge_index (torch.Tensor): Batched hyperedge indices, *i.e.*
                the sparse incidence matrix
                :math:`\mathbf{H} \in {\{ 0, 1 \}}^{B \times 2 \times E}` mapping
                nodes to edges.
            hyperedge_weight (torch.Tensor, optional): Batched hyperedge weights
                :math:`\mathbf{W} \in \mathbb{R}^{B \times E}`.
                (default: :obj:`None`)
            hyperedge_attr (torch.Tensor, optional): Batched hyperedge feature matrix
                :math:`\mathbf{H_{attr}} \in \mathbb{R}^{B \times E \times F}`.
                These features only need to be passed in case
                :obj:`use_attention=True`. (default: :obj:`None`)
            num_edges (torch.Tensor, optional): The number of edges :math:`M` for
                each batch. (default: :obj:`None`)
        """
        batch_size, num_nodes, in_channels = x.size()  # Batch size, nodes per batch, input feature size
        _, _, total_edges = hyperedge_index.size()  # Batch size, 2, total edges per batch

        # Flatten node features for batch processing
        x = x.view(-1, in_channels)  # Shape: [B * N, F]

        # Add batch offsets to hyperedge_index  try to fix ../aten/src/ATen/native/cuda/ScatterGatherKernel.cu:144: 
        # operator(): block: [1,0,0], thread: [63,0,0] Assertion `idx_dim >= 0 && idx_dim < index_size && "index out of bounds"` failed.
        # node_offsets = torch.arange(batch_size, device=x.device).view(-1, 1) * num_nodes
        # node_offsets = node_offsets.repeat(1, total_edges).view(-1)  # Shape: [B * E]
        # hyperedge_index = hyperedge_index.view(2, -1) + node_offsets  # Shape: [2, B * E]  2 是边的两个端点，B*E 是边的数量

        node_offsets = torch.arange(batch_size, device=x.device).view(-1, 1) * num_nodes
        node_offsets = node_offsets.repeat(1, total_edges).view(-1)  # Shape: [B * E]
        hyperedge_index = hyperedge_index.view(2, -1)
        hyperedge_index[0] = hyperedge_index[0] + node_offsets  # Shape: [2, B * E]
      
      
        # hyperedge_index = hyperedge_index.view(2, -1)
        # node_offsets = torch.arange(batch_size, device=x.device).unsqueeze(1) * num_nodes  # Shape: [B, 1]
        # node_offsets = node_offsets.repeat(1, total_edges).view(-1)  # Shape: [B * E]
        # hyperedge_index[0] = hyperedge_index[0] + node_offsets  # 只对节点索引加偏移
        # Flatten hyperedge weights
        if hyperedge_weight is None:
            hyperedge_weight = x.new_ones(batch_size, total_edges)
        hyperedge_weight = hyperedge_weight.view(-1)  # Shape: [B * E]

        # Compute the number of edges per batch
        if num_edges is None:
            num_edges = int(hyperedge_index[1].max()) + 1  

        # Apply linear transformation
        x = self.lin(x)  # Shape: [B * N, out_channels] or [B * N, heads * out_channels]

        alpha = None
        if self.use_attention:
            assert hyperedge_attr is not None, "Hyperedge attributes must be provided when use_attention=True."
            
            # Prepare node and hyperedge attributes for attention
            x = x.view(-1, self.heads, self.out_channels)  # Shape: [B * N, heads, out_channels]
            hyperedge_attr = self.lin(hyperedge_attr.view(-1, hyperedge_attr.size(-1)))  # Shape: [B * E, heads * out_channels]
            hyperedge_attr = hyperedge_attr.view(-1, self.heads, self.out_channels)  # Shape: [B * E, heads, out_channels]

            # Attention mechanism
            x_i = x[hyperedge_index[0]]  # Source node features
            x_j = hyperedge_attr[hyperedge_index[1]]  # Target hyperedge features
            alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)  # Shape: [B * E, heads]
            alpha = F.leaky_relu(alpha, self.negative_slope)
            
            # Apply softmax normalization
            if self.attention_mode == 'node':
                alpha = softmax(alpha, hyperedge_index[1], num_nodes=num_edges)  # Normalize over edges
            else:  # 'edge' mode
                alpha = softmax(alpha, hyperedge_index[0], num_nodes=num_nodes)  # Normalize over nodes
            
            # Apply dropout to attention scores
            alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # Compute degree matrices
        D = scatter(hyperedge_weight[hyperedge_index[1]], hyperedge_index[0], dim=0, dim_size=batch_size * num_nodes, reduce='sum') #tensor([8., 9.], device='cuda:0', grad_fn=<ScatterAddBackward0>)
        D = 1.0 / D
        D[D == float("inf")] = 0

        B = scatter(x.new_ones(hyperedge_index.size(1)), hyperedge_index[1], dim=0, dim_size=batch_size * num_edges, reduce='sum')
        B = 1.0 / B
        B[B == float("inf")] = 0

        # First propagation: Node -> Hyperedge
        out = self.propagate(edge_index=hyperedge_index, x=x, norm=B, alpha=alpha, size=(batch_size * num_nodes, batch_size * num_edges))

        # Second propagation: Hyperedge -> Node
        out = self.propagate(edge_index=hyperedge_index.flip([0]), x=out, norm=D, alpha=alpha, size=(batch_size * num_edges, batch_size * num_nodes))

        # Reshape output
        if self.concat:
            out = out.view(batch_size, num_nodes, -1)  # Shape: [B, N, heads * out_channels]
        else:
            out = out.view(batch_size, num_nodes, self.heads, self.out_channels).mean(dim=2)  # Shape: [B, N, out_channels]

        # Add bias if applicable
        if self.bias is not None:
            out = out + self.bias

        return out
        def message(self, x_j: Tensor, norm_i: Tensor, alpha: Tensor) -> Tensor:
            H, F = self.heads, self.out_channels

            out = norm_i.view(-1, 1, 1) * x_j.view(-1, H, F)

            if alpha is not None:
                out = alpha.view(-1, self.heads, 1) * out

            return out

import torch
import torch.nn as nn
from torch import Tensor

class DenseHypergraphConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bias: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)  # Glorot初始化权重
        if self.bias is not None:
            nn.init.zeros_(self.bias)        # 偏置初始化为零

    def forward(self, x: Tensor, H: Tensor, hyperedge_weight: Tensor = None) -> Tensor:
        """
        Args:
            x (Tensor): 节点特征，形状为 [B, N, in_channels]
            H (Tensor): 密集入射矩阵，形状为 [B, N, M]
            hyperedge_weight (Tensor, optional): 超边权重，形状为 [B, M]
        Returns:
            Tensor: 输出节点特征，形状为 [B, N, out_channels]
        """
        B, N, M = H.shape
        
        # 1. 计算度矩阵 D (节点度) 和 B (超边度)
        if hyperedge_weight is not None:
            W = hyperedge_weight.unsqueeze(1)  # [B, 1, M]
            D = (H * W).sum(dim=2)             # [B, N]
            B_degree = (H * W).sum(dim=1)      # [B, M]
        else:
            D = H.sum(dim=2)                   # [B, N]
            B_degree = H.sum(dim=1)            # [B, M]
        
        # 2. 计算正则化的 D^{-1} 和 B^{-1}
        D_inv = torch.reciprocal(D + 1e-8).unsqueeze(-1)  # [B, N, 1]
        B_inv = torch.reciprocal(B_degree + 1e-8).unsqueeze(1)  # [B, 1, M]
        
        # 3. 核心公式: X' = D^{-1} H (W B^{-1}) H^T X W_theta
        # 分步计算以优化内存
        # (a) X W_theta: [B, N, in] @ [in, out] -> [B, N, out]
        x_transformed = torch.matmul(x, self.weight)
        
        # (b) H^T X_transformed: [B, M, N] @ [B, N, out] -> [B, M, out]
        H_T = H.transpose(1, 2)  # [B, M, N]
        msg = torch.bmm(H_T, x_transformed)  # [B, M, out]
        
        # (c) Apply W B^{-1} (若有权重)
        if hyperedge_weight is not None:
            W_B_inv = hyperedge_weight.unsqueeze(-1) * B_inv.unsqueeze(-1)  # [B, M, 1]
            msg = msg * W_B_inv  # [B, M, out]
        
        # (d) H (msg): [B, N, M] @ [B, M, out] -> [B, N, out]
        aggregated = torch.bmm(H, msg)  # [B, N, out]
        
        # (e) Apply D^{-1}
        out = aggregated * D_inv  # [B, N, out]
        
        # 4. 添加偏置
        if self.bias is not None:
            out += self.bias.view(1, 1, -1)
        
        return out

class Encoder(nn.Module):
    def __init__(self, aggregator, feature_dim):
        super(Encoder, self).__init__()
        self.aggregator = aggregator
        self.feature_dim = feature_dim

    def forward(self, node_features, hyper_graph):
        output = self.aggregator.forward(node_features, hyper_graph)
        return output


class RNNgraph_constructor(nn.Module):
    def __init__(self, input_shape, hidden_dim, n_hyper_edges,n_agents):
        super(RNNgraph_constructor, self).__init__()
        self.input_shape = input_shape
        self.hidden_dim = hidden_dim
        self.n_hyper_edges=n_hyper_edges
        self.n_agents=n_agents
        self.hyper_graph_mat=self.n_hyper_edges*self.n_agents  # 超图邻接矩阵大小
        self.fc1 = nn.Linear(input_shape, hidden_dim)
        self.rnn = nn.GRUCell(hidden_dim, hidden_dim)

        self.fc2 = nn.Linear(hidden_dim, self.hyper_graph_mat)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new( self.hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        # print(f"RNNgraph_constructor forward: inputs shape: {inputs.shape}")  # [batch_size *n_agents, input_shape]
        # print(f"RNNgraph_constructor forward: hidden_state shape: {hidden_state.shape}")  # [batch_size *n_agents, hidden_dim]
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.hidden_dim)
        h = self.rnn(x, h_in)
        # graph = self.gumbel_sigmoid(self.fc2(h), 1)
        # print(f"RNNgraph_constructor forward: h shape: {h.shape}")  # [batch_size *n_agents, hidden_dim]
        graph = torch.sigmoid(self.fc2(h))
        # print(f"RNNgraph_constructor forward: graph shape: {graph.shape}")  # [batch_size *n_agents, n_hyper_edges]
        return graph, h
    
class RNNgraph_constructor_bernoulli(nn.Module):
    def __init__(self, input_shape, hidden_dim, n_hyper_edges,n_agents):
        """
        超图生成网络，使用 RNN 生成超图邻接矩阵的分布。
        :param input_shape: 输入维度（例如所有节点特征拼接后的维度）。
        :param hidden_dim: RNN 隐藏状态的维度。
        :param hyper_graph_mat: 超图邻接矩阵的大小 (n_agents * n_hyper_edges)。
        """
        super(RNNgraph_constructor_bernoulli, self).__init__()
        self.input_shape = input_shape
        self.hidden_dim = hidden_dim
        self.n_hyper_edges=n_hyper_edges
        self.n_agents=n_agents
        self.hyper_graph_mat=self.n_hyper_edges*self.n_agents  # 超图邻接矩阵大小

        # 定义网络层
        self.fc1 = nn.Linear(input_shape, hidden_dim)  # 输入到隐藏层
        self.rnn = nn.GRUCell(hidden_dim, hidden_dim)  # 使用 GRU 作为 RNN
        self.fc2 = nn.Linear(hidden_dim, self.hyper_graph_mat)  # 隐藏层到超图邻接矩阵参数

    def init_hidden(self):
        """
        初始化隐藏状态，形状为 (hidden_dim)。
        """
        return self.fc1.weight.new( self.hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        """
        前向传播，生成超图邻接矩阵的分布参数，并通过 Gumbel-Softmax 采样生成超图。
        :param inputs: 输入特征，形状为 [batch_size * n_agents, input_shape]。
        :param hidden_state: 隐藏状态，形状为 [batch_size * n_agents, hidden_dim]。
        :return: 采样后的超图邻接矩阵 H 和更新后的隐藏状态。
        """
        # 输入特征通过第一层全连接层
        x = F.relu(self.fc1(inputs))  # [batch_size * n_agents, hidden_dim]

        # 更新 RNN 隐藏状态
        h_in = hidden_state.reshape(-1, self.hidden_dim)  # [batch_size * n_agents, hidden_dim]
        h = self.rnn(x, h_in)  # [batch_size * n_agents, hidden_dim]

        # 输出超图邻接矩阵的分布参数 (logits)
        logits = torch.sigmoid(self.fc2(h)) # [batch_size * n_agents, hyper_graph_mat]

        # Gumbel-Softmax 采样
        logits = logits.view(inputs.size(0), -1)  # [batch_size, n_agents * n_hyper_edges]
        sampled_H = self.gumbel_sigmoid(logits, tau=1, hard=True)  # [batch_size, n_agents * n_hyper_edges]

        # 将采样结果 reshape 成超图邻接矩阵的形状
        sampled_H = sampled_H.view(inputs.size(0), self.n_agents, self.n_hyper_edges)  # [batch_size, n_agents, n_hyper_edges]

        return sampled_H, h  # 返回采样后的超图邻接矩阵和更新后的隐藏状态
    def gumbel_sigmoid(self,logits, tau=1.0, hard=False):
        noise = torch.rand_like(logits)
        gumbel_noise = -torch.log(-torch.log(noise + 1e-20) + 1e-20)
        y = torch.sigmoid((logits + gumbel_noise) / tau)
        if hard:
            y_hard = (y > 0.5).float()
            y = (y_hard - y).detach() + y  # 梯度通过 y，但前向值为 y_hard
        return y
#TODO: 当前直接训练hyper graph，我希望能够增加一个神经网络hyper_graph_construction，
# hyper_graph_construction 是一个RNN负责根据输入信息，生成hyper graph ,后续hyper graph再按照现在的方式，进行卷积，生成embedding 
#hyper_graph_construction是可训练的，hyper graph不可训练
# rnn的具体使用方式，参考HGCN_controller中对agent的处理方式：# 通过 agent 网络 agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)

class HGCN_EMBEDDING(nn.Module):
    def __init__(self, in_feature_dim, embedding_dim, n_hyper_edges, n_agents, hyper_graph=None, args=None):
        """
        HGCN_EMBEDDING 初始化方法。
        :param in_feature_dim: 输入特征的维度 (比如每个节点的特征维度)。
        :param embedding_dim: 输出嵌入的维度。
        :param n_hyper_edges: 超图的超边数量。
        :param n_agents: 节点数量（智能体数量）。
        :param hyper_graph: 可选参数，初始化超图邻接矩阵，形状为 [n_agents, n_hyper_edges]。
                            如果为 None，则随机初始化一个可训练的超图。
        """
        
        super(HGCN_EMBEDDING, self).__init__()
        
        self.n_agents = n_agents
        self.n_hyper_edges = n_hyper_edges
        self.in_feature_dim=in_feature_dim
        self.embedding_dim=embedding_dim
        self.graph_update_interval = args.graph_update_interval
        if args.hg_generate_type == "certain": 
            self.hyper_graph_construction = RNNgraph_constructor(
                input_shape=self.in_feature_dim*self.n_agents,
                hidden_dim=self.n_hyper_edges*self.n_agents,
                n_hyper_edges=self.n_hyper_edges,
                n_agents=self.n_agents,
            ) #RNN用于生成hyper_graph
        elif args.hg_generate_type == "bernoulli":
            self.hyper_graph_construction = RNNgraph_constructor_bernoulli(
                input_shape=self.in_feature_dim*self.n_agents,
                hidden_dim=self.n_hyper_edges*self.n_agents,
                n_hyper_edges=self.n_hyper_edges,
                n_agents=self.n_agents
            ) #RNN用于生成hyper_graph
        else:
            #error
            raise ValueError(f"unsupported hg_generate_type: {args.hg_generate_type}")
        
        
        # 将 HGCN 替换为 HypergraphConv
        # self.hgcn_layer = HypergraphConv(
        #     in_channels=in_feature_dim,          # 输入特征维度
        #     out_channels=embedding_dim,         # 输出嵌入维度
        #     use_attention=False,                 # 根据需求设置是否使用注意力机制
        #     attention_mode='node',               # 注意力模式
        #     heads=1,                             # 多头注意力的头数
        #     concat=True,                         # 是否连接多头的输出
        #     dropout=0.0,                         # dropout 概率
        #     bias=True                            # 是否使用偏置
        # )  

        self.hgcn_layer = DenseHypergraphConv(
            in_channels=in_feature_dim,          # 输入特征维度
            out_channels=embedding_dim,         # 输出嵌入维度
            bias=True                            # 是否使用偏置
        )
        
        # 输出维度调整，确保输出为指定的 embedding_dim
        self.output_layer = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU()
        )

        # 用于缓存上一次计算的结果
        self.last_edge_index = None
        self.last_edge_weight = None
        self.last_hyper_graph = None
        self.t_hg_update=0
        self.current_batch_size=1

    def init_hidden(self):
        """
        初始化RNN的隐藏状态。
        :param batch_size: 批量大小。
        :return: 隐藏状态，形状为 [batch_size * n_agents, n_hyper_edges]
        """
        # 初始化隐藏状态为零
        return self.hyper_graph_construction.init_hidden()

    def forward(self, input,hidden_states):
        """
        前向传播方法，使用 hyper_graph_construction RNN 生成 Hyper Graph，并使用 HypergraphConv 生成嵌入。
        :param input: 智能体输入，形状为 [batch_size, n_agents, in_feature_dim]
        :param hidden_states: 当前的隐藏状态，形状为 [batch_size , n_hyper_edges*self.n_agents*2]
        :return: 嵌入，形状为 [batch_size * n_agents, embedding_dim]
                 更新后的隐藏状态，形状为 [batch_size * n_agents, n_hyper_edges]
        """
        
        input=input.contiguous()
        edge_index= None
        edge_weight= None
        batch_size = input.size(0)  # 获取批量大小
        if self.current_batch_size!=batch_size: #if batch_size changed, reset the hyper_graph_construction
            self.current_batch_size=batch_size
            self.t_hg_update=0
        if self.t_hg_update % self.graph_update_interval == 0:
            self.t_hg_update=0
            hyper_graph, hidden_states = self.hyper_graph_construction(input.view(batch_size, -1), hidden_states)  # [batch_size * n_agents, n_hyper_edges]
            hyper_graph = hyper_graph.view(batch_size, self.n_agents, self.n_hyper_edges)
            # 生成新的 edge_index, edge_weight
            edge_index, edge_weight = self.build_edge_index(hyper_graph)
            # 缓存
            self.last_hyper_graph = hyper_graph
            self.last_edge_index = edge_index
            self.last_edge_weight = edge_weight
        else:
            
            # 复用缓存的 hyper_graph / edge_index / edge_weight
            hyper_graph = self.last_hyper_graph.detach()
            edge_index = self.last_edge_index.detach()
            edge_weight = self.last_edge_weight.detach()
        self.t_hg_update+=1

        x = self.hgcn_layer(input, hyper_graph)  # [batch_size * n_agents, embedding_dim]
        # # 通过 HypergraphConv 层
        # x = self.hgcn_layer(input, edge_index, edge_weight)  # [batch_size * n_agents, embedding_dim]

        # 通过输出层
        embeddings = self.output_layer(x)  # [batch_size * n_agents, embedding_dim]
        # 打印最终嵌入的形状
        # print(f"forward: embeddings shape: {embeddings.shape}")  # [batch_size *n_agents, embedding_dim]
        
        return embeddings, hidden_states, hyper_graph  # 返回嵌入

    def build_edge_index(self, hyper_graph):
        """
        构建 edge_index 和 edge_weight，返回形状为 [batch_size, 2, max_edges]
        与 [batch_size, max_edges]。与原函数调用方式和输出格式相同，避免
        影响其余代码。

        参数:
        --------
        hyper_graph: torch.Tensor
            形状为 [batch_size, n_agents, n_hyper_edges] 的超图关联矩阵；
            其中 hyper_graph[b, i, j] != 0 表示第 b 个 batch 的第 i 个节点
            与第 j 个超边相连，值通常是某种权重或 1/0。

        返回:
        --------
        all_edge_index: torch.Tensor
            形状为 [batch_size, 2, max_edges]，其中:
            - all_edge_index[b, 0, :] 表示第 b 个 batch 的“节点”索引
            - all_edge_index[b, 1, :] 表示第 b 个 batch 的“超边”索引(内部加 n_agents)
            这里尚未对“batch”进行额外偏移(比如 b × num_nodes)，
            因为在 HypergraphConv.forward() 中会再次加 node_offsets。

        all_edge_weight: torch.Tensor
            形状为 [batch_size, max_edges]，对应每条 (node, hyperedge) 连边的权重。
            若某些条目为 padding，则为 0。

        注意:
        --------
        1. 仅在此处做 “hyperedge_indices + n_agents” 用以区分节点索引与超边索引，
        不对 batch 维度做额外偏移，从而避免与 HypergraphConv.forward() 中
        的 node_offsets 叠加造成越界。
        2. 当某个 batch 里 hyper_graph 全为 0 时，这意味着该 batch 无任何边。
        此处会添加最小占位处理(即空边)，并在后续 padding。
        如果不想这样处理，可根据需要修改逻辑，例如直接跳过该 batch 的 GNN。
        3. 仍保持 [batch_size, 2, max_edges] 形状的返回，以兼容后续可能的
        padding 及其他逻辑。
        """
        device = hyper_graph.device

        batch_size, n_agents, n_hyper_edges = hyper_graph.shape
        edge_indices_per_batch = []
        edge_weights_per_batch = []

        max_edges = 0  # 跟踪所有 batch 中的最大边数量, 用于后续 padding

        # 逐个 batch 处理
        for b in range(batch_size):
            # 取出第 b 个 batch 的超图
            H = hyper_graph[b]  # [n_agents, n_hyper_edges]

            # 找到非零元素(节点 i, 超边 j)
            agent_indices, hyperedge_indices = H.nonzero(as_tuple=False).t()

            if agent_indices.numel() == 0:
                # 若全为 0，意味着没有任何边
                # 可以选择直接空边, 后面 padding 时会补齐
                # 或者根据业务需求在此添加一条“虚拟边”
                # 这里示例: 不加任何边, 仅留空
                empty_edge_idx = torch.empty((2, 0), dtype=torch.long, device=device)
                empty_edge_w = torch.empty((0,), dtype=hyper_graph.dtype, device=device)
                edge_indices_per_batch.append(empty_edge_idx)
                edge_weights_per_batch.append(empty_edge_w)
                continue

            # 分别处理节点与超边索引:
            # 节点范围 [0, n_agents-1], 超边范围 [n_agents, n_agents + n_hyper_edges - 1]
            nodes = agent_indices
            hyperedges = hyperedge_indices #+ b*n_agents

            # 拼成二部图格式: 0 行是节点索引，1 行是超边索引
            edge_index_b = torch.stack([nodes, hyperedges], dim=0)  # [2, E]

            # 取对应的权重
            weights_b = H[agent_indices, hyperedge_indices]  # [E]

            edge_indices_per_batch.append(edge_index_b)
            edge_weights_per_batch.append(weights_b)

            # 更新 max_edges
            max_edges = max(max_edges, edge_index_b.size(1))

        # 针对每个 batch 做 padding，使它们有相同数量的边
        all_edge_index_list = []
        all_edge_weight_list = []

        for b in range(batch_size):
            edge_index_b = edge_indices_per_batch[b]
            weight_b = edge_weights_per_batch[b]
            e_count = edge_index_b.size(1)

            # pad edge_index
            if e_count < max_edges:
                # 构造一个 2 x max_edges 的空张量
                pad_idx = torch.zeros((2, max_edges), dtype=torch.long, device=device)
                pad_idx[:, :e_count] = edge_index_b
            else:
                pad_idx = edge_index_b

            # pad edge_weight
            if e_count < max_edges:
                pad_w = torch.zeros((max_edges,), dtype=hyper_graph.dtype, device=device)
                pad_w[:e_count] = weight_b
            else:
                pad_w = weight_b

            all_edge_index_list.append(pad_idx)
            all_edge_weight_list.append(pad_w)

        # 将 list 拼成 [batch_size, 2, max_edges]
        all_edge_index = torch.stack(all_edge_index_list, dim=0)
        all_edge_weight = torch.stack(all_edge_weight_list, dim=0)

        return all_edge_index, all_edge_weight
    

    
    # def build_edge_index(self, hyper_graph):
    #     """
    #     构建 edge_index 和 edge_weight 对应于 hyper_graph（batch_size, n_agents, n_hyper_edges）。
    #     这里我们将超图转换为 bipartite graph 的 edge_index，其中节点索引和超边的索引分开处理。

    #     :param hyper_graph: [batch_size, n_agents, n_hyper_edges]
    #     :return: edge_index [batch_size, 2, total_E], edge_weight [batch_size, total_E]
    #     """
        
    #     batch_size, n_agents, n_hyper_edges = hyper_graph.size()
            
    #     edge_indices = []
    #     edge_weights = []
    #     max_edges = 0  # 用于记录所有批次中最大的边数量
    #     debug_b=None
    #     debug_H=None
    #     for b in range(batch_size):
    #         H = hyper_graph[b]  # [n_agents, n_hyper_edges]
    #         H=torch.zeros_like(H)
    #         # 找出 H 中的非零元素
    #         agent_indices, hyperedge_indices = H.nonzero(as_tuple=False).t()  # [E]

    #         if agent_indices.numel() == 0:
    #             # 如果没有连接，添加空边
    #             edge_indices.append(torch.empty((2, 0), dtype=torch.long, device=hyper_graph.device))
    #             edge_weights.append(torch.empty((0,), dtype=hyper_graph.dtype, device=hyper_graph.device))
    #             continue

            
    #         # 调整索引（每个批次的节点和超边索引独立）
    #         nodes = agent_indices  # [E]
    #         hyperedges = hyperedge_indices + n_agents  # [E]
    #         edge_index = torch.stack([nodes, hyperedges], dim=0)  # [2, E]
    #         edge_indices.append(edge_index)

    #         # 提取对应的权重
    #         weights = H[agent_indices, hyperedge_indices]  # [E]
    #         edge_weights.append(weights)

    #         # 更新最大边数量
    #         max_edges = max(max_edges, edge_index.size(1))

    #     # 对所有批次进行填充，使得每批次的边数量一致
    #     padded_edge_indices = []
    #     padded_edge_weights = []

    #     for edge_index, edge_weight in zip(edge_indices, edge_weights):
    #         num_edges = edge_index.size(1)
    #         # 填充 edge_index
    #         if num_edges < max_edges:
    #             pad_edge_index = torch.zeros((2, max_edges), dtype=torch.long, device=hyper_graph.device)
    #             pad_edge_index[:, :num_edges] = edge_index
    #         else:
    #             pad_edge_index = edge_index
    #         padded_edge_indices.append(pad_edge_index)

    #         # 填充 edge_weight
    #         if num_edges < max_edges:
    #             pad_edge_weight = torch.zeros((max_edges,), dtype=hyper_graph.dtype, device=hyper_graph.device)
    #             pad_edge_weight[:num_edges] = edge_weight
    #         else:
    #             pad_edge_weight = edge_weight
    #         padded_edge_weights.append(pad_edge_weight)

    #     # 将所有批次的 edge_index 和 edge_weight 转换为张量
    #     all_edge_index = torch.stack(padded_edge_indices, dim=0)  # [batch_size, 2, total_E]
    #     all_edge_weight = torch.stack(padded_edge_weights, dim=0)  # [batch_size, total_E]
        
    #     return all_edge_index, all_edge_weight
  