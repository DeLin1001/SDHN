import torch
# from torch._C import INSERT_FOLD_PREPACK_OPS, Node
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np

class HGCN(nn.Module):
    def __init__(self, n_edges, in_feature, out_feature, n_agents):
        super(HGCN, self).__init__()
        print(n_edges)
        self.W_line = nn.Parameter(torch.ones(n_edges).cuda())
        self.W = None

    def forward(self, node_features, hyper_graph):
        self.W = torch.diag_embed(self.W_line)
        B_inv = torch.sum(hyper_graph.detach(), dim=-2)
        B_inv = torch.diag_embed(B_inv)
        softmax_w = torch.abs(self.W).detach()
        D_inv = torch.matmul(hyper_graph.detach(), softmax_w).sum(dim=-1)
        D_inv = torch.diag_embed(D_inv)
        D_inv = D_inv **(-0.5)
        B_inv = B_inv **(-1)
        D_inv[D_inv == float('inf')] = 0
        D_inv[D_inv == float('nan')] = 0
        B_inv[B_inv == float('inf')] = 0
        B_inv[B_inv == float('nan')] = 0
        A = torch.bmm(D_inv, hyper_graph)
        A = torch.matmul(A, torch.abs(self.W))
        A = torch.bmm(A, B_inv)
        A = torch.bmm(A, hyper_graph.transpose(-2, -1))
        A = torch.bmm(A, D_inv)
        X = torch.bmm(A, node_features)
        return X


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
        r"""Runs the forward pass of the module.

        Args:
            x (torch.Tensor): Node feature matrix
                :math:`\mathbf{X} \in \mathbb{R}^{N \times F}`.
            hyperedge_index (torch.Tensor): The hyperedge indices, *i.e.*
                the sparse incidence matrix
                :math:`\mathbf{H} \in {\{ 0, 1 \}}^{N \times M}` mapping from
                nodes to edges.
            hyperedge_weight (torch.Tensor, optional): Hyperedge weights
                :math:`\mathbf{W} \in \mathbb{R}^M`. (default: :obj:`None`)
            hyperedge_attr (torch.Tensor, optional): Hyperedge feature matrix
                in :math:`\mathbb{R}^{M \times F}`.
                These features only need to get passed in case
                :obj:`use_attention=True`. (default: :obj:`None`)
            num_edges (int, optional) : The number of edges :math:`M`.
                (default: :obj:`None`)
        """
        num_nodes = x.size(0)

        if num_edges is None:
            num_edges = 0
            if hyperedge_index.numel() > 0:
                num_edges = int(hyperedge_index[1].max()) + 1

        if hyperedge_weight is None:
            hyperedge_weight = x.new_ones(num_edges)

        x = self.lin(x)

        alpha = None
        if self.use_attention:
            assert hyperedge_attr is not None
            x = x.view(-1, self.heads, self.out_channels)
            hyperedge_attr = self.lin(hyperedge_attr)
            hyperedge_attr = hyperedge_attr.view(-1, self.heads,
                                                 self.out_channels)
            x_i = x[hyperedge_index[0]]
            x_j = hyperedge_attr[hyperedge_index[1]]
            alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
            alpha = F.leaky_relu(alpha, self.negative_slope)
            if self.attention_mode == 'node':
                alpha = softmax(alpha, hyperedge_index[1], num_nodes=num_edges)
            else:
                alpha = softmax(alpha, hyperedge_index[0], num_nodes=num_nodes)
            alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        D = scatter(hyperedge_weight[hyperedge_index[1]], hyperedge_index[0],
                    dim=0, dim_size=num_nodes, reduce='sum')
        D = 1.0 / D
        D[D == float("inf")] = 0

        B = scatter(x.new_ones(hyperedge_index.size(1)), hyperedge_index[1],
                    dim=0, dim_size=num_edges, reduce='sum')
        B = 1.0 / B
        B[B == float("inf")] = 0

        out = self.propagate(hyperedge_index, x=x, norm=B, alpha=alpha,
                             size=(num_nodes, num_edges))
        out = self.propagate(hyperedge_index.flip([0]), x=out, norm=D,
                             alpha=alpha, size=(num_edges, num_nodes))

        if self.concat is True:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias

        return out

    def message(self, x_j: Tensor, norm_i: Tensor, alpha: Tensor) -> Tensor:
        H, F = self.heads, self.out_channels

        out = norm_i.view(-1, 1, 1) * x_j.view(-1, H, F)

        if alpha is not None:
            out = alpha.view(-1, self.heads, 1) * out

        return out



class Encoder(nn.Module):
    def __init__(self, aggregator, feature_dim):
        super(Encoder, self).__init__()
        self.aggregator = aggregator
        self.feature_dim = feature_dim

    def forward(self, node_features, hyper_graph):
        output = self.aggregator.forward(node_features, hyper_graph)
        return output

class HGCNMixer(nn.Module):
    def __init__(self, args):
        super(HGCNMixer, self).__init__()
        self.args = args
        self.add_self = args.add_self
        self.hyper_hidden_dim = args.hyper_hidden_dim
        self.head_num = 1
        self.hyper_edge_num = args.hyper_edge_num
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))
        self.indiv_u_dim = int(np.prod(args.observation_shape))
        self.use_one_hot = False
        self.n_hyper_edge = self.hyper_edge_num
        if self.use_one_hot:
            self.n_hyper_edge += self.n_agents
        self.use_elu = True
        self.hyper_edge_net = nn.Sequential(
            nn.Linear(in_features=self.indiv_u_dim, out_features=self.hyper_edge_num),
            nn.ReLU(),
        )
        self.hidden_dim = 64
        self.encoder_1 = nn.ModuleList([Encoder(HGCN(self.n_hyper_edge, 1, self.hidden_dim, self.n_agents), self.indiv_u_dim) for _ in range(self.head_num)])
        self.encoder_2 = nn.ModuleList([Encoder(HGCN(self.n_hyper_edge, 1, self.hidden_dim, self.n_agents), self.indiv_u_dim) for _ in range(self.head_num)])
        self.hyper_weight_layer_1 = nn.Sequential(
            nn.Linear(self.state_dim, self.hyper_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hyper_hidden_dim, self.n_agents)
        )
        self.hyper_const_layer_1 = nn.Sequential(
            nn.Linear(self.state_dim, self.hyper_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hyper_hidden_dim, self.n_agents)
        )

        self.hyper_weight_layer = nn.Sequential(
            nn.Linear(self.state_dim, self.hyper_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hyper_hidden_dim, self.n_agents)
        )
        self.hyper_const_layer = nn.Sequential(
            nn.Linear(self.state_dim, self.hyper_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hyper_hidden_dim, 1)
        )

    def build_hyper_net(self, indiv_us):
        """
        生成动态超图，用于后续超图卷积操作。
        
        :param indiv_us: 每个智能体的输入特征，形状为 [batch_size, n_agents, indiv_feature_dim]
        :return: 动态生成的超图，形状为 [batch_size, n_agents, hyper_edge_num]
        """
        out = self.hyper_edge_net(indiv_us) # [batch_size, n_agents, hyper_edge_num]
        mean = out.clone().detach().mean()
        out = out.reshape([out.shape[0], self.n_agents, -1])
        # if self.use_one_hot:
        #     one_hot = torch.eye(self.n_agents)
        #     one_hot = one_hot.flatten().cuda()
        #     mean = out.clone().detach().mean()
        #     one_hot = one_hot * mean
        #     one_hot = one_hot.repeat(indiv_us.shape[0], 1).reshape([indiv_us.shape[0],self.n_agents, -1]).cuda()
        #     out = torch.cat([out, one_hot], dim=-1)
        return out.reshape([out.shape[0], out.shape[1], -1])

    def forward(self, agent_qs, states, indiv_us):
        bs = agent_qs.size(0)
        sl = agent_qs.size(1)
        agent_qs = agent_qs.view(-1, agent_qs.size(-1))
        indiv_us = indiv_us.reshape(-1, indiv_us.size(-2), indiv_us.size(-1))
        hyper_graph = self.build_hyper_net(indiv_us)
        states = states.reshape(-1, states.size(-1))
        hyper_graph = hyper_graph.reshape(-1, hyper_graph.size(-2), hyper_graph.size(-1))
        node_features = agent_qs.unsqueeze(dim=-1)
        # qs_tot = node_features.squeeze(dim=-1)
        qs_tot = self.encoder_2[0](self.encoder_1[0].forward(node_features, hyper_graph), hyper_graph).squeeze(dim=-1)
        hyper_weight_1 = torch.abs(self.hyper_weight_layer_1(states))
        hyper_const_1 = self.hyper_const_layer_1(states)
        q_tot = (qs_tot * hyper_weight_1) + hyper_const_1
        if self.use_elu:
            q_tot = F.elu(q_tot)
        hyper_weight = torch.abs(self.hyper_weight_layer(states))
        hyper_const = self.hyper_const_layer(states).squeeze(dim=-1)
        q_tot = (q_tot*hyper_weight).sum(dim=-1) + hyper_const.squeeze(dim=-1).squeeze(dim=-1)
        return q_tot.view(bs, sl, 1)




class HGCN_EMBEDDING(nn.Module):
    def __init__(self, in_feature_dim, embedding_dim, n_hyper_edges, n_agents, hyper_graph=None):
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
        
        
        # 初始化超图，如果未指定，则随机生成一个可学习的超图
        if hyper_graph is None:
            # 使用正态分布随机初始化超图权重
            hyper_graph = torch.randn(n_agents, n_hyper_edges)
        
        # 将超图设置为可训练的参数
        self.hyper_graph = nn.Parameter(hyper_graph)

        # 使用 HGCN 作为嵌入层

        # self.hgcn_layer = HGCN(n_hyper_edges, in_feature_dim, embedding_dim, n_agents)

        # 将 HGCN 替换为 HypergraphConv
        self.hgcn_layer = HypergraphConv(
            in_channels=in_feature_dim,          # 输入特征维度
            out_channels=embedding_dim,         # 输出嵌入维度
            use_attention=False,                 # 根据需求设置是否使用注意力机制
            attention_mode='node',               # 注意力模式
            heads=1,                             # 多头注意力的头数
            concat=True,                         # 是否连接多头的输出
            dropout=0.0,                         # dropout 概率
            bias=True                            # 是否使用偏置
        )  
        
        # 输出维度调整，确保输出为指定的 embedding_dim
        self.output_layer = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU()
        )

    def forward(self, input):
        """
        前向传播方法，使用 HypergraphConv 将输入转化为嵌入。
        :param input: 智能体输入，形状为 [batch_size, n_agents, in_feature_dim]
        :return: 嵌入，形状为 [batch_size * n_agents, embedding_dim]
        """
        batch_size = input.size(0)  # 获取批量大小

        # 重复超图以匹配批量大小
        hyper_graph = self.hyper_graph.unsqueeze(0).repeat(batch_size, 1, 1)  # [batch_size, n_agents, n_hyper_edges]

        # 构建 edge_index 和 edge_weight
        edge_index, edge_weight = self.build_edge_index(hyper_graph)  # [2, total_E], [total_E]

        # 重塑 node_features 为 [batch_size * n_agents, in_feature_dim]
        node_features = input.view(-1, self.in_feature_dim).cuda()  # 修改部分：确保 node_features 在 GPU 上

        # 通过 HypergraphConv 层
        x = self.hgcn_layer(node_features, edge_index, hyperedge_weight=edge_weight)  # [batch_size * n_agents, embedding_dim]
        r'''PyTorch中，nn.Module 的 forward 方法通常不应被直接调用。
        相反，应该通过调用模块本身来触发 forward 方法。这是因为调用模块本身
        （如 self.hgcn_layer(...)）不仅会调用 forward 方法，还会处理hook、注册的模块等内部机制，
        而直接调用 forward 方法会绕过这些机制，可能导致一些副作用或意外行为'''
        # 通过输出层
        embeddings = self.output_layer(x)  # [batch_size * n_agents, embedding_dim]

        return embeddings  # 返回嵌入

    def build_edge_index(self, hyper_graph):
        """
        构建 edge_index 和 edge_weight 对应于 hyper_graph（batch_size, n_agents, n_hyper_edges）。
        这里我们将超图转换为 bipartite graph 的 edge_index，其中节点索引和超边的索引分开处理。

        :param hyper_graph: [batch_size, n_agents, n_hyper_edges]
        :return: edge_index [2, total_E], edge_weight [total_E]
        """
        batch_size, n_agents, n_hyper_edges = hyper_graph.size()
        edge_indices = []
        edge_weights = []
        # Offset for node and hyperedge indices per sample in batch
        node_offset = 0
        hyperedge_offset = n_agents
        for b in range(batch_size):
            H = hyper_graph[b]  # [n_agents, n_hyper_edges]
            # 找出 H 中的非零元素
            agent_indices, hyperedge_indices = H.nonzero(as_tuple=False).t()  # [E]
            if agent_indices.numel() == 0:
                # 没有连接，跳过
                continue
            # 调整索引以避免不同样本之间的重叠
            nodes = agent_indices + node_offset  # [E]
            hyperedges = hyperedge_indices + hyperedge_offset  # [E]
            edge_index = torch.stack([nodes, hyperedges], dim=0)  # [2, E]
            edge_indices.append(edge_index)
            weights = H[agent_indices, hyperedge_indices]  # [E]
            edge_weights.append(weights)
            node_offset += n_agents
            hyperedge_offset += n_hyper_edges
        if len(edge_indices) == 0:
            # 如果所有样本都没有连接，返回空
            return torch.empty((2, 0), dtype=torch.long).to(hyper_graph.device), torch.empty((0,)).to(hyper_graph.device)
        # 拼接所有样本的 edge_index 和 edge_weight
        all_edge_index = torch.cat(edge_indices, dim=1)  # [2, total_E]
        all_edge_weight = torch.cat(edge_weights, dim=0)  # [total_E]
        return all_edge_index, all_edge_weight