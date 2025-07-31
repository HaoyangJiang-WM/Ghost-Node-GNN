import torch
from abc import ABC, abstractmethod
from torch.nn import Module, ModuleList, Linear
from torch.nn.functional import relu, tanh
from torch_geometric.nn import GATConv, GCNConv, GCN2Conv, SAGEConv
from torch_geometric.utils import add_self_loops


def initialize_weights(layer):
    if isinstance(layer, Linear):
        torch.nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')


class EdgeFeatureMLP(Module):
    def __init__(self):
        super().__init__()
        self.mlp = ModuleList([
            Linear(3, 16),
            torch.nn.ReLU(),      # First ReLU activation
            Linear(16, 16),
            torch.nn.ReLU(),
            Linear(16, 1),
            # torch.nn.LeakyReLU(0.01)  # Final LeakyReLU activation
        ])
        self.apply(initialize_weights)

    def forward(self, edge_features):
        for layer in self.mlp:
            edge_features = layer(edge_features)  # Sequentially apply each layer and activation
        return edge_features



class BaseModel(Module, ABC):
    def __init__(self, in_channels, hidden_channels, num_hidden, param_sharing, layerfun, edge_orientation,
                 edge_weights):
        super(BaseModel, self).__init__()
        self.encoder = Linear(in_channels, hidden_channels)
        self.decoder = Linear(hidden_channels, 1)
        self.apply(initialize_weights)

        # Initialize MLP for edge features if they have a dimension of 3
        self.edge_mlp = EdgeFeatureMLP() if edge_weights is not None and edge_weights.size(-1) == 3 else None

        # Initialize layers based on parameter sharing setting
        if param_sharing:
            self.layers = ModuleList([layerfun()] * num_hidden)
        else:
            self.layers = ModuleList([layerfun() for _ in range(num_hidden)])

        self.edge_weights = edge_weights
        self.edge_orientation = edge_orientation
        if self.edge_weights is not None:
            # self.loop_fill_value = 1.0 if (self.edge_weights == 0).all() else "mean"
            self.loop_fill_value = 1.0

    def forward(self, x, edge_index, evo_tracking=False):
        device = x.device  # Get the device from input `x`
        x = x.flatten(1)

        # Handle edge weights with device compatibility
        if self.edge_weights is not None:
            edge_weights = self.edge_weights.to(device)  # Ensure edge weights are on the same device as `x`

            if self.edge_mlp:
                edge_weights = self.edge_mlp(edge_weights.to(device))  # Move edge weights to device for MLP
            num_graphs = edge_index.size(1) // len(edge_weights)
            edge_weights = torch.cat(num_graphs * [edge_weights], dim=0).to(device)
            edge_weights = edge_weights.abs()
        else:
            edge_weights = torch.zeros(edge_index.size(1), device=device)

        # Handle edge orientation
        if self.edge_orientation is not None:
            if self.edge_orientation == "upstream":
                edge_index = edge_index[[1, 0]].to(device)
            elif self.edge_orientation == "bidirectional":
                edge_index = torch.cat([edge_index, edge_index[[1, 0]]], dim=1).to(device)
                edge_weights = torch.cat(2 * [edge_weights], dim=0).to(device)
            elif self.edge_orientation != "downstream":
                raise ValueError("unknown edge direction", self.edge_orientation)

        if self.edge_weights is not None:
            edge_index, edge_weights = add_self_loops(edge_index, edge_weights, fill_value=self.loop_fill_value)

        x_0 = self.encoder(x)
        evolution = [x_0.detach()] if evo_tracking else None
        x = x_0
        for layer in self.layers:
            x = self.apply_layer(layer, x, x_0, edge_index, edge_weights)
            if evo_tracking:
                evolution.append(x.detach())
        x = self.decoder(x)
        if evo_tracking:
            return x, evolution
        return x

    @abstractmethod
    def apply_layer(self, layer, x, x_0, edge_index, edge_weights):
        pass


class MLP(BaseModel):
    def __init__(self, in_channels, hidden_channels, num_hidden, param_sharing):
        layer_gen = lambda: Linear(hidden_channels, hidden_channels)
        super().__init__(in_channels, hidden_channels, num_hidden, param_sharing, layer_gen, None, None)

    def apply_layer(self, layer, x, x_0, edge_index, edge_weights):
        return relu(layer(x))


class GCN(BaseModel):
    def __init__(self, in_channels, hidden_channels, num_hidden, param_sharing, edge_orientation, edge_weights):
        layer_gen = lambda: GCNConv(hidden_channels, hidden_channels, add_self_loops=False)
        super().__init__(in_channels, hidden_channels, num_hidden, param_sharing, layer_gen, edge_orientation,
                         edge_weights)

    def apply_layer(self, layer, x, x_0, edge_index, edge_weights):
        return relu(layer(x, edge_index, edge_weights))


class ResGCN(GCN):
    def __init__(self, in_channels, hidden_channels, num_hidden, param_sharing, edge_orientation, edge_weights):
        super().__init__(in_channels, hidden_channels, num_hidden, param_sharing, edge_orientation, edge_weights)

    def apply_layer(self, layer, x, x_0, edge_index, edge_weights):
        return x + super().apply_layer(layer, x, x_0, edge_index, edge_weights)


class GCNII(BaseModel):
    def __init__(self, in_channels, hidden_channels, num_hidden, param_sharing, edge_orientation, edge_weights):
        layer_gen = lambda: GCN2Conv(hidden_channels, alpha=0.5, add_self_loops=False)
        super().__init__(in_channels, hidden_channels, num_hidden, param_sharing, layer_gen, edge_orientation,
                         edge_weights)

    def apply_layer(self, layer, x, x_0, edge_index, edge_weights):
        return relu(layer(x, x_0, edge_index, edge_weights))


class ResGAT(BaseModel):
    def __init__(self, in_channels, hidden_channels, num_hidden, param_sharing, edge_orientation, edge_weights):
        layer_gen = lambda: GATConv(hidden_channels, hidden_channels, add_self_loops=False)
        super().__init__(in_channels, hidden_channels, num_hidden, param_sharing, layer_gen, edge_orientation,
                         edge_weights)

    def apply_layer(self, layer, x, x_0, edge_index, edge_weights):
        if edge_weights.dim() == 1:
            edge_index = edge_index[:, edge_weights != 0]
        return x + relu(layer(x, edge_index, edge_weights))

class ResGraphSAGE(BaseModel):
    def __init__(self, in_channels, hidden_channels, num_hidden, param_sharing, edge_orientation, edge_weights):
        # 使用 SAGEConv 定义层生成器
        layer_gen = lambda: SAGEConv(hidden_channels, hidden_channels)

        # 初始化 BaseModel
        super().__init__(in_channels, hidden_channels, num_hidden, param_sharing, layer_gen, edge_orientation,
                         edge_weights)

    def apply_layer(self, layer, x, x_0, edge_index, edge_weights):
        # 判断 edge_weights 是否为一维张量，如果是，则过滤掉权重为 0 的边
        if edge_weights.dim() == 1:
            edge_index = edge_index[:, edge_weights != 0]
            edge_weights = edge_weights[edge_weights != 0]
        # 应用 GraphSAGE 卷积层并添加残差连接
        return relu(layer(x, edge_index))

from torch_geometric.nn import MessagePassing
from torch_geometric.nn.inits import glorot, zeros

class CustomMPNN(BaseModel):
    def __init__(self, in_channels, hidden_channels, num_hidden, param_sharing, edge_orientation, edge_weights):
        # Define a custom message-passing layer generator
        layer_gen = lambda: MPNNLayer(hidden_channels)
        super().__init__(in_channels, hidden_channels, num_hidden, param_sharing, layer_gen, edge_orientation, edge_weights)

    def apply_layer(self, layer, x, x_0, edge_index, edge_weights):
        return layer(x, edge_index, edge_weights)


class MPNNLayer(MessagePassing):
    def __init__(self, hidden_channels):
        super().__init__(aggr="mean")  # Use "mean" aggregation
        self.linear_message = Linear(hidden_channels, hidden_channels)
        self.linear_update = Linear(hidden_channels, hidden_channels)
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.linear_message.weight)
        zeros(self.linear_message.bias)
        glorot(self.linear_update.weight)
        zeros(self.linear_update.bias)

    def forward(self, x, edge_index, edge_weight=None):
        # Message-passing mechanism
        return self.propagate(edge_index, x=x, edge_weight=edge_weight)

    def message(self, x_j, edge_weight):
        # Compute messages, considering edge weights if provided
        if edge_weight is not None:
            return self.linear_message(x_j) * edge_weight.view(-1, 1)
        else:
            return self.linear_message(x_j)

    def update(self, aggr_out, x):
        # Update node features with aggregated messages
        return relu(self.linear_update(aggr_out) + x)


from torch_geometric.utils import degree
from torch_geometric.nn import GCNConv
from torch.nn.functional import relu


class ResRWGCN(BaseModel):  # 使用 BaseModel 继承
    def __init__(self, in_channels, hidden_channels, num_hidden, param_sharing, edge_orientation, edge_weights):
        # 定义 GCNConv，并禁用内部归一化和自环
        layer_gen = lambda: GCNConv(hidden_channels, hidden_channels, normalize=False, add_self_loops=False)
        super().__init__(in_channels, hidden_channels, num_hidden, param_sharing, layer_gen, edge_orientation, edge_weights)

    def apply_layer(self, layer, x, x_0, edge_index, edge_weights):
        # --- Step 1: 计算随机游走归一化 ---

        # 计算节点度数 (按源节点计算出度)
        row, col = edge_index
        deg = degree(row, x.size(0), dtype=x.dtype)  # 出度 D
        deg_inv = 1.0 / deg  # D^-1
        deg_inv[deg == 0] = 0  # 避免除零错误

        # --- Step 2: 仅对边权重归一化 ---
        if edge_weights is None:  # 如果没有边权重，默认为1
            edge_weights = torch.ones(edge_index.size(1), dtype=x.dtype, device=x.device)

        # 归一化边权重 (按源节点出度 D^-1 缩放)
        edge_weights = edge_weights * deg_inv[row]

        # --- Step 3: 执行 GCNConv 并返回结果 ---
        return relu(layer(x, edge_index, edge_weights))

#
# import torch
# import torch.nn as nn
# from abc import ABC, abstractmethod
#
# from torch.nn import Module, ModuleList, LSTM
# from torch.nn.functional import mse_loss, relu, leaky_relu, tanh
# from torch_geometric.nn import GATConv, GCNConv, GCN2Conv, Linear
# from torch_geometric.utils import add_self_loops
# from torch_geometric.nn import SAGEConv
# from torch_geometric.utils import degree
#
#
# class BaseModel(Module, ABC):
#     def __init__(self, in_channels, hidden_channels, num_hidden, param_sharing, layerfun, edge_orientation,
#                  edge_weights):
#         super().__init__()
#         self.encoder = Linear(in_channels, hidden_channels, weight_initializer="kaiming_uniform")
#         self.decoder = Linear(hidden_channels, 1, weight_initializer="kaiming_uniform")
#         if param_sharing:
#             self.layers = ModuleList(num_hidden * [layerfun()])
#         else:
#             self.layers = ModuleList([layerfun() for _ in range(num_hidden)])
#
#         # 固定原始边权重，ghost边权重可学习
#         self.edge_weights = edge_weights  # 保留原始的self.edge_weights
#         self.original_weights = edge_weights[:357]
#         self.ghost_weights = nn.Parameter(edge_weights[357:])
#
#         self.edge_orientation = edge_orientation
#         self.loop_fill_value = 1.0 if (edge_weights == 0).all() else "mean"
#
#     def forward(self, x, edge_index, evo_tracking=False):
#         x = x.flatten(1)
#
#         # 1. 获取单图的边权重
#         base_weights = torch.cat([self.original_weights.to(x.device), self.ghost_weights.to(x.device)])
#
#         # 2. 计算每个batch需要的边数量
#         num_edges = edge_index.size(1)
#         edges_per_graph = len(base_weights)
#         batch_size = num_edges // edges_per_graph
#
#         # 3. 复制边权重以匹配batch
#         edge_weights = base_weights.repeat(batch_size)
#
#         # # 打印调试信息
#         # print(f"edge_index shape: {edge_index.shape}")
#         # print(f"edge_weights shape: {edge_weights.shape}")
#
#     # def forward(self, x, edge_index, evo_tracking=False):
#     #     x = x.flatten(1)
#     #
#     #     # 计算每个batch中的原始+ghost边权重
#     #     edge_weights = torch.cat([self.original_weights.to(x.device), self.ghost_weights.to(x.device)])
#     #
#     #     # 获取batch大小
#     #     num_nodes_per_graph = len(self.original_weights)
#     #     batch_size = x.size(0) // num_nodes_per_graph
#     #
#     #     # 复制边权重以匹配batch
#     #     edge_weights = edge_weights.repeat(batch_size)
#     #     edge_weights = edge_weights.abs()
#
#         if self.edge_orientation is not None:
#             if self.edge_orientation == "upstream":
#                 edge_index = edge_index[[1, 0]].to(x.device)
#             elif self.edge_orientation == "bidirectional":
#                 edge_index = torch.cat([edge_index, edge_index[[1, 0]]], dim=1).to(x.device)
#                 edge_weights = torch.cat([edge_weights, edge_weights])
#             elif self.edge_orientation != "downstream":
#                 raise ValueError("unknown edge direction", self.edge_orientation)
#
#         # 为每个batch添加自环
#         num_nodes = x.size(0)
#         edge_index, edge_weights = add_self_loops(edge_index, edge_weights,
#                                                   num_nodes=num_nodes,
#                                                   fill_value=self.loop_fill_value)
#
#         x_0 = self.encoder(x)
#         evolution = [x_0.detach()] if evo_tracking else None
#         x = x_0
#         for layer in self.layers:
#             x = self.apply_layer(layer, x, x_0, edge_index, edge_weights)
#             if evo_tracking:
#                 evolution.append(x.detach())
#         x = self.decoder(x)
#         if evo_tracking:
#             return x, evolution
#         return x
#
# # class BaseModel(Module, ABC):
# #     def __init__(self, in_channels, hidden_channels, num_hidden, param_sharing, layerfun, edge_orientation, edge_weights):
# #         super().__init__()
# #         self.encoder = Linear(in_channels, hidden_channels, weight_initializer="kaiming_uniform")
# #         self.decoder = Linear(hidden_channels, 1, weight_initializer="kaiming_uniform")
# #         if param_sharing:
# #             self.layers = ModuleList(num_hidden * [layerfun()])
# #         else:
# #             self.layers = ModuleList([layerfun() for _ in range(num_hidden)])
# #         self.edge_weights = edge_weights
# #         self.edge_orientation = edge_orientation
# #         if self.edge_weights is not None:
# #             self.loop_fill_value = 1.0 if (self.edge_weights == 0).all() else "mean"
# #
# #     def forward(self, x, edge_index, evo_tracking=False):
# #         # print('before flatten',x.shape)
# #         x = x.flatten(1)
# #         # print('after flatten',x.shape)
# #         if self.edge_weights is not None:
# #             num_graphs = edge_index.size(1) // len(self.edge_weights)
# #             edge_weights = torch.cat(num_graphs * [self.edge_weights], dim=0).to(x.device)
# #             edge_weights = edge_weights.abs()  # relevant when edge weights are learned
# #         else:
# #             edge_weights = torch.zeros(edge_index.size(1)).to(x.device)
# #         # print('edge_weights1',edge_weights.shape)
# #         if self.edge_orientation is not None:
# #             if self.edge_orientation == "upstream":
# #                 edge_index = edge_index[[1, 0]].to(x.device)
# #             elif self.edge_orientation == "bidirectional":
# #                 edge_index = torch.cat([edge_index, edge_index[[1, 0]]], dim=1).to(x.device)
# #                 edge_weights = torch.cat(2 * [edge_weights], dim=0).to(x.device)
# #             elif self.edge_orientation != "downstream":
# #                 raise ValueError("unknown edge direction", self.edge_orientation)
# #         if self.edge_weights is not None:
# #             # print(edge_index.shape, edge_weights.shape)
# #             edge_index, edge_weights = add_self_loops(edge_index, edge_weights, fill_value=self.loop_fill_value)
# #
# #         # print('before encoder',x.shape)
# #         x_0 = self.encoder(x)
# #         # print('after encoder',x_0.shape)
# #
# #         evolution = [x_0.detach()] if evo_tracking else None
# #         # print('edge_weights2',edge_weights.shape)
# #         x = x_0
# #         for layer in self.layers:
# #             x = self.apply_layer(layer, x, x_0, edge_index, edge_weights)
# #             if evo_tracking:
# #                 evolution.append(x.detach())
# #         # print('before decoder',x.shape)
# #         x = self.decoder(x)
# #         # print('after decoder',x.shape)
# #         if evo_tracking:
# #             return x, evolution
# #         return x
#
#     @abstractmethod
#     def apply_layer(self, layer, x, x_0, edge_index, edge_weights):
#         pass
#
#
# class MLP(BaseModel):
#     def __init__(self, in_channels, hidden_channels, num_hidden, param_sharing):
#         layer_gen = lambda: Linear(hidden_channels, hidden_channels, weight_initializer="kaiming_uniform")
#         super().__init__(in_channels, hidden_channels, num_hidden, param_sharing, layer_gen, None, None)
#
#     def apply_layer(self, layer, x, x_0, edge_index, edge_weights):
#         return relu(layer(x))
#         # return x + relu(layer(x))
#
#
# class GCN(BaseModel):
#     def __init__(self, in_channels, hidden_channels, num_hidden, param_sharing, edge_orientation, edge_weights):
#         layer_gen = lambda: GCNConv(hidden_channels, hidden_channels, add_self_loops=False)
#         super().__init__(in_channels, hidden_channels, num_hidden, param_sharing, layer_gen, edge_orientation, edge_weights)
#
#     def apply_layer(self, layer, x, x_0, edge_index, edge_weights):
#         return relu(layer(x, edge_index, edge_weights))
#
#
# class ResGCN(GCN):
#     def __init__(self, in_channels, hidden_channels, num_hidden, param_sharing, edge_orientation, edge_weights):
#         super().__init__(in_channels, hidden_channels, num_hidden, param_sharing, edge_orientation, edge_weights)
#
#     def apply_layer(self, layer, x, x_0, edge_index, edge_weights):
#         return x + super().apply_layer(layer, x, x_0, edge_index, edge_weights)
#
#
# class GCNII(BaseModel):
#     def __init__(self, in_channels, hidden_channels, num_hidden, param_sharing, edge_orientation, edge_weights):
#         layer_gen = lambda: GCN2Conv(hidden_channels, alpha=0.5, add_self_loops=False)
#         super().__init__(in_channels, hidden_channels, num_hidden, param_sharing, layer_gen, edge_orientation, edge_weights)
#
#     def apply_layer(self, layer, x, x_0, edge_index, edge_weights):
#         return relu(layer(x, x_0, edge_index, edge_weights))
#
#
# class ResGAT(BaseModel):
#     def __init__(self, in_channels, hidden_channels, num_hidden, param_sharing, edge_orientation, edge_weights):
#         layer_gen = lambda: GATConv(hidden_channels, hidden_channels, add_self_loops=False)
#         super().__init__(in_channels, hidden_channels, num_hidden, param_sharing, layer_gen, edge_orientation, edge_weights)
#
#     def apply_layer(self, layer, x, x_0, edge_index, edge_weights):
#         if edge_weights.dim() == 1:
#             edge_index = edge_index[:, edge_weights != 0]
#         # print('gat')
#         # return x + tanh(layer(x, edge_index, edge_weights))
#         # return x + relu(layer(x, edge_index, edge_weights))
#         return relu(layer(x, edge_index, edge_weights))
#
#
#
# class ResGraphSAGE(BaseModel):
#     def __init__(self, in_channels, hidden_channels, num_hidden, param_sharing, edge_orientation, edge_weights):
#         # 使用 SAGEConv 定义层生成器
#         layer_gen = lambda: SAGEConv(hidden_channels, hidden_channels)
#
#         # 初始化 BaseModel
#         super().__init__(in_channels, hidden_channels, num_hidden, param_sharing, layer_gen, edge_orientation,
#                          edge_weights)
#
#     def apply_layer(self, layer, x, x_0, edge_index, edge_weights):
#         # 判断 edge_weights 是否为一维张量，如果是，则过滤掉权重为 0 的边
#         if edge_weights.dim() == 1:
#             edge_index = edge_index[:, edge_weights != 0]
#             edge_weights = edge_weights[edge_weights != 0]
#         # 应用 GraphSAGE 卷积层并添加残差连接
#         return relu(layer(x, edge_index))
#
# from torch_geometric.nn import MessagePassing
# from torch_geometric.nn.inits import glorot, zeros
#
# class MPNNLayer(MessagePassing):
#     def __init__(self, hidden_channels):
#         super().__init__(aggr="mean")
#                 # 两层MLP处理边特征
#         self.edge_transform = ModuleList([
#             Linear(3, 16),  # 第一层先升维到8
#             torch.nn.ReLU(),
#             Linear(16, 12),  # 第二层到最终的12维
#             torch.nn.ReLU()
#         ])
#         self.linear_message = Linear(hidden_channels + 12, hidden_channels)  # 处理节点特征和扩展后的边特征
#         self.linear_update = Linear(hidden_channels, hidden_channels)
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         for layer in self.edge_transform:
#             if isinstance(layer, Linear):
#                 glorot(layer.weight)
#                 zeros(layer.bias)
#         glorot(self.linear_message.weight)
#         zeros(self.linear_message.bias)
#         glorot(self.linear_update.weight)
#         zeros(self.linear_update.bias)
#
#     def forward(self, x, edge_index, edge_attr):
#         return self.propagate(edge_index, x=x, edge_attr=edge_attr)
#
#     def message(self, x_j, edge_attr):
#         # 先将边特征转换到更高维度
#         # 通过两层MLP处理边特征
#         edge_features = edge_attr
#         for layer in self.edge_transform:
#             edge_features = layer(edge_features)  # 依次经过Linear->ReLU->Linear->ReLU
#         # 将节点特征和扩展后的边特征拼接
#         combined = torch.cat([x_j, edge_features], dim=-1)  # [num_edges, hidden_channels + 12]
#         # 通过线性层处理组合特征
#         return self.linear_message(combined)
#
#     def update(self, aggr_out, x):
#         return relu(self.linear_update(aggr_out) + x)
#
#
# class CustomMPNN(BaseModel):
#     def __init__(self, in_channels, hidden_channels, num_hidden, param_sharing, edge_orientation, edge_weights):
#         layer_gen = lambda: MPNNLayer(hidden_channels)
#         super().__init__(in_channels, hidden_channels, num_hidden, param_sharing, layer_gen, edge_orientation, edge_weights)
#
#     def apply_layer(self, layer, x, x_0, edge_index, edge_weights):
#         # edge_weights在这里就是原始的3维边特征
#         return layer(x, edge_index, edge_weights)
#
# # class CustomMPNN(BaseModel):
# #     def __init__(self, in_channels, hidden_channels, num_hidden, param_sharing, edge_orientation, edge_weights):
# #         # Define a custom message-passing layer generator
# #         layer_gen = lambda: MPNNLayer(hidden_channels)
# #         super().__init__(in_channels, hidden_channels, num_hidden, param_sharing, layer_gen, edge_orientation, edge_weights)
# #
# #     def apply_layer(self, layer, x, x_0, edge_index, edge_weights):
# #         return layer(x, edge_index, edge_weights)
# #
# #
# # class MPNNLayer(MessagePassing):
# #     def __init__(self, hidden_channels):
# #         super().__init__(aggr="mean")  # Use "mean" aggregation
# #         self.linear_message = Linear(hidden_channels, hidden_channels, weight_initializer="kaiming_uniform")
# #         self.linear_update = Linear(hidden_channels, hidden_channels, weight_initializer="kaiming_uniform")
# #         self.reset_parameters()
# #
# #     def reset_parameters(self):
# #         glorot(self.linear_message.weight)
# #         zeros(self.linear_message.bias)
# #         glorot(self.linear_update.weight)
# #         zeros(self.linear_update.bias)
# #
# #     def forward(self, x, edge_index, edge_weight=None):
# #         # Message-passing mechanism
# #         return self.propagate(edge_index, x=x, edge_weight=edge_weight)
# #
# #     def message(self, x_j, edge_weight):
# #         # Compute messages, considering edge weights if provided
# #         if edge_weight is not None:
# #             return self.linear_message(x_j) * edge_weight.view(-1, 1)
# #         else:
# #             return self.linear_message(x_j)
# #
# #     def update(self, aggr_out, x):
# #         # Update node features with aggregated messages
# #         return relu(self.linear_update(aggr_out) + x)
#
# # class MPNNLayer(MessagePassing):
# #     def __init__(self, hidden_channels):
# #         super().__init__(aggr="mean")
# #         self.linear_message = Linear(hidden_channels, hidden_channels)
# #         self.linear = Linear(2 * hidden_channels, hidden_channels)  # 处理节点自身特征和聚合特征的拼接
# #         self.reset_parameters()
# #
# #     def reset_parameters(self):
# #         glorot(self.linear_message.weight)
# #         zeros(self.linear_message.bias)
# #         glorot(self.linear.weight)
# #         zeros(self.linear.bias)
# #
# #     def forward(self, x, edge_index):
# #         return self.propagate(edge_index, x=x)
# #
# #     def message(self, x_j):
# #         return self.linear_message(x_j)
# #
# #     def update(self, aggr_out, x):
# #         # 直接拼接节点自身特征和聚合后的邻居特征
# #         return relu(self.linear(torch.cat([x, aggr_out], dim=-1)))
# #
# #
# # class CustomMPNN(BaseModel):
# #     def __init__(self, in_channels, hidden_channels, num_hidden, param_sharing, edge_orientation, edge_weights):
# #         layer_gen = lambda: MPNNLayer(hidden_channels)
# #         super().__init__(in_channels, hidden_channels, num_hidden, param_sharing,
# #                         layer_gen, edge_orientation, edge_weights)
# #
# #     def apply_layer(self, layer, x, x_0, edge_index, edge_weights):
# #         return layer(x, edge_index)
#
#
# class WENOGNNLayer(MessagePassing):
#     def __init__(self, in_features, out_features, hidden_features):
#         super().__init__(node_dim=-2, aggr='mean')
#         self.in_features = in_features
#         self.out_features = out_features
#         self.hidden_features = hidden_features
#
#         # WENO coefficients
#         self.register_buffer('recon_stencils', torch.tensor([
#             [1 / 3, -7 / 6, 11 / 6],  # stencil 1
#             [-1 / 6, 5 / 6, 1 / 3],  # stencil 2
#             [1 / 3, 5 / 6, -1 / 6]  # stencil 3
#         ]))
#         self.register_buffer('linear_weights', torch.tensor([0.1, 0.6, 0.3]))
#         self.register_buffer('smooth_coefs', torch.tensor([13 / 12, 1 / 4]))
#
#         self.message_net = nn.Sequential(
#             nn.Linear(2 * in_features + 1, hidden_features),  # +1 for spatial difference
#             nn.ReLU(),
#             nn.Linear(hidden_features, hidden_features)
#         )
#
#         self.update_net = nn.Sequential(
#             nn.Linear(in_features + hidden_features, hidden_features),
#             nn.ReLU(),
#             nn.Linear(hidden_features, out_features)
#         )
#
#     def forward(self, x, edge_index, edge_weight=None):
#         x = self.propagate(edge_index, x=x, edge_weight=edge_weight)
#         return x
#
#     def update(self, aggr_out, x):
#         return self.update_net(torch.cat([x, aggr_out], dim=-1))
#
#     def message(self, x_i, x_j, edge_weight=None):
#         spatial_diff = torch.norm(x_i - x_j, dim=-1, keepdim=True)  # [45760, 1]
#         diff = x_i - x_j  # [45760, 128]
#         batch_size = diff.size(0)
#
#         # 计算每个边的3个WENO权重
#         mean_diff = diff.mean(dim=1)  # [45760]
#         beta = torch.zeros(batch_size, 3, device=x_i.device)
#
#         for k in range(3):
#             stencil_weight = self.recon_stencils[k].to(x_i.device)  # [3]
#             weighted_mean = mean_diff.unsqueeze(-1) * stencil_weight  # [45760, 3]
#             beta[:, k] = (self.smooth_coefs[0] * weighted_mean.pow(2).mean(1) +
#                           self.smooth_coefs[1] * weighted_mean.mean(1).pow(2))
#
#         # 权重归一化
#         w = self.get_nonlinear_weights(beta)  # [45760, 3]
#
#         # 生成消息
#         message_input = torch.cat([x_i, x_j, spatial_diff], dim=-1)  # [45760, 257]
#         message = self.message_net(message_input)  # [45760, hidden]
#
#         # 加权
#         message = message.unsqueeze(1).expand(-1, 3, -1)  # [45760, 3, hidden]
#         w = w.unsqueeze(-1)  # [45760, 3, 1]
#         weighted_message = (message * w).sum(dim=1)  # [45760, hidden]
#
#         return weighted_message
#
#     def calculate_smoothness(self, x_i, x_j):
#         diff = x_i - x_j  # [45760, hidden_features]
#         batch_size = diff.size(0)
#         beta = torch.zeros(batch_size, 3, device=x_i.device)
#
#         for k in range(3):
#             weighted_diff = diff * self.recon_stencils[k].view(1, 1)
#             beta[:, k] = (self.smooth_coefs[0] * weighted_diff.pow(2).mean(1) +
#                           self.smooth_coefs[1] * weighted_diff.mean(1).pow(2))
#
#         return beta  # [45760, 3]
#
#     def get_nonlinear_weights(self, beta, epsilon=1e-6):
#         alpha = self.linear_weights.view(1, -1) / (epsilon + beta).pow(2)  # [45760, 3]
#         return alpha / alpha.sum(dim=1, keepdim=True)  # [45760, 3]
#
#
# class WENOPDE_Solver(BaseModel):
#     def __init__(self, in_channels, hidden_channels, num_hidden, param_sharing=True,
#                  edge_orientation="bidirectional", edge_weights=None):
#         layerfun = lambda: WENOGNNLayer(
#             in_features=hidden_channels,
#             out_features=hidden_channels,
#             hidden_features=hidden_channels
#         )
#         super().__init__(in_channels, hidden_channels, num_hidden, param_sharing,
#                          layerfun, edge_orientation, edge_weights)
#
#     def apply_layer(self, layer, x, x_0, edge_index, edge_weights):
#         return layer(x, edge_index, edge_weights)
#
#
#
# # from torch_geometric.nn import MessagePassing
# # from torch_geometric.nn.inits import glorot, zeros
# #
# # class CustomMPNN(BaseModel):
# #     def __init__(self, in_channels, hidden_channels, num_hidden, param_sharing, edge_orientation, edge_weights):
# #         # Define a custom message-passing layer generator
# #         layer_gen = lambda: MPNNLayer(hidden_channels)
# #         super().__init__(in_channels, hidden_channels, num_hidden, param_sharing, layer_gen, edge_orientation, edge_weights)
# #
# #     def apply_layer(self, layer, x, x_0, edge_index, edge_weights):
# #         return layer(x, edge_index, edge_weights)
# #
# #
# # class MPNNLayer(MessagePassing):
# #     def __init__(self, hidden_channels):
# #         super().__init__(aggr="mean")  # Use "mean" aggregation
# #         self.linear_message = Linear(hidden_channels, hidden_channels, weight_initializer="kaiming_uniform")
# #         self.linear_update = Linear(hidden_channels, hidden_channels, weight_initializer="kaiming_uniform")
# #         self.reset_parameters()
# #
# #     def reset_parameters(self):
# #         glorot(self.linear_message.weight)
# #         zeros(self.linear_message.bias)
# #         glorot(self.linear_update.weight)
# #         zeros(self.linear_update.bias)
# #
# #     def forward(self, x, edge_index, edge_weight=None):
# #         # Message-passing mechanism
# #         return self.propagate(edge_index, x=x, edge_weight=edge_weight)
# #
# #     def message(self, x_j, edge_weight):
# #         # Compute messages, considering edge weights if provided
# #         if edge_weight is not None:
# #             return self.linear_message(x_j) * edge_weight.view(-1, 1)
# #         else:
# #             return self.linear_message(x_j)
# #
# #     def update(self, aggr_out, x):
# #         # Update node features with aggregated messages
# #         return relu(self.linear_update(aggr_out) + x)
# #         # return relu(self.linear_update(aggr_out) + x)
#
# from torch_geometric.utils import degree
# from torch_geometric.nn import GCNConv
# from torch.nn.functional import relu
#
# class ResRWGCN(BaseModel):  # 使用 BaseModel 继承
#     def __init__(self, in_channels, hidden_channels, num_hidden, param_sharing, edge_orientation, edge_weights):
#         # 定义 GCNConv，并禁用内部归一化和自环
#         layer_gen = lambda: GCNConv(hidden_channels, hidden_channels, normalize=False, add_self_loops=False)
#         super().__init__(in_channels, hidden_channels, num_hidden, param_sharing, layer_gen, edge_orientation, edge_weights)
#
#     def apply_layer(self, layer, x, x_0, edge_index, edge_weights):
#         # --- Step 1: 计算随机游走归一化 ---
#
#         # 计算节点度数 (按源节点计算出度)
#         row, col = edge_index
#         deg = degree(row, x.size(0), dtype=x.dtype)  # 出度 D
#         deg_inv = 1.0 / deg  # D^-1
#         deg_inv[deg == 0] = 0  # 避免除零错误
#
#         # --- Step 2: 仅对边权重归一化 ---
#         if edge_weights is None:  # 如果没有边权重，默认为1
#             edge_weights = torch.ones(edge_index.size(1), dtype=x.dtype, device=x.device)
#
#         # 归一化边权重 (按源节点出度 D^-1 缩放)
#         edge_weights = edge_weights * deg_inv[row]
#
#         # --- Step 3: 执行 GCNConv 并返回结果 ---
#         return relu(layer(x, edge_index, edge_weights))
#
#
# import torch
# from torch_geometric.utils import get_laplacian, to_dense_adj
# from torch.nn.functional import softplus
#
#
# class FastWaveletLayer(Module):
#     def __init__(self, in_channels, out_channels, K=4):
#         super().__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.K = K
#
#         # 学习参数
#         self.weight = torch.nn.Parameter(torch.Tensor(in_channels, out_channels))
#         self.wavelet_weight = torch.nn.Parameter(torch.Tensor(K + 1))  # 每个切比雪夫多项式的系数
#         self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
#
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         torch.nn.init.kaiming_uniform_(self.weight)
#         torch.nn.init.uniform_(self.wavelet_weight, -0.5, 0.5)
#         torch.nn.init.zeros_(self.bias)
#
#     def forward(self, x, edge_index, edge_weight=None):
#         # 线性变换
#         x = torch.mm(x, self.weight)
#
#         # 获取归一化拉普拉斯矩阵
#         if edge_weight is not None:
#             edge_index, norm = get_laplacian(edge_index, edge_weight, normalization='sym')
#         else:
#             edge_index, norm = get_laplacian(edge_index, normalization='sym')
#
#         # 初始化切比雪夫多项式
#         Tk_0 = x  # 零阶项
#         if self.K < 1:
#             return self.wavelet_weight[0] * Tk_0 + self.bias
#
#         Tk_1 = self.propagate(edge_index, x=x, norm=norm)  # 一阶项
#         out = self.wavelet_weight[0] * Tk_0 + self.wavelet_weight[1] * Tk_1
#
#         # 递归计算高阶切比雪夫多项式
#         for k in range(2, self.K + 1):
#             Tk_2 = 2 * self.propagate(edge_index, x=Tk_1, norm=norm) - Tk_0  # T_k = 2xT_{k-1} - T_{k-2}
#             out = out + self.wavelet_weight[k] * Tk_2
#             Tk_0, Tk_1 = Tk_1, Tk_2  # 更新项
#
#         return out + self.bias
#
#     def propagate(self, edge_index, x, norm):
#         row, col = edge_index
#         out = torch.zeros_like(x)
#         out.index_add_(0, row, x[col] * norm.view(-1, 1))
#         return out
#
#
# class ResGraphWavelet(BaseModel):
#     def __init__(self, in_channels, hidden_channels, num_hidden, param_sharing, edge_orientation, edge_weights, K=3):
#         layer_gen = lambda: FastWaveletLayer(hidden_channels, hidden_channels, K)
#         super().__init__(in_channels, hidden_channels, num_hidden, param_sharing, layer_gen, edge_orientation,
#                          edge_weights)
#
#     def apply_layer(self, layer, x, x_0, edge_index, edge_weights):
#         if edge_weights is not None and edge_weights.dim() == 1:
#             mask = edge_weights != 0
#             edge_index = edge_index[:, mask]
#             edge_weights = edge_weights[mask]
#
#         wavelet_out = layer(x, edge_index, edge_weights)
#         return x + relu(wavelet_out)  # 残差连接
