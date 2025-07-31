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

        layer_gen = lambda: SAGEConv(hidden_channels, hidden_channels)

        super().__init__(in_channels, hidden_channels, num_hidden, param_sharing, layer_gen, edge_orientation,
                         edge_weights)

    def apply_layer(self, layer, x, x_0, edge_index, edge_weights):

        if edge_weights.dim() == 1:
            edge_index = edge_index[:, edge_weights != 0]
            edge_weights = edge_weights[edge_weights != 0]

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


class ResRWGCN(BaseModel):
    def __init__(self, in_channels, hidden_channels, num_hidden, param_sharing, edge_orientation, edge_weights):

        layer_gen = lambda: GCNConv(hidden_channels, hidden_channels, normalize=False, add_self_loops=False)
        super().__init__(in_channels, hidden_channels, num_hidden, param_sharing, layer_gen, edge_orientation, edge_weights)

    def apply_layer(self, layer, x, x_0, edge_index, edge_weights):

        row, col = edge_index
        deg = degree(row, x.size(0), dtype=x.dtype)  # 出度 D
        deg_inv = 1.0 / deg  # D^-1
        deg_inv[deg == 0] = 0

        if edge_weights is None:
            edge_weights = torch.ones(edge_index.size(1), dtype=x.dtype, device=x.device)


        edge_weights = edge_weights * deg_inv[row]

        return relu(layer(x, edge_index, edge_weights))

