import torch
import torch.nn as nn
from abc import ABC, abstractmethod

from torch.nn import Module, ModuleList, LSTM
from torch.nn.functional import mse_loss, relu, leaky_relu, tanh
from torch_geometric.nn import GATConv, GCNConv, GCN2Conv, Linear
from torch_geometric.utils import add_self_loops
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import degree


class BaseModel(Module, ABC):
    def __init__(self, in_channels, hidden_channels, num_hidden, param_sharing, layerfun, edge_orientation,
                 edge_weights):
        super().__init__()

        # ---  process for concatted ghost feature---
        # dimenson of input  2 * in_channels, dimension of out:  in_channels
        self.ghost_concat_transform = nn.Sequential(
            Linear(in_channels * 2, hidden_channels, weight_initializer="kaiming_uniform"),
            nn.ReLU(),
            Linear(hidden_channels, in_channels, weight_initializer="kaiming_uniform")
        )


        self.encoder = Linear(in_channels, hidden_channels, weight_initializer="kaiming_uniform")
        self.decoder = Linear(hidden_channels, 1, weight_initializer="kaiming_uniform")
        if param_sharing:
            self.layers = ModuleList(num_hidden * [layerfun()])
        else:
            self.layers = ModuleList([layerfun() for _ in range(num_hidden)])
        self.edge_weights = edge_weights
        self.edge_orientation = edge_orientation
        if self.edge_weights is not None:
            self.loop_fill_value = 1.0 if (self.edge_weights == 0).all() else "mean"

    # --- forward take in  ghost_mask and  x_ghost_concat ---
    def forward(self, x, edge_index, ghost_mask=None, x_ghost_concat=None, evo_tracking=False, **kwargs):


        x = x.flatten(1)

        # if input in concated ghost ，use new transform layer
        if hasattr(self, 'ghost_concat_transform') and x_ghost_concat is not None and x_ghost_concat.numel() > 0:

            # use mlp to lower the dimension
            transformed_features = self.ghost_concat_transform(x_ghost_concat)

            # create the x copy，restore the changged feature
            if ghost_mask is not None and ghost_mask.any():
                x_new = x.clone()
                x_new[ghost_mask] = transformed_features

                # use  the x new
                x = x_new

        if self.edge_weights is not None:
            num_graphs = edge_index.size(1) // len(self.edge_weights)
            edge_weights = torch.cat(num_graphs * [self.edge_weights], dim=0).to(x.device)
            edge_weights = edge_weights.abs()
        else:
            edge_weights = torch.zeros(edge_index.size(1)).to(x.device)

        if self.edge_orientation is not None:
            if self.edge_orientation == "upstream":
                edge_index = edge_index[[1, 0]].to(x.device)
            elif self.edge_orientation == "bidirectional":
                edge_index = torch.cat([edge_index, edge_index[[1, 0]]], dim=1).to(x.device)
                edge_weights = torch.cat(2 * [edge_weights], dim=0).to(x.device)
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
        super().__init__(in_channels, hidden_channels, num_hidden, param_sharing, lambda: None, None, None)
        self.layers = ModuleList(
            [Linear(hidden_channels, hidden_channels, weight_initializer="kaiming_uniform") for _ in range(num_hidden)])

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
        return x + relu(layer(x, edge_index))




from torch_geometric.nn import MessagePassing
from torch_geometric.nn.inits import glorot, zeros

class CustomMPNN(BaseModel):
    def __init__(self, in_channels, hidden_channels, num_hidden, param_sharing, edge_orientation, edge_weights):
        # Define a custom message-passing layer generator
        layer_gen = lambda: MPNNLayer(hidden_channels)
        super().__init__(in_channels, hidden_channels, num_hidden, param_sharing, layer_gen, edge_orientation,
                         edge_weights)

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
        return tanh(self.linear_update(aggr_out) + x)