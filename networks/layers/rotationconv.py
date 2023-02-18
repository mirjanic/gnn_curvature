import functorch
import torch
from torch import Tensor, einsum

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import (
  Adj,
  OptTensor,
  )

from networks.util.rotationizer import rotationize


class RotationConv(MessagePassing):

  def __init__(self,
               in_channels: int,
               out_channels: int,
               eigen_count: int,
               dimension: int,
               improved: bool = False,
               add_self_loops: bool = True,
               normalize: bool = True,
               **kwargs):

    kwargs.setdefault('aggr', 'add')
    super().__init__(**kwargs)

    self.in_channels = in_channels
    self.out_channels = out_channels
    self.dimension = dimension
    self.improved = improved
    self.add_self_loops = add_self_loops
    self.normalize = normalize

    self.feature_lin = Linear(in_channels, dimension * out_channels, bias=False, weight_initializer='glorot')
    self.eigen_lin = Linear(eigen_count, dimension * dimension * out_channels, bias=False, weight_initializer='glorot')

    self.reset_parameters()

  def reset_parameters(self):
    self.feature_lin.reset_parameters()
    self.eigen_lin.reset_parameters()

  def forward(self, x: Tensor,
              edge_index: Adj,
              eigens: Tensor) -> Tensor:

    if self.normalize:
      edge_index, edge_weight = gcn_norm(
        edge_index, None, x.size(self.node_dim),
        self.improved, self.add_self_loops, self.flow, x.dtype)
    else:
      edge_weight = None

    x = self.feature_lin(x)

    out = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=None)

    eigens = self.eigen_lin(eigens)
    eigens = eigens.reshape(-1, self.dimension, self.dimension)

    rotations = rotationize(eigens)
    rotations = rotations.reshape(-1, self.out_channels, self.dimension, self.dimension)

    out = out.reshape(-1, self.out_channels, self.dimension)
    out = einsum('ncij,nci->ncj', rotations, out)  # Rotations applied here
    out = out.reshape(-1, self.out_channels * self.dimension)
    # TODO is this the correct way to apply rotations?
    # TODO are rotations in correct shape?
    # TODO: Is not having bias correct?

    return out

  def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
    return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j
