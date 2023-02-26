#
# This code is based on the following:
# https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/conv/gat_conv.html#GATConv
#

from typing import Union, Optional

import torch
import torch.nn.functional as F
import torch_sparse
from torch import Tensor
from torch.nn import Parameter

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import (
  Adj,
  OptTensor,
  Size,
  SparseTensor,
  )
from torch_geometric.utils import add_self_loops, remove_self_loops, softmax


class EigenGAT(MessagePassing):

  def __init__(
      self,
      in_channels: int,  # Simplified from original GAT
      out_channels: int,
      eigen_count: int,  # Number of eigenvalues to use for attention
      heads: int = 1,
      negative_slope: float = 0.2,
      add_self_loops: bool = True,
      edge_dim: Optional[int] = None,
      fill_value: Union[float, Tensor, str] = 'mean',
      bias: bool = True,
      **kwargs,
      ):
    kwargs.setdefault('aggr', 'add')
    super().__init__(node_dim=0, **kwargs)

    self.in_channels = in_channels
    self.out_channels = out_channels
    self.heads = heads
    self.negative_slope = negative_slope
    self.add_self_loops = add_self_loops
    self.edge_dim = edge_dim
    self.fill_value = fill_value

    self.eigen_count = eigen_count

    self.feature_lin = Linear(in_channels, heads * out_channels, bias=False, weight_initializer='glorot')
    self.eigen_lin = Linear(eigen_count, heads, bias=False, weight_initializer='glorot')

    # The learnable parameters to compute attention coefficients:
    self.att_src = Parameter(torch.Tensor(1, heads, out_channels))
    self.att_dst = Parameter(torch.Tensor(1, heads, out_channels))

    if edge_dim is not None:
      self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False,
                             weight_initializer='glorot')
      self.att_edge = Parameter(torch.Tensor(1, heads, out_channels))
    else:
      self.lin_edge = None
      self.register_parameter('att_edge', None)

    if bias:
      self.bias = Parameter(torch.Tensor(heads * out_channels))
    else:
      self.register_parameter('bias', None)

    self.reset_parameters()

  def reset_parameters(self):
    # super().reset_parameters()
    self.feature_lin.reset_parameters()
    self.eigen_lin.reset_parameters()
    if self.lin_edge is not None:
      self.lin_edge.reset_parameters()
    glorot(self.att_src)
    glorot(self.att_dst)
    glorot(self.att_edge)
    zeros(self.bias)

  def forward(self,
              x: Tensor,  # Simplified from original GAT
              edge_index: Adj,
              eigens: Tensor,  # EigenGAT uses eigenvalues for attention rather than features!
              ):

    H, C = self.heads, self.out_channels

    # We first transform the input node features.
    assert eigens.dim() == 2 and x.dim() == 2, "Static graphs not supported in 'GATConv'"
    x = self.feature_lin(x).view(-1, H, C)
    eigens = self.eigen_lin(eigens).view(-1, H, 1)

    # Next, we compute node-level attention coefficients, both for source
    # and target nodes (if present):
    alpha_src = (eigens * self.att_src).sum(dim=-1)
    alpha_dst = (eigens * self.att_dst).sum(dim=-1)
    alpha = (alpha_src, alpha_dst)

    # Do self loops
    num_nodes = eigens.size(0)
    edge_index, _ = remove_self_loops(edge_index)
    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)

    alpha = self.edge_updater(edge_index, alpha=alpha)

    # Note: we only use x for propagation / message passing
    out = self.propagate(edge_index, x=x, alpha=alpha)

    out = out.mean(dim=1)

    if self.bias is not None:
      out = out + self.bias

    return out

  def edge_update(self, alpha_j: Tensor, alpha_i: OptTensor, index: Tensor, ptr: OptTensor) -> Tensor:
    # Given edge-level attention coefficients for source and target nodes,
    # we simply need to sum them up to "emulate" concatenation:
    alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
    alpha = F.leaky_relu(alpha, self.negative_slope)
    alpha = softmax(alpha, index, ptr)
    return alpha

  def message(self, x_j: Tensor, alpha: Tensor) -> Tensor:
    return alpha.unsqueeze(-1) * x_j

  def __repr__(self) -> str:
    return (f'{self.__class__.__name__}({self.in_channels}, '
            f'{self.out_channels}, heads={self.heads})')
