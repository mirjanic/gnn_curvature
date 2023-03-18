from torch import Tensor, einsum
import torch.nn.functional as F

import torch
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.utils import add_self_loops, remove_self_loops, softmax
from torch_geometric.typing import (
  Adj,
  OptTensor,
  Optional
  )

from networks.util.rotationizer import rotationize


# noinspection PyMethodOverriding
class FeatureRotationConv(MessagePassing):

  def __init__(self,
               in_channels: int,
               out_channels: int,
               eigen_count: int,
               dimension: int,
               normalize: bool = True,
               **kwargs):

    kwargs.setdefault('aggr', 'add')
    super().__init__(**kwargs)

    self.in_channels = in_channels
    self.out_channels = out_channels
    self.dimension = dimension
    self.normalize = normalize
    self.leak = 0.2

    self.feature_lin = Linear(dimension * out_channels,
                              dimension * out_channels,
                              bias=False, weight_initializer='glorot')

    self.rot_lin = Linear(dimension * out_channels + eigen_count,
                          dimension * dimension * out_channels,
                          bias=False, weight_initializer='glorot')

    self.reset_parameters()

  def reset_parameters(self):
    self.feature_lin.reset_parameters()
    self.rot_lin.reset_parameters()

  def forward(self, x: Tensor,
              edge_index: Adj,
              eigens: Tensor) -> Tensor:

    # Apply weights to input
    x = self.feature_lin(x)

    # Apply weights to eigens and input
    rot_features = self.rot_lin(torch.cat([x, eigens], dim=1))

    # Do self loops
    num_nodes = rot_features.size(0)
    edge_index, _ = remove_self_loops(edge_index)
    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)

    # Create edge attention
    rot_features = self.edge_updater(edge_index, e=rot_features)
    rot_features = rot_features.reshape(-1, self.out_channels, self.dimension, self.dimension)

    # Convert edge attentions into rotations
    rotations = rotationize(rot_features)

    # Propagate with rotations
    out = self.propagate(edge_index, x=x, rotations=rotations, size=None)

    return out

  def edge_update(self, e_j: Tensor, e_i: OptTensor,
                  index: Tensor,
                  ptr: OptTensor,
                  size_i: Optional[int]) -> Tensor:
    # Using sum to emulate concatenation as in GAT
    eigens = e_j + e_i
    eigens = F.leaky_relu(eigens, self.leak)
    eigens = softmax(eigens, index, ptr, size_i)
    return eigens

  def message(self, x_j: Tensor, rotations: Tensor) -> Tensor:
    # Reshape messages to be easier to rotate
    x_j = x_j.reshape(-1, self.out_channels, self.dimension)
    # Multiply channel blocks
    out = torch.einsum('ncij,nci->ncj', rotations, x_j)
    # Reshape back
    out = out.reshape(-1, self.out_channels * self.dimension)
    return out
