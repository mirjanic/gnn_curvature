from enum import Enum

import numpy as np
import torch
import torch.nn as nn
from torch.nn import Embedding, Sequential, ReLU, Linear
from torch_geometric.nn import GATConv, GCNConv, GraphConv, SAGEConv, GINConv, MLP
from torch_scatter import scatter_sum

from networks.layers.eigengat import EigenGAT
from networks.layers.gcn_layer import GCNLayer
from networks.layers.rotationconv import RotationConv
from networks.layers.featrotationconv import FeatureRotationConv

import torch.nn.functional as F

from networks.layers.sheafconv import SheafConv


class ModelType(Enum):
  gcn = 'gcn'
  gat = 'gat'
  eigen_gat = 'eigen_gat'
  rotation_conv = 'rotations'
  feature_rotation_conv = 'feat_rotations'
  sheaf = 'sheaf'
  mpnn = 'mpnn'
  sage = 'sage'
  gin = 'gin'

  def takes_eigens(self):
    match self:
      case ModelType.gat | ModelType.sheaf | ModelType.mpnn | ModelType.gcn | ModelType.sage | ModelType.gin:
        return False
      case ModelType.eigen_gat | ModelType.rotation_conv | ModelType.feature_rotation_conv:
        return True
      case _:
        raise ValueError('Invalid model')

  def has_dimensions(self):
    match self:
      case ModelType.gat | ModelType.eigen_gat | ModelType.mpnn | ModelType.gcn | ModelType.sage | ModelType.gin:
        return False
      case ModelType.rotation_conv | ModelType.feature_rotation_conv | ModelType.sheaf:
        return True
      case _:
        raise ValueError('Invalid model')

  def make_layer(self, in_channels, out_channels, **kwargs):
    match self:
      case ModelType.gcn:
        return GCNConv(in_channels=in_channels,
                       out_channels=out_channels)

      case ModelType.gat:
        return GATConv(in_channels=in_channels,
                       out_channels=out_channels)

      case ModelType.eigen_gat:
        return EigenGAT(in_channels=in_channels,
                        out_channels=out_channels,
                        eigen_count=kwargs['eigen_count'])

      case ModelType.rotation_conv:
        return RotationConv(in_channels=in_channels,
                            out_channels=out_channels,
                            eigen_count=kwargs['eigen_count'],
                            dimension=kwargs['dimension'])

      case ModelType.feature_rotation_conv:
        return FeatureRotationConv(in_channels=in_channels,
                                   out_channels=out_channels,
                                   eigen_count=kwargs['eigen_count'],
                                   dimension=kwargs['dimension'])

      case ModelType.sheaf:
        return SheafConv(in_channels=in_channels,
                         out_channels=out_channels,
                         dim=kwargs['dimension'])

      case ModelType.mpnn:
        return GraphConv(in_channels=in_channels,
                         out_channels=out_channels)

      case ModelType.gcn:
        return GCNConv(in_channels=in_channels,
                       out_channels=out_channels)

      case ModelType.sage:
        return SAGEConv(in_channels=in_channels,
                        out_channels=out_channels)

      case ModelType.gin:
        return GINConv(nn=MLP([in_channels, out_channels, out_channels]))

      case _:
        raise ValueError('Invalid model')


class TaskType(Enum):
  node = 'node'
  graph = 'graph'


class TestModel(nn.Module):

  def __init__(self, model: ModelType,
               input_dim: int,
               hidden_dim: int,
               output_dim: int,
               num_layers: int,
               **kwargs):
    super(TestModel, self).__init__()

    self.model = model
    self.task = kwargs['task']

    total_hidden_dim = hidden_dim

    if self.model.takes_eigens():
      self.spft_name = kwargs['spatial_name']
    if self.model.has_dimensions():
      hidden_dim = hidden_dim // kwargs['dimension']

    self.num_layers = num_layers

    match self.task:
      case TaskType.node:
        self.embed = Linear(input_dim, total_hidden_dim)
      case TaskType.graph:
        self.embed = Embedding(28, total_hidden_dim)

    self.layers = [self.model.make_layer(in_channels=total_hidden_dim,
                                         out_channels=hidden_dim,
                                         **kwargs)
                   for _ in range(num_layers)]

    self.layers = nn.ModuleList(self.layers)
    self.mlp = Sequential(Linear(total_hidden_dim, hidden_dim),
                          ReLU(inplace=True),
                          Linear(hidden_dim, output_dim))

  def forward(self, data):
    x = data.x
    match self.task:
      case TaskType.node:
        x = self.embed(x).squeeze(1)
      case TaskType.graph:
        batch = data.batch
        x = self.embed(x.long()).squeeze(1)

    edge_index = data.edge_index

    if self.model.takes_eigens():
      eigens = data[self.spft_name]
    else:
      eigens = None

    for i in range(self.num_layers):
      if self.model.takes_eigens():
        x = self.layers[i](x, edge_index, eigens)
      else:
        x = self.layers[i](x, edge_index)
      x = F.relu(x)

    if self.task == TaskType.graph:
      x = scatter_sum(x, batch, dim=0)

    x = F.dropout(x, p=0.5, training=self.training)
    y = self.mlp(x)
    y = y.squeeze(-1)
    return y


class SimpleGCN(torch.nn.Module):
  def __init__(self, input_dim, hidden_dim, output_dim):
    super().__init__()
    torch.manual_seed(1234567)
    self.conv1 = GCNConv(input_dim, hidden_dim)
    self.conv2 = GCNConv(hidden_dim, output_dim)

  def forward(self, data):
    x = data.x
    edge_index = data.edge_index
    x = self.conv1(x, edge_index)
    x = x.relu()
    x = F.dropout(x, p=0.5, training=self.training)
    x = self.conv2(x, edge_index)
    return x