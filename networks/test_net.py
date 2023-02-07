import torch.nn as nn
from torch.nn import Embedding
from torch_scatter import scatter_sum

from networks.gcn_layer import GCNLayer


class TestNet(nn.Module):

  def __init__(self, hidden_dim, output_dim):
    super(TestNet, self).__init__()

    self.embed = Embedding(28, hidden_dim)
    self.conv = GCNLayer(hidden_dim, output_dim)

  def forward(self, data):
    x = data.x
    edge_index = data.edge_index
    batch = data.batch
    y = self.embed(x)[:, 0, :]
    y = self.conv(y, edge_index)
    y = y.squeeze(-1)
    y = scatter_sum(y, batch)
    return y
