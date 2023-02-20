import torch.nn as nn
from torch.nn import Embedding, Sequential, ReLU, Linear
from torch_scatter import scatter_sum
import torch.nn.functional as F

from networks.layers.sheafconv import SheafConv


class TestSheafConv(nn.Module):

    def __init__(self, hidden_dim, output_dim, num_layers, dimension):
        super(TestSheafConv, self).__init__()

        self.num_layers = num_layers
        self.embed = Embedding(28, hidden_dim * dimension)

        self.layers = [SheafConv(in_channels=hidden_dim * dimension,
                                 out_channels=hidden_dim,
                                 dim=dimension)
                       for _ in range(num_layers)]
        self.layers = nn.ModuleList(self.layers)
        self.mlp = Sequential(Linear(hidden_dim * dimension, hidden_dim),
                              ReLU(inplace=True),
                              Linear(hidden_dim, output_dim))

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        batch = data.batch

        x = self.embed(x.long()).squeeze(1)

        for i in range(self.num_layers):
            x = self.layers[i](x, edge_index)
            x = F.relu(x)

        y = scatter_sum(x, batch, dim=0)
        y = self.mlp(y)
        y = y.squeeze(-1)
        return y
