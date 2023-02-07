import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ReLU
from torch_geometric.data.data import Data
from torch_geometric.utils import add_self_loops, spmm, degree, to_torch_coo_tensor

class GCNLayer(nn.Module):

  def __init__(self, input_dim, output_dim):
    super(GCNLayer, self).__init__()
    self.input_dim = input_dim
    self.output_dim = output_dim

    self.linear = nn.Linear(input_dim, output_dim)
    self.act = ReLU(inplace=True)

  def forward(self, x, edge_index):

    A = to_torch_coo_tensor(add_self_loops(edge_index)[0]).to_dense()  # TODO keep everything sparse?
    D = (1 / torch.sqrt(degree(edge_index[0])))

    adj_norm = torch.einsum('i,ij,j->ij', D, A, D)

    x = adj_norm @ x
    x = self.linear(x)
    y = self.act(x)
    return y