import torch
from torch import Tensor
from torch_geometric.nn.dense.linear import Linear
from torch.nn import Module
from torch_geometric.typing import Adj
from torch_geometric.utils import add_self_loops, degree, to_torch_coo_tensor


class GCNLayer(Module):

  def __init__(self, input_dim, output_dim):
    super(GCNLayer, self).__init__()
    self.input_dim = input_dim
    self.output_dim = output_dim

    self.linear = Linear(input_dim, output_dim)
    # self.act = ReLU(inplace=True)

  def forward(self, x: Tensor, edge_index: Adj):

    A = to_torch_coo_tensor(add_self_loops(edge_index)[0]).to_dense()  # TODO keep everything sparse?
    D = (1 / torch.sqrt(degree(edge_index[0])))

    adj_norm = torch.einsum('i,ij,j->ij', D, A, D)

    x = self.linear(x)
    x = adj_norm @ x
    # y = self.act(x)
    return x