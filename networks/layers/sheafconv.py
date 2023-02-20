import torch
from torch import Tensor
from torch.nn import Parameter
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptTensor

from networks.util.rotationizer import rotationize


class SheafConv(MessagePassing):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 dim: int,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dim = dim

        self.feature_lin = Linear(in_channels, out_channels * dim)
        self.V1 = Linear(out_channels, 1)
        self.V2 = Linear(out_channels, 1)
        self.W = Parameter(torch.eye(dim))

    def forward(self, x: Tensor, edge_index: Adj):
        n = x.shape[0]
        x = self.feature_lin(x)
        out = self.propagate(edge_index, x=x, edge_weight=None, size=None)
        x = x.reshape(-1, self.out_channels)

        # Compute d x d matrix for each pair of vertices (should this be for each edge?)
        V1 = self.V1(x).unsqueeze(0)
        V2 = self.V2(x).unsqueeze(1)
        V = V1 + V2
        V = V.reshape(-1, self.dim, self.dim)

        # Make each d x d matrix orthogonal, then form block matrix out of these
        sheaf_laplacian = rotationize(V)
        sheaf_laplacian = sheaf_laplacian.reshape(n, n, self.dim, self.dim)
        sheaf_laplacian = sheaf_laplacian.swapaxes(1, 2)
        sheaf_laplacian = sheaf_laplacian.reshape(n * self.dim, n * self.dim)

        # Construct I (x) W
        W = self.W.reshape(1, self.dim, self.dim)
        W = W.repeat((n, 1, 1))
        W = torch.block_diag(*W)

        out = torch.softmax(sheaf_laplacian @ W @ out.reshape(-1, self.out_channels), dim=0)
        # Residual connection
        out = x - out
        out = out.reshape(-1, self.out_channels * self.dim)

        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j
