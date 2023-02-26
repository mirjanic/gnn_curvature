import torch
from torch import Tensor


def rotationize(x: Tensor):
  """
  Convert an input tensor into block diagonal rotation matrix
  :param x: 3D Tensor of shape x*d*d
  :return: Tensor of shape x*d*d where all d*d matrices are in O(d)
  """
  # TODO make special implementations for d=2,3
  upper = torch.triu(x)                                 # Take upper triangular part of input
  skew_sym = upper - torch.transpose(upper, -1, -2)     # Create skew symmetric matrix
  rotation = torch.linalg.matrix_exp(skew_sym)          # Exponentiate to get an orthogonal matrix
  return rotation
