import torch
from torch import Tensor


def rotationize(x: Tensor):
  """
  Convert an input tensor into block diagonal rotation matrix
  :param x: 3D Tensor of shape d*d*k
  :return: Block diagonal matrix of shape (d*k)*(d*k) where each d*d block is in O(d)
  """
  upper = torch.triu(x)                                 # Take upper triangular part of input
  skew_sym = upper - torch.transpose(upper, 1, 2)       # Create skew symmetric matrix
  rotation = torch.linalg.matrix_exp(skew_sym)          # Exponentiate to get an orthogonal matrix
  #block_rot = torch.block_diag(*rotation)               # Stack orthogonal matrices into block diagonal
  return rotation
