import torch
from torch import Tensor


def rotationize(x: Tensor):
  """
  Convert an input tensor into block diagonal rotation matrix
  :param x: 3D Tensor of shape (...,d,d)
  :return: Tensor of shape (...,d,d) where all d*d matrices are in O(d)
  """
  d = x.shape[-1]

  match d:
    case 2:
      angle = x[..., 0, 0] * torch.pi
      c = torch.cos(angle)
      s = torch.sin(angle)
      rotation = torch.stack([c, s, -s, c], dim=-1)
      rotation = torch.reshape(rotation, (*rotation.shape[:-1], 2, 2))
    case 3:
      angles = x[..., 0, :] * torch.pi
      c = torch.cos(angles)
      s = torch.sin(angles)
      # From https://en.wikipedia.org/wiki/Euler_angles#Rotation_matrix (Example 1)
      rotation = torch.stack([c[..., 1], -c[..., 2]*s[..., 1], s[..., 1]*s[..., 2],

                               c[..., 0]*s[..., 1], c[..., 0]*c[..., 1]*c[..., 2] - s[..., 0]*s[..., 2],
                                -c[..., 2]*s[..., 0] - c[..., 0]*c[..., 1]*s[..., 2],

                               s[..., 0]*s[..., 1], c[..., 0]*s[..., 2]+c[..., 1]*c[..., 2]*s[..., 0],
                                c[..., 0]*c[..., 2]-c[..., 1]*s[..., 0]*s[..., 2],
                               ], dim=-1)
      rotation = torch.reshape(rotation, (*rotation.shape[:-1], 3, 3))
    case _:
      upper = torch.triu(x)                                 # Take upper triangular part of input
      skew_sym = upper - torch.transpose(upper, -1, -2)     # Create skew symmetric matrix
      rotation = torch.linalg.matrix_exp(skew_sym)          # Exponentiate to get an orthogonal matrix

  return rotation
