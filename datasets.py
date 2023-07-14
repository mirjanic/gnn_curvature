from typing import Iterable, Tuple

import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import WebKB, ZINC, Planetoid, WikipediaNetwork
from torch_geometric.transforms import NormalizeFeatures


def load_node_class_datasets(device, name, spatial_count):
  """
  Precompute eigenvectors, move data to GPU, shuffle and minibatch
  """
  transforms = T.Compose([
    T.AddLaplacianEigenvectorPE(k=spatial_count, attr_name='eigens'),
    T.AddRandomWalkPE(walk_length=spatial_count, attr_name='walks')
    ])

  if name in ['texas', 'wisconsin', 'cornell']:
    raw = WebKB(root=f'data/{name}', name=name, pre_transform=transforms)
  elif name in ['chameleon', 'crocodile', 'squirrel']:
    raw = WikipediaNetwork(root=f'data/{name}', name=name, pre_transform=transforms)
    # For some reason, PubMed and CiteSeer don't work
  elif name in ['cora']:
    raw = Planetoid(root=f'data/{name}', name=name, pre_transform=transforms, transform=NormalizeFeatures())
  else:
    raise NameError(f'Error: {name} dataset not included.')

  raw.data = raw.data.to(device)

  return raw


def load_zinc_datasets(device, batch_size, spatial_count) -> Tuple[Iterable, Iterable, Iterable]:
  transforms = T.Compose([
    T.AddLaplacianEigenvectorPE(k=spatial_count, attr_name='eigens'),
    T.AddRandomWalkPE(walk_length=spatial_count, attr_name='walks')
    ])

  splits = ['train', 'val', 'test']

  datasets = {k: ZINC(root='data/zinc', split=k, subset=True, pre_transform=transforms) for k in splits}

  for v in datasets.values():
    v.data = v.data.to(device)

  dataloaders = {k: DataLoader(v, batch_size=batch_size, shuffle=True) for (k, v) in datasets.items()}

  return dataloaders['train'], dataloaders['val'], dataloaders['test']