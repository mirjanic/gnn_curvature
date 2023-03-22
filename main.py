from typing import Tuple, Iterable

import torch
from torch_geometric.datasets import WebKB, Planetoid
from torch_geometric.loader import DataLoader
from torch_geometric.data.data import Data
from torch_geometric.datasets.zinc import ZINC

import torch_geometric.transforms as T
import numpy as np
import random

from absl import app
from absl import flags
from absl import logging

from datasets import get_dataset, get_fixed_splits
from networks.test_model import TestModel, ModelType, TaskType

FLAGS = flags.FLAGS

# Training params
flags.DEFINE_float('lr', 1e-3, 'Learning rate')
flags.DEFINE_integer('epochs', 10, 'Epochs')
flags.DEFINE_integer('seed', 42, 'Random seed')
flags.DEFINE_integer('bs', 64, 'Batch size')

# Model params
flags.DEFINE_enum('model_name', 'rotations', ['gcn', 'gat', 'eigen_gat', 'rotations', 'feat_rotations', 'sheaf'], 'Model to train')
flags.DEFINE_integer('num_layers', 4, 'Number of convolutions to perform')
flags.DEFINE_integer('hidden_dim', 64, 'Number of latent dimensions')
flags.DEFINE_enum('task', 'node', ['node', 'graph'], 'Type of classification task')

# Features params
# TODO I couldn't get more than 6 features to work
flags.DEFINE_integer('spatial_features_count', 6, 'Number of eigenvector and random walk features to generate.')
flags.DEFINE_enum('spatial_features_name', 'eigens', ['eigens', 'walks'], 'Whether to use eigenvectors or random walks')
flags.DEFINE_integer('dimension', 2, 'Rotation dimensions. Used only for rotations.')


def set_seed(seed):
  torch.manual_seed(seed)
  np.random.seed(seed)
  random.seed(seed)
  # torch.backends.cudnn.benchmark = False
  # torch.use_deterministic_algorithms(True)

def train(model, optimizer, data: Data, criterion):
  model.train()
  optimizer.zero_grad()
  pred = model(data)
  loss = criterion(pred, data.y)
  loss.backward()
  optimizer.step()
  return loss


def test(model, data, criterion):
  model.eval()
  with torch.no_grad():
    pred = model(data)
    loss = criterion(pred, data.y)
    return loss


def load_datasets(device) -> Tuple[Iterable, Iterable, Iterable]:
  """
  Precompute eigenvectors, move data to GPU, shuffle and minibatch
  """
  transforms = T.Compose([
    T.AddLaplacianEigenvectorPE(k=FLAGS.spatial_features_count, attr_name='eigens'),
    T.AddRandomWalkPE(walk_length=FLAGS.spatial_features_count, attr_name='walks')
    ])

  splits = ['train', 'val', 'test']

  datasets = {k: ZINC(root='data/zinc', split=k, subset=True, pre_transform=transforms) for k in splits}

  for v in datasets.values():
    v.data = v.data.to(device)

  dataloaders = {k: DataLoader(v, batch_size=FLAGS.bs, shuffle=True) for (k, v) in datasets.items()}

  return dataloaders['train'], dataloaders['val'], dataloaders['test']


def main(unused_argv):
  if FLAGS.task == 'node':
    masked_main(unused_argv)
    return

  set_seed(FLAGS.seed)
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  # Important! If you already downloaded non-transformed dataset you need to delete it
  train_dl, val_dl, test_dl = load_datasets(device)

  logging.info(f"Training model '{FLAGS.model_name}'")

  model_kwargs = {
    'model': ModelType(FLAGS.model_name),
    'num_layers': FLAGS.num_layers,
    'eigen_count': FLAGS.spatial_features_count,
    'spatial_name': FLAGS.spatial_features_name,
    'input_dim': 'TODO',
    'hidden_dim': FLAGS.hidden_dim,
    'dimension': FLAGS.dimension,
    'output_dim': 1,
    'task': TaskType(FLAGS.task)
  }

  model = TestModel(**model_kwargs)
  model.to(device)

  optimizer = torch.optim.Adam(params=model.parameters(), lr=FLAGS.lr)
  criterion = torch.nn.L1Loss()

  for epoch in range(FLAGS.epochs):

    losses = []
    for batch in train_dl:
      losses += [train(model, optimizer, batch, criterion)]
    train_loss = torch.mean(torch.Tensor(losses))

    losses = []
    for batch in val_dl:
      losses += [test(model, batch, criterion)]
    val_loss = torch.mean(torch.Tensor(losses))

    logging.info(f"epoch: {epoch} \t train: {train_loss:.4f} \t val: {val_loss:.4f}")

  losses = []
  for batch in test_dl:
    losses += [test(model, batch, criterion)]
  logging.info(f"Final loss: <<< {torch.mean(torch.Tensor(losses)):.4f} >>>")

  return


def train_masked(model, optimizer, data, criterion):
  model.train()
  optimizer.zero_grad()
  pred = model(data)[data.train_mask]
  loss = criterion(pred, data.y[data.train_mask])
  loss.backward()
  optimizer.step()
  return loss


def test_masked(model, data, criterion):
  model.eval()
  with torch.no_grad():
    pred = model(data)
    loss = criterion(pred[data.test_mask], data.y[data.test_mask])
    return loss


def load_masked_datasets(device):
  """
  Precompute eigenvectors, move data to GPU, shuffle and minibatch
  """
  transforms = T.Compose([
    T.AddLaplacianEigenvectorPE(k=FLAGS.spatial_features_count, attr_name='eigens'),
    T.AddRandomWalkPE(walk_length=FLAGS.spatial_features_count, attr_name='walks')
    ])

  splits = ['train', 'val', 'test']

  # raw = WebKB(root='data/texas', name='texas', pre_transform=transforms)
  raw = Planetoid(root='data/cora', name='cora', pre_transform=transforms)
  raw.data = raw.data.to(device)

  return raw[0]


def masked_main(unused_argv):
  set_seed(FLAGS.seed)
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  data = load_masked_datasets(device)

  logging.info(f"Training model '{FLAGS.model_name}'")

  model_kwargs = {
    'model': ModelType(FLAGS.model_name),
    'num_layers': FLAGS.num_layers,
    'eigen_count': FLAGS.spatial_features_count,
    'spatial_name': FLAGS.spatial_features_name,
    'input_dim': data.x.shape[1],
    'hidden_dim': FLAGS.hidden_dim,
    'dimension': FLAGS.dimension,
    'output_dim': 1,
    'task': TaskType(FLAGS.task)
  }

  model = TestModel(**model_kwargs)
  model.to(device)

  optimizer = torch.optim.Adam(params=model.parameters(), lr=FLAGS.lr)
  criterion = torch.nn.L1Loss()

  for epoch in range(FLAGS.epochs):
    loss = train_masked(model, optimizer, data, criterion)
    print(f"loss: {loss}")


if __name__ == '__main__':
  app.run(main)
