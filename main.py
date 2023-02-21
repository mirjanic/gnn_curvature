from typing import Tuple, Iterable

import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data.data import Data
from torch_geometric.datasets.zinc import ZINC

import torch_geometric.transforms as T
import numpy as np
import random

from absl import app
from absl import flags
from absl import logging

from networks.test_gat import TestGAT
from networks.test_rot import TestRotConv

FLAGS = flags.FLAGS

flags.DEFINE_float('lr', 0.001, 'Learning rate')
flags.DEFINE_integer('epochs', 100, 'Epochs')
flags.DEFINE_integer('seed', 42, 'Random seed')
flags.DEFINE_integer('bs', 64, 'Batch size')
flags.DEFINE_integer('num_eigens', 5, 'Number of eigenvector features to generate.')
flags.DEFINE_integer('hidden_dim', 128, 'Number of latent dimensions')
flags.DEFINE_integer('num_layers', 4, 'Number of convolutions to perform')


def set_seed(seed):
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  np.random.seed(seed)
  random.seed(seed)


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
  transform = T.AddLaplacianEigenvectorPE(k=FLAGS.num_eigens, attr_name='eigens')

  splits = ['train', 'val', 'test']

  datasets = {k: ZINC(root='data/zinc', split=k, subset=True, pre_transform=transform) for k in splits}

  for v in datasets.values():
    v.data = v.data.to(device)

  dataloaders = {k: DataLoader(v, batch_size=FLAGS.bs, shuffle=True) for (k, v) in datasets.items()}

  return dataloaders['train'], dataloaders['val'], dataloaders['test']


def main(unused_argv):
  set_seed(FLAGS.seed)
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  # Important! If you already downloaded non-transformed dataset you need to delete it
  train_dl, val_dl, test_dl = load_datasets(device)

  # Create model and move to device
  # model = TestGAT(hidden_dim=FLAGS.hidden_dim,
  #                 output_dim=1,
  #                 num_layers=FLAGS.num_layers,
  #                 eigen_count=FLAGS.num_eigens)
  model = TestRotConv(hidden_dim=FLAGS.hidden_dim // 2,
                      dimension=2,
                      output_dim=1,
                      num_layers=FLAGS.num_layers,
                      eigen_count=FLAGS.num_eigens)
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


if __name__ == '__main__':
  app.run(main)
