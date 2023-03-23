from typing import Tuple, Iterable, Optional

import time

import torch
from torch_geometric.data.data import Data
import numpy as np
import random

from absl import app
from absl import flags
from absl import logging

from datasets import load_zinc_datasets, load_node_class_datasets
from networks.test_model import TestModel, ModelType, TaskType, SimpleGCN

FLAGS = flags.FLAGS

# Training params
flags.DEFINE_float('lr', 1e-3, 'Learning rate')
flags.DEFINE_integer('epochs', 1000, 'Epochs')
flags.DEFINE_integer('seed', 42, 'Random seed')
flags.DEFINE_integer('bs', 64, 'Batch size')

# Model params
flags.DEFINE_enum('model_name', 'feat_rotations',
                  ['gcn', 'gat', 'eigen_gat', 'rotations', 'feat_rotations', 'sheaf', 'mpnn', 'sage', 'gin'],
                  'Model to train')
flags.DEFINE_integer('num_layers', 4, 'Number of convolutions to perform')
flags.DEFINE_integer('hidden_dim', 16, 'Number of latent dimensions')

flags.DEFINE_enum('dataset', 'zinc', ['zinc', 'texas', 'wisconsin', 'cornell', 'cora'],
                  'Dataset used for training (includes both node and graph-level classification.)')

# Features params
flags.DEFINE_integer('spatial_features_count', 6, 'Number of eigenvector and random walk features to generate.')
flags.DEFINE_enum('spatial_features_name', 'eigens', ['eigens', 'walks'], 'Whether to use eigenvectors or random walks')
flags.DEFINE_integer('dimension', 2, 'Rotation dimensions. Used only for rotations.')


def set_seed(seed):
  torch.manual_seed(seed)
  np.random.seed(seed)
  random.seed(seed)
  # torch.backends.cudnn.benchmark = False
  # torch.use_deterministic_algorithms(True)


def train(model, optimizer, data: Data, criterion, mask=Optional[Iterable]):
  model.train()
  optimizer.zero_grad()
  pred = model(data)

  if mask is None:
    loss = criterion(pred, data.y)
  else:
    loss = criterion(pred[mask], data.y[mask])

  loss.backward()
  optimizer.step()
  return loss


def test(model, data, criterion, mask=Optional[Iterable]):
  model.eval()
  with torch.no_grad():
    pred = model(data)
    if mask is None:
      return criterion(pred, data.y)
    else:
      top_pred = pred.argmax(dim=1)
      correct = top_pred[mask] == (data.y[mask])
      return int(correct.sum()) / int(mask.sum())


def run_node_class_experiment(model, data: Data, device):
  criterion = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
  model.to(device)

  train_mask = data.train_mask
  test_mask = data.val_mask
  val_mask = data.test_mask

  # If mask is 2D, use first split
  if len(train_mask.shape) == 2:
    train_mask = train_mask[:, 0]
    val_mask = val_mask[:, 0]
    test_mask = test_mask[:, 0]

  train_losses = []
  val_losses = []
  start = time.time()
  for epoch in range(FLAGS.epochs):

    train_losses += [train(model, optimizer, data, criterion, train_mask)]
    val_losses += [test(model, data, criterion, val_mask)]

    # Checkpoint best model
    if val_losses[-1] == max(val_losses):
      torch.save(model.state_dict(), 'best.pt')

    if epoch % 10 == 0:
      logging.info(f"epoch: {epoch} \t train: {train_losses[-1]:.4f} \t val: {val_losses[-1]:.4f}")

  end = time.time()
  duration = end - start

  model.load_state_dict(torch.load('best.pt'))

  train_losses = torch.stack(train_losses).cpu().detach().numpy()
  test_loss = test(model, data, criterion, test_mask)

  logging.info(f"Final accuracy: {test_loss:.4f}, at {duration / FLAGS.epochs} seconds per epochs.")

  return train_losses, val_losses, test_loss, duration


def run_graph_class_experiment(model, dataloaders: Tuple[Iterable, Iterable, Iterable], device):
  train_dl, val_dl, test_dl = dataloaders

  criterion = torch.nn.L1Loss()
  optimizer = torch.optim.Adam(params=model.parameters(), lr=FLAGS.lr)
  model.to(device)

  train_losses = []
  val_losses = []
  start = time.time()

  for epoch in range(FLAGS.epochs):
    losses = []
    for batch in train_dl:
      losses += [train(model, optimizer, batch, criterion, mask=None)]
    train_loss = torch.mean(torch.Tensor(losses))
    train_losses.append(train_loss)

    losses = []
    for batch in val_dl:
      losses += [test(model, batch, criterion, mask=None)]
    val_loss = torch.mean(torch.Tensor(losses))
    val_losses.append(val_loss)

    # Checkpoint best model
    if val_loss == min(val_losses):
      torch.save(model.state_dict(), 'best.pt')

    logging.info(f"epoch: {epoch} \t train: {train_loss:.4f} \t val: {val_loss:.4f}")
  end = time.time()
  duration = end - start

  model.load_state_dict(torch.load('best.pt'))

  losses = []
  for batch in test_dl:
    losses += [test(model, batch, criterion, mask=None)]
  test_loss = torch.mean(torch.Tensor(losses))

  logging.info(f"Final loss: {test_loss:.4f}, at {duration / FLAGS.epochs} seconds per epochs.")

  return train_losses, val_losses, test_loss, duration


def main(unused_argv):
  set_seed(FLAGS.seed)
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  task = TaskType.graph if FLAGS.dataset == 'zinc' else TaskType.node

  model_kwargs = {
    'model': ModelType(FLAGS.model_name),
    'num_layers': FLAGS.num_layers,
    'eigen_count': FLAGS.spatial_features_count,
    'spatial_name': FLAGS.spatial_features_name,
    'hidden_dim': FLAGS.hidden_dim,
    'dimension': FLAGS.dimension,
    'task': task
  }

  logging.info(f"Training model '{FLAGS.model_name}'")

  # If performing node classification, run experiments with masking applied to data.
  match task:
    case TaskType.node:
      data = load_node_class_datasets(device, name=FLAGS.dataset, spatial_count=FLAGS.spatial_features_count)
      model_kwargs['input_dim'] = data.num_features
      model_kwargs['output_dim'] = data.num_classes
      model = TestModel(**model_kwargs)
      # model = SimpleGCN(FLAGS.hidden_dim)
      train_losses, val_losses, test_loss, duration = run_node_class_experiment(model, data[0], device)
    case TaskType.graph:
      dataloaders = load_zinc_datasets(device, batch_size=FLAGS.bs, spatial_count=FLAGS.spatial_features_count)
      model_kwargs['input_dim'] = 1
      model_kwargs['output_dim'] = 1
      model = TestModel(**model_kwargs)
      train_losses, val_losses, test_loss, duration = run_graph_class_experiment(model, dataloaders, device)

  np.savez(f'runs/{FLAGS.model_name}_'
           f'{FLAGS.dimension}d_'
           f'{FLAGS.spatial_features_name}_{FLAGS.spatial_features_count}k_'
           f'{FLAGS.seed}',
           params=FLAGS,
           duration=duration / FLAGS.epochs,
           train=train_losses,
           val=val_losses,
           test=test_loss)
  return


if __name__ == '__main__':
  app.run(main)
