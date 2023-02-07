
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data.data import Data
from torch_geometric.datasets.zinc import ZINC
from torch_geometric.datasets.planetoid import Planetoid

import torch.nn.functional as F
import torch_geometric.transforms as T
import numpy as np
import random

from absl import app
from absl import flags
from absl import logging

from networks.gcn_layer import GCNLayer
from networks.test_net import TestNet

FLAGS = flags.FLAGS

flags.DEFINE_float('lr', 0.001, 'Learning rate')
flags.DEFINE_integer('epochs', 50, 'Epochs')
flags.DEFINE_integer('seed', 42, 'Random seed')
flags.DEFINE_integer('bs', 64, 'Batch size')


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def train(model, optimizer, data: Data):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.mse_loss(out, data.y)
    loss.backward()
    optimizer.step()
    del out


def test(model, data):
    model.eval()
    with torch.no_grad():
        out, accs, losses, preds = model(data), [], [], []
        loss = F.mse_loss(out, data.y)
        return loss


def main(unused_argv):
    set_seed(FLAGS.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = ZINC(root='data/zinc', split='train', subset=True)
    val_dataset = ZINC(root='data/zinc', split='val', subset=True)

    model = TestNet(10, 1)
    model.to(device)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=FLAGS.lr)

    for epoch in range(FLAGS.epochs):

        logging.info(f"Epoch {epoch}")

        train_loader = DataLoader(dataset, batch_size=256, shuffle=True)

        for batch in train_loader:
            batch = batch.to(device)
            train(model, optimizer, batch)


        losses = []
        val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
        for batch in val_loader:
            batch = batch.to(device)
            losses += [test(model, batch)]
        logging.info(f"Validation loss: {torch.mean(torch.Tensor(losses))}")

    return


if __name__ == '__main__':
  app.run(main)
