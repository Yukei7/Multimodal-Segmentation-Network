import argparse
import pandas as pd
import numpy as np
import os
from BratsDataset import BratsDataset
from torch.utils import data
from utils import split_dataset
import torchvision
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, default="brats19_h5")
parser.add_argument("--seed", type=int, default=2021)
args = parser.parse_args()


def init():
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def main():
    train_id, test_id = split_dataset(0.8, len(os.listdir(args.input)))
    # TODO: Cross-val
    t_id, val_id = split_dataset(0.8, len(train_id))
    datasets = {x: BratsDataset(args.input, y)
                for (x, y) in [('train', t_id), ('val', val_id)]}
    dataloaders = {x: data.DataLoader(datasets[x], batch_size=4, shuffle=True, num_workers=4)
                   for x in ['train', 'val']}

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")


if __name__ == "__main__":
    np.random.seed(args.seed)
    split_dataset()
