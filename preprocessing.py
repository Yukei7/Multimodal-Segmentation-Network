import argparse
import pandas as pd
import numpy as np
from BratsDataset import BratsDataset
import torch
from torch.utils import data

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, default="brats19_h5")
parser.add_argument("--seed", type=int, default=2021)
args = parser.parse_args()


def init():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")


def main():
    pass


if __name__ == "__main__":
    np.random.seed(args.seed)
    main()
