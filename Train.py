import argparse
import pandas as pd
import numpy as np
import os
from BratsDataset import BratsDataset
from sklearn.model_selection import train_test_split, KFold
from torch.utils import data
import torchvision
import torch
from torchvision.models.segmentation import fcn, deeplabv3

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, default="brats19_h5")
parser.add_argument("--seed", type=int, default=2021)
parser.add_argument("--testratio", type=float, default=0.8)
parser.add_argument("--nsplits", type=int, default=5)
parser.add_argument("--epoch", type=int, default=50)
args = parser.parse_args()


def init():
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def train():
    pass


def validation():
    pass


def main():
    # split train set and test set
    dataset = BratsDataset(args.input)
    train_idx, test_idx = train_test_split(list(range(len(dataset))), test_size=args.testratio)
    datasets = {x: data.Subset(dataset, y)
                for (x, y) in [("train", train_idx), ("test", test_idx)]}
    del dataset

    # k-fold cross-val
    kf = KFold(args.nsplits)
    fold_counter = 1
    for train_idx, val_idx in kf.split(datasets["train"]):
        print(f"Fold [{fold_counter}/{args.nsplits}]:")
        # k-fold
        trainset, valset = data.Subset(datasets["train"], train_idx), data.Subset(datasets["train"], val_idx)
        train_dataloader = data.DataLoader(trainset, 8, shuffle=True, num_workers=4)

        # go through each fold
        for epoch in range(1, args.epoch):
            print(f"Epoch [{epoch}/{args.epoch}]:")
            # Train

            # Val

            # Save model

        fold_counter += 1

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")


if __name__ == "__main__":
    init()
    main()
