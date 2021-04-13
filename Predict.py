import json
import numpy as np
import os
import shutil
import warnings
import time
from BratsDataset import BratsDataset
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from torch.utils import data
import torch.optim as optim
import torch
import pandas as pd
from loss import DiceLoss
from utils import epoch_train, epoch_validation, get_net


def init():
    # read config from json file
    with open("config.json", "r") as f:
        args = json.load(f)
    # seed everything
    np.random.seed(args["seed"])
    torch.manual_seed(args["seed"])
    # create checkpoint directory if not exists
    if not os.path.exists(args["modelpath"]):
        os.mkdir(args["modelpath"])
    return args


def predict(args):
    # CPU/GPU config
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")


if __name__ == "__main__":
    args = init()
    predict(args)
