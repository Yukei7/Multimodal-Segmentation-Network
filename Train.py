import argparse
import pandas as pd
import numpy as np
import os
import shutil
import warnings
from BratsDataset import BratsDataset
from sklearn.model_selection import train_test_split, KFold
from torch.utils import data
import torch.optim as optim
import torch
import pandas as pd
from torchvision.models.segmentation import fcn, deeplabv3
import torchvision

from net import unet
from loss import DiceLoss
from utils import epoch_train, epoch_validation

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, default="brats19_h5")
parser.add_argument("--seed", type=int, default=2021)
parser.add_argument("--testratio", type=float, default=0.8)
parser.add_argument("--nsplits", type=int, default=5)
parser.add_argument("--epoch", type=int, default=50)
parser.add_argument("--net", type=str, default="unet")
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--ngpus", type=int, default=1)
parser.add_argument("--log", type=str, default="log.csv")
parser.add_argument("--modelpath", type=str, default="checkpoint")
args = parser.parse_args()


def main():
    # CPU/GPU config
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # split train set and test set
    dataset = BratsDataset(args.input, modal='all')
    n_modals = dataset.n_modals
    train_idx, test_idx = train_test_split(list(range(len(dataset))), test_size=args.testratio)
    datasets = {x: data.Subset(dataset, y)
                for (x, y) in [("train", train_idx), ("test", test_idx)]}
    del dataset

    # Net config
    if args.net == "unet":
        model = unet.UNet(in_channels=n_modals, out_channels=4)
    else:
        raise NotImplementedError("Net unfounded! Please check the input name and the net file.")

    # Loss func
    dice_loss = DiceLoss()

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # k-fold cross-val
    kf = KFold(args.nsplits)
    fold_counter = 1
    for train_idx, val_idx in kf.split(datasets["train"]):
        print(f"Fold [{fold_counter}/{args.nsplits}]:")
        # k-fold
        trainset, valset = data.Subset(datasets["train"], train_idx), data.Subset(datasets["train"], val_idx)
        train_dataloader = data.DataLoader(trainset, 8, shuffle=True, num_workers=4)
        val_dataloader = data.DataLoader(valset, 8, shuffle=False, num_workers=4)
        model_file_name = os.path.join(args.modelpath, f"{args.net}_fold_{fold_counter}.h5")
        train(model=model,
              optimizer=optimizer,
              criterion=dice_loss,
              n_epochs=args.epoch,
              training_loader=train_dataloader,
              validation_loader=val_dataloader,
              n_gpus=args.ngpus,
              training_log_filename=f"{args.net}_{args.log}",
              model_filename=model_file_name)
        fold_counter += 1


def init():
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not os.path.exists(args.modelpath):
        os.mkdir(args.modelpath)


def train(model, optimizer, criterion, n_epochs, training_loader, validation_loader,
          training_log_filename, metric_to_monitor="val_loss", n_gpus=1, decay_factor=0.1,
          lr_decay_step=3, min_lr=1e-8, model_filename=None):
    # Train Log
    training_log = list()
    if os.path.exists(training_log_filename):
        training_log.extend(pd.read_csv(training_log_filename).values)
        start_epoch = int(training_log[-1][0]) + 1
    else:
        start_epoch = 0
    training_log_header = ["epoch", "loss", "lr", "val_loss"]

    # Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                           patience=lr_decay_step,
                                                           factor=decay_factor,
                                                           min_lr=min_lr)

    # Go through each epoch
    for epoch in range(start_epoch, n_epochs):
        loss = epoch_train(training_loader, model, criterion, optimizer, epoch=epoch, n_gpus=n_gpus)

        try:
            training_loader.dataset.on_epoch_end()
        except AttributeError:
            warnings.warn("'on_epoch_end' method not implemented for the {} dataset.".format(
                type(training_loader.dataset)))

        # validate the model
        if validation_loader:
            val_loss = epoch_validation(validation_loader, model, criterion, n_gpus=n_gpus)
        else:
            val_loss = None

        # update logger
        training_log.append([epoch, loss, get_lr(optimizer), val_loss])
        pd.DataFrame(training_log, columns=training_log_header).set_index("epoch").to_csv(training_log_filename)
        min_epoch = np.asarray(training_log)[:, training_log_header.index(metric_to_monitor)].argmin()

        # Scheduler: adjust lr
        if scheduler:
            if validation_loader and scheduler.__class__ == torch.optim.lr_scheduler.ReduceLROnPlateau:
                scheduler.step(val_loss)
            elif scheduler.__class__ == torch.optim.lr_scheduler.ReduceLROnPlateau:
                scheduler.step(loss)
            else:
                scheduler.step()

        # Save the best one
        torch.save(model.state_dict(), model_filename)
        if min_epoch == len(training_log) - 1:
            best_filename = model_filename.replace(".h5", f"_epoch_{epoch}.h5")
            forced_copy(model_filename, best_filename)


def forced_copy(source, target):
    if os.path.exists(target):
        os.remove(target)
    shutil.copy(source, target)


def get_lr(optimizer):
    lrs = [params['lr'] for params in optimizer.param_groups]
    return np.squeeze(np.unique(lrs))


if __name__ == "__main__":
    init()
    main()
