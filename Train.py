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


def main(args):
    # CPU/GPU config
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # get number of samples
    df = pd.read_csv(os.path.join(args["input"], "name_mapping.csv"))
    # split train set and test set
    train_idx, test_idx = train_test_split(list(range(df.shape[0])),
                                           test_size=args["testratio"],
                                           random_state=args["seed"],
                                           stratify=df.Grade)
    if args["use3d"]:
        datasets = {x: BratsDataset(folder=args["input"],
                                    modal=args["modal"],
                                    fileidx=y,
                                    phase=x)
                    for (x, y) in [("train", train_idx), ("test", test_idx)]}
    else:
        # TODO: 2d dataset
        datasets = {x: BratsDataset(folder=args["input"],
                                    modal=args["modal"],
                                    fileidx=y,
                                    phase=x)
                    for (x, y) in [("train", train_idx), ("test", test_idx)]}
    n_modals = 4 if args["modal"] == "all" else 1

    # k-fold cross-val
    skf = StratifiedKFold(args["nsplits"])
    fold_counter = 1
    for sub_train_idx, val_idx in skf.split(datasets["train"], datasets["train"].grades):
        time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(f"Fold [{fold_counter}/{args['nsplits']}] Start at {time_str}")
        print(f"Train: {sub_train_idx.shape}, Val:{val_idx.shape}")
        # Net config: utils.get_net
        model = get_net(name=args["net"], n_modals=n_modals, device=device)
        # Loss func
        dice_loss = DiceLoss()
        # optimizer
        optimizer = optim.Adam(model.parameters(), lr=args["lr"])

        # k-fold
        trainset = data.Subset(datasets["train"], sub_train_idx)
        valset = data.Subset(datasets["train"], val_idx)
        train_dataloader = data.DataLoader(dataset=trainset,
                                           batch_size=args["bs"],
                                           shuffle=True,
                                           num_workers=args["num_workers"],
                                           pin_memory=True)
        val_dataloader = data.DataLoader(dataset=valset,
                                         batch_size=args["bs"],
                                         shuffle=False,
                                         num_workers=args["num_workers"],
                                         pin_memory=True)
        # save model for each fold
        model_file_name = os.path.join(args["modelpath"], f"{args['net']}_fold_{fold_counter}.h5")
        train(model=model,
              optimizer=optimizer,
              criterion=dice_loss,
              n_epochs=args["epoch"],
              fold_count=fold_counter,
              training_loader=train_dataloader,
              validation_loader=val_dataloader,
              n_gpus=args["ngpus"],
              training_log_filename=f"{args['net']}_fold{fold_counter}_{args['log']}",
              model_filename=model_file_name)
        fold_counter += 1


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


def train(model, optimizer, criterion, n_epochs, training_loader, validation_loader,
          training_log_filename, fold_count, metric_to_monitor="val_loss", n_gpus=1,
          decay_factor=0.5, lr_decay_step=2, min_lr=1e-7, model_filename=None,
          use_scheduler=True):
    # Train Log
    training_log = list()
    # If csv found, continue from the point where it stops last time
    # TODO: I haven't implement the method to read model from last checkpoint
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
        loss = epoch_train(train_loader=training_loader,
                           model=model,
                           criterion=criterion,
                           optimizer=optimizer,
                           epoch=epoch,
                           n_gpus=n_gpus)
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
        if use_scheduler:
            if validation_loader and scheduler.__class__ == torch.optim.lr_scheduler.ReduceLROnPlateau:
                scheduler.step(val_loss)
            elif scheduler.__class__ == torch.optim.lr_scheduler.ReduceLROnPlateau:
                scheduler.step(loss)
            else:
                scheduler.step()

        # Save the best one
        torch.save(model.state_dict(), model_filename)
        if min_epoch == len(training_log) - 1:
            best_filename = model_filename.replace(".h5", f"_best.h5")
            forced_copy(model_filename, best_filename)


def forced_copy(source, target):
    if os.path.exists(target):
        os.remove(target)
    shutil.copy(source, target)


def get_lr(optimizer):
    lrs = [params['lr'] for params in optimizer.param_groups]
    return np.squeeze(np.unique(lrs))


if __name__ == "__main__":
    args = init()
    main(args)
