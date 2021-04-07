import torch
import torch.nn as nn
from torch.autograd import Variable

from collections import OrderedDict
import numpy as np
import time
from meters import AverageMeter, ProgressMeter


# https://github.com/ellisdg/3DUnetCNN
def epoch_train(train_loader, model, criterion, optimizer, epoch, n_gpus=None, print_freq=1):
    # Log loading time and loss for each epoch
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    progress = ProgressMeter(len(train_loader),
                             [batch_time, data_time, losses],
                             prefix=f"Epoch: [{epoch + 1}]")
    model.train()
    end = time.time()

    for idx, patient in enumerate(train_loader):
        im = patient["image"]
        label = patient["mask"]
        data_time.update(time.time() - end)
        if n_gpus:
            torch.cuda.empty_cache()

        optimizer.zero_grad()
        loss, batch_size = batch_loss(model=model,
                                      im=im,
                                      label=label,
                                      criterion=criterion,
                                      n_gpus=n_gpus)
        if n_gpus:
            torch.cuda.empty_cache()

        losses.update(loss.item(), batch_size)

        loss.backward()
        optimizer.step()

        del loss

        batch_time.update(time.time() - end)
        end = time.time()

        if idx % print_freq == 0:
            progress.display(idx)
    return losses.avg


def batch_loss(model, im, label, criterion, n_gpus=0):
    # TODO: type?
    im = im.type(torch.FloatTensor)
    label = label.type(torch.FloatTensor)
    if n_gpus is not None:
        im = im.cuda()
        label = label.cuda()
    output = model(im)
    batch_size = im.shape[0]
    loss = criterion(output, label)
    return loss, batch_size


def epoch_validation(val_loader, model, criterion, n_gpus=None, print_freq=1):
    losses = AverageMeter("Loss", ":.4e")
    batch_time = AverageMeter("Time", ":6.3f")
    progress = ProgressMeter(num_batches=len(val_loader),
                             meters=[batch_time, losses],
                             prefix="Validation: ")

    model.eval()
    with torch.no_grad():
        end = time.time()
        for idx, patient in enumerate(val_loader):
            im = patient["image"]
            label = patient["mask"]
            loss, batch_size = batch_loss(model, im, label, criterion, n_gpus)
            losses.update(loss.item(), batch_size)
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % print_freq == 0:
                progress.display(idx)
    return losses.avg


# https://github.com/sksq96/pytorch-summary

def summary(model, input_size, batch_size=-1, device=torch.device('cuda:0'), dtypes=None):
    result, params_info = summary_string(
        model, input_size, batch_size, device, dtypes)
    print(result)

    return params_info


def summary_string(model, input_size, batch_size=-1, device=torch.device('cuda:0'), dtypes=None):
    if dtypes == None:
        dtypes = [torch.FloatTensor] * len(input_size)

    summary_str = ''

    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        if (
                not isinstance(module, nn.Sequential)
                and not isinstance(module, nn.ModuleList)
        ):
            hooks.append(module.register_forward_hook(hook))

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    x = [torch.rand(2, *in_size).type(dtype).to(device=device)
         for in_size, dtype in zip(input_size, dtypes)]

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    # print(x.shape)
    model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()

    summary_str += "----------------------------------------------------------------" + "\n"
    line_new = "{:>20}  {:>25} {:>15}".format(
        "Layer (type)", "Output Shape", "Param #")
    summary_str += line_new + "\n"
    summary_str += "================================================================" + "\n"
    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        line_new = "{:>20}  {:>25} {:>15}".format(
            layer,
            str(summary[layer]["output_shape"]),
            "{0:,}".format(summary[layer]["nb_params"]),
        )
        total_params += summary[layer]["nb_params"]

        total_output += np.prod(summary[layer]["output_shape"])
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"]:
                trainable_params += summary[layer]["nb_params"]
        summary_str += line_new + "\n"

    # assume 4 bytes/number (float on cuda).
    total_input_size = abs(np.prod(sum(input_size, ()))
                           * batch_size * 4. / (1024 ** 2.))
    total_output_size = abs(2. * total_output * 4. /
                            (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params * 4. / (1024 ** 2.))
    total_size = total_params_size + total_output_size + total_input_size

    summary_str += "================================================================" + "\n"
    summary_str += "Total params: {0:,}".format(total_params) + "\n"
    summary_str += "Trainable params: {0:,}".format(trainable_params) + "\n"
    summary_str += "Non-trainable params: {0:,}".format(total_params -
                                                        trainable_params) + "\n"
    summary_str += "----------------------------------------------------------------" + "\n"
    summary_str += "Input size (MB): %0.2f" % total_input_size + "\n"
    summary_str += "Forward/backward pass size (MB): %0.2f" % total_output_size + "\n"
    summary_str += "Params size (MB): %0.2f" % total_params_size + "\n"
    summary_str += "Estimated Total Size (MB): %0.2f" % total_size + "\n"
    summary_str += "----------------------------------------------------------------" + "\n"
    # return summary
    return summary_str, (total_params, trainable_params)


def calculate_stats(images):
    """
    Calculates min, max, mean, std given a list of ndarrays
    """
    # flatten first since the images might not be the same size
    flat = np.concatenate(
        [img.ravel() for img in images]
    )
    return np.min(flat), np.max(flat), np.mean(flat), np.std(flat)


def minmax_normalize(img_npy):
    '''
    img_npy: ndarray
    '''
    min_value = np.min(img_npy)
    max_value = np.max(img_npy)
    return (img_npy - min_value) / (max_value - min_value)
