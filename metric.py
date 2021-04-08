import numpy as np
import torch


def dice_coef(y_pred: torch.Tensor,
              y_true: torch.Tensor,
              ts: float = 0.5,
              eps: float = 1e-9):
    assert y_pred.shape == y_true.shape
    scores = []
    n_samples = y_pred.shape[0]
    preds = (y_pred > ts).float()
    for i in range(n_samples):
        pred = preds[i]
        true = y_true[i]
        intersect = 2.0 * (true * pred).sum()
        union = true.sum() + pred.sum()
        if true.sum() == 0 and pred.sum() == 0:
            scores.append(1.0)
        else:
            scores.append((intersect + eps) / union)
    return np.mean(scores)


def jaccard_coef(y_pred: torch.Tensor,
                 y_true: torch.Tensor,
                 ts: float = 0.5,
                 eps: float = 1e-9):
    assert y_pred.shape == y_true.shape
    scores = []
    n_samples = y_pred.shape[0]
    preds = (y_pred > ts).float()
    for i in range(n_samples):
        pred = preds[i]
        true = y_true[i]
        intersect = (pred * true).sum()
        union = (true.sum() + pred.sum()) - intersect + eps
        if true.sum() == 0 and pred.sum() == 0:
            scores.append(1.0)
        else:
            scores.append((intersect + eps) / union)
    return np.mean(scores)


def dice_coef_per_classes(probabilities: np.ndarray,
                          truth: np.ndarray,
                          treshold: float = 0.5,
                          eps: float = 1e-9,
                          classes: list = ['WT', 'TC', 'ET']) -> dict:
    scores = {key: list() for key in classes}
    num = probabilities.shape[0]
    num_classes = probabilities.shape[1]
    predictions = (probabilities >= treshold).astype(np.float32)
    assert (predictions.shape == truth.shape)

    for i in range(num):
        for class_ in range(num_classes):
            prediction = predictions[i][class_]
            truth_ = truth[i][class_]
            intersection = 2.0 * (truth_ * prediction).sum()
            union = truth_.sum() + prediction.sum()
            if truth_.sum() == 0 and prediction.sum() == 0:
                scores[classes[class_]].append(1.0)
            else:
                scores[classes[class_]].append((intersection + eps) / union)
    return scores
