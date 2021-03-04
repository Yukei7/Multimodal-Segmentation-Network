import numpy as np


def split_dataset(ratio: float, size: int):
    """
    Split into two sets
    The ground truth of the validation data is not given, so we manually
    select part of the train data as the test data to test our algorithm.
    Also this function is used to split train set and validation set.
    :param ratio: The ratio of group 1 to the full-size data
    :param size: Size of the full data
    :return: id1, id2
    """
    id1 = np.random.choice(size, size=int(ratio * size), replace=False)
    id2 = np.arange(0, size)
    id2 = np.delete(id2, id1)
    return id1, id2
