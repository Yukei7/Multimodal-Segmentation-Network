import albumentations as A
from albumentations import Compose, HorizontalFlip
from albumentations.pytorch import ToTensor, ToTensorV2
from utils import calculate_stats


def get_augmentations(phase, files):
    if phase != "train":
        return None

    # read all files and get the statistics
    # TODO: FINISH

    list_transforms = []
    list_trfms = Compose(list_transforms)
    return list_trfms


def ds_stat(files):
    config = {}
    return config
