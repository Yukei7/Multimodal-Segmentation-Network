from volumentations.core.composition import Compose
import volumentations.augmentations.transforms as T


def get_augmentations3d(phase, patch_size):
    if phase != "train":
        return None
    list_transforms = [T.Resize(patch_size, always_apply=True),
                       T.ElasticTransform((0, 0.25)),
                       T.Rotate((-15, 15), (-15, 15), (-15, 15)),
                       T.Flip(0),
                       T.Flip(1),
                       T.Flip(2),
                       T.RandomBrightness(factor=0.2)]
    list_trfms = Compose(list_transforms)
    return list_trfms
