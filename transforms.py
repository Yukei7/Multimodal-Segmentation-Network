# from volumentations import Compose, Rotate, RandomCropFromBorders, ElasticTransform
# from volumentations import Resize, Flip, RandomRotate90, GaussianNoise, RandomGamma


def get_augmentations3d(phase, patch_size):
    return None
    # if phase != "train":
    #     return None
    # return Compose([
    #     # Rotate((-15, 15), (-15, 15), (-15, 15), p=0.5),
    #     RandomCropFromBorders(crop_value=0.1, p=0.5),
    #     # ElasticTransform((0, 0.25), interpolation=2, p=0.1),
    #     Resize(patch_size, interpolation=1, always_apply=True, p=1.0),
    #     Flip(0, p=0.5),
    #     Flip(1, p=0.5),
    #     Flip(2, p=0.5),
    #     # RandomRotate90((1, 2), p=0.5),
    #     # GaussianNoise(var_limit=(0, 2), p=0.2),
    #     # RandomGamma(gamma_limit=(0.7, 1.3), p=0.2),
    # ], p=1.0)

# def get_augmentations3d(phase, patch_size):
#     if phase != "train":
#         return None
#     list_transforms = [T.Resize(patch_size, always_apply=True),
#                        T.ElasticTransform((0, 0.25)),
#                        T.Rotate((-15, 15), (-15, 15), (-15, 15)),
#                        T.Flip(0),
#                        T.Flip(1),
#                        T.Flip(2),
#                        T.RandomBrightness(factor=0.2)]
#     list_trfms = Compose(list_transforms)
#     return list_trfms
