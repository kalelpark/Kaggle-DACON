import albumentations as A
from albumentations.pytorch import ToTensorV2
from setting import config


def train_transform():
    return A.Compose([
        A.Resize(config.IMAGE_SIZE, config.IMAGE_SIZE),
        A.HueSaturationValue(p = 0.8),
        A.Normalize(mean= config.MEAN_NORMAL,
                    std = config.STD_NORMAL),
        ToTensorV2(p = 1),
    ])


def test_transform():
    return A.Compose([
        A.Resize(config.IMAGE_SIZE, config.IMAGE_SIZE),
        A.Normalize(mean = config.MEAN_NORMAL,
                    std = config.STD_NORMAL),
        ToTensorV2(p = 1),
    ])