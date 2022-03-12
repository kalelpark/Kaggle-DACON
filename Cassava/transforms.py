import albumentations as A
from albumentations.pytorch import ToTensorV2
from setting import setting
config = setting

def train_transform():
    return A.Compose([
        A.RandomResizedCrop(config.sz, config.sz),
        # A.Transpose(p = 0.5),
        A.HorizontalFlip(p = 0.5),
        # A.ShiftScaleRotate(p = 0.5),
        # A.VerticalFlip(p = 0.5),
        A.Normalize(mean = config.Normalize_mean, std = config.Normalize_std,
                     max_pixel_value= config.mpl, p = 1.0),
        ToTensorV2(p = 1)])
    
def test_transform():
    return A.Compose([
        A.Resize(config.sz, config.sz),
        A.Normalize(mean = config.Normalize_mean, std = config.Normalize_std,
                     max_pixel_value= config.mpl, p = 1.0),
        ToTensorV2(p = 1)])