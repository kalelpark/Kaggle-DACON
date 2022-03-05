import os
import cv2
import numpy as np
import pandas as pd
import random
import albumentations as A
from albumentations import pytorch
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.modules.module import Module
import torch.nn.functional as F
from preprocess import Data_preprocess

# CustomDataLoader
class CustomDataset(Dataset):
    def __init__(self, root_dir, df, label_to_id, transform):
        self.root_dir = root_dir
        self.df = df
        self.label_to_id = label_to_id

        
        if 'train' in root_dir:
            self.transform = transform['train_aug']
        else:
            self.transform = transform['val_test_aug']

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        image_path = os.path.join(self.root_dir, self.df.iloc[index, 0])
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if 'train' in self.root_dir:
            label = self.df.iloc[index, 2]
            target = self.label_to_id[label]
            image = self.transform(image = image)
            image = image['image']
            return image, torch.tensor(target)

        else:
            image = self.transform(image = image)
            image = image['image']
            image = image.unsqeeze(0)

            return image

def onehot(size, target):
    vec = torch.zeros(size, dtype=torch.float32)
    vec[target] = 1.
    return vec


def rand_bbox(size, lam):
    if len(size) == 4:
        W = size[2]
        H = size[3]
    elif len(size) == 3:
        W = size[1]
        H = size[2]
    else:
        raise Exception

    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


class CutMix(Dataset):
    def __init__(self, dataset, num_class, num_mix=1, beta=1., prob=1.0):
        self.dataset = dataset
        self.num_class = num_class
        self.num_mix = num_mix
        self.beta = beta
        self.prob = prob

    def __getitem__(self, index):
        img, lb = self.dataset[index]
        lb_onehot = onehot(self.num_class, lb)

        for _ in range(self.num_mix):
            r = np.random.rand(1)
            if self.beta <= 0 or r > self.prob:
                continue

            # generate mixed sample
            lam = np.random.beta(self.beta, self.beta)
            rand_index = random.choice(range(len(self)))

            img2, lb2 = self.dataset[rand_index]
            lb2_onehot = onehot(self.num_class, lb2)

            bbx1, bby1, bbx2, bby2 = rand_bbox(img.size(), lam)
            img[:, bbx1:bbx2, bby1:bby2] = img2[:, bbx1:bbx2, bby1:bby2]
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (img.size()[-1] * img.size()[-2]))
            lb_onehot = lb_onehot * lam + lb2_onehot * (1. - lam)

        return img, lb_onehot

    def __len__(self):
        return len(self.dataset)

# Image Trnasformer
def data_transforms():
    transform = {
        'train_aug' : A.Compose([
            A.HorizontalFlip(p = 0.5),
            A.ImageCompression(quality_lower=99, quality_upper=100),
            A.ShiftScaleRotate(shift_limit = 0.2, scale_limit = 0.2, rotate_limit = 10, border_mode = 0, p = 0.7),
            A.Resize(224, 224),
            A.Normalize(),
            pytorch.ToTensorV2()
        ]),
        'val_test_aug' : A.Compose([
            A.Resize(224, 224),
            A.Normalize(),
            pytorch.ToTensorV2()
        ])
    } 
    return transform
 
def all_custom_setting():
    train_df, test_df, target, label_encoder, label_to_id, id_to_label = Data_preprocess()
    root_dir = 'images\\train_images'
    root_test_dir = 'images\\test_images'

    transform = data_transforms()
    
    trainset = CustomDataset(root_dir, train_df, label_to_id, transform)
    trainset = CutMix(trainset, num_class = len(label_to_id), num_mix = 2, beta = 1.0, prob = 0.7)
    validset = CustomDataset(root_dir, train_df, label_to_id, transform) 
    testset = CustomDataset(root_test_dir, test_df, label_to_id, transform)

    return trainset, validset, testset