import os
import cv2
import torch
from torch.utils.data import Dataset

class mvtecDataset(Dataset):
    def __init__(self, dataframe, root_dir, transforms = None):
        super().__init__()
        self.dataframe = dataframe
        self.root_dir = root_dir
        self.transforms = transforms
    
    def __len__(self):
        return len(self.dataframe)
    
    def get_img(self, path):
        img_bgr = cv2.imread(path)
        img_rgb = img_bgr[:, :, ::-1]
        return img_rgb
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.root_dir,
                                self.dataframe.iloc[idx, 1])
        image = self.get_img(img_name)
        if self.transforms:
            image = self.transforms(image = image)['image']
        csv_row = self.dataframe.iloc[idx, 1:]
        sample = {
            'image' : image,
            'label' : csv_row.label
        }
        return sample

class mvtectestDataset(Dataset):
    def __init__(self, dataframe, root_dir, transforms = None):
        super().__init__()
        self.dataframe = dataframe
        self.root_dir = root_dir
        self.transforms = transforms
    
    def __len__(self):
        return len(self.dataframe)
    
    def get_img(self, path):
        img_bgr = cv2.imread(path)
        img_rgb = img_bgr[:, :, ::-1]
        return img_rgb
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.root_dir,
                                self.dataframe.iloc[idx, 1])
        image = self.get_img(img_name)
        if self.transforms:
            image = self.transforms(image = image)['image']
        sample = {
            'image' : image
        }
        return sample