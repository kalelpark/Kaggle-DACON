import os
import random
import torch
import numpy as np
import pandas as pd

class config:
    train_dir = 'open/train_df.csv'
    sample_dir = 'open/sample_submission.csv'
    test_dir = 'open/test_df.csv'
    SEED = None

    # SETTING
    RANDON_STATE = 1912
    NUM_FOLDS = 5
    BATCH_SIZE = 32
    EPOCHS = 50
    IMAGE_SIZE = 300

    # Albumentation Normalize
    MEAN_NORMAL = [0.485, 0.456, 0.406]
    STD_NORMAL = [0.229, 0.224, 0.225]

    pre_trained_model = 'efficientnet_b3'
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def label_to_unique(df):
    labels = df['label']
    label_unique = sorted(np.unique(labels))
    label_unique = {key : value for key, value in zip(label_unique, range(len(label_unique)))}
    train_labels = [label_unique[k] for k in labels]
    df['label'] = train_labels

    return df, label_unique