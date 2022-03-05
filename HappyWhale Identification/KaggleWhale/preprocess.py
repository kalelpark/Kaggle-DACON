import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import albumentations as A
from torch.nn.modules.module import Module
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import torchvision.transforms as transforms


def Replace_Duplicate_Features(data, species):
    for duplicate, change in species:
        data['species'] = data['species'].str.replace(duplicate, change)
    return data

def prepare_labeling(data):
    values = np.array(data)
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    onehot_encoder = OneHotEncoder(sparse = False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    return onehot_encoded, label_encoder

def dict_labeling_ID(data):
    Label_to_id = {}
    id_to_Label = {}
    idx = 0
    for label in data:
        Label_to_id[label] = idx
        id_to_Label[idx] = label
        idx += 1
    return Label_to_id, id_to_Label

def Data_preprocess():
    train_df = pd.read_csv('csv_file/train.csv')
    test_df = pd.read_csv('csv_file/sample_submission.csv')
    dup_species = [['bottlenose_dolpin','bottlenose_dolphin'],
                   ['kiler_whale','killer_whale']]
    train_df = Replace_Duplicate_Features(train_df, dup_species)
    target, label_encoder = prepare_labeling(train_df['individual_id'])
    label_to_id , id_to_label = dict_labeling_ID(train_df['individual_id'].unique())

    return train_df, test_df, target, label_encoder, label_to_id, id_to_label 


