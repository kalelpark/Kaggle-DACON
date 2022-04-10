import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from Data_Loader import mvtectestDataset
from setting import label_to_unique
from setting import config
from transforms import test_transform
from pre_model import MvtecNet

def label_to_key():
    train_df = pd.read_csv(config.train_dir)
    df, label_unique = label_to_unique(train_df)
    reverse_dict = dict(map(reversed, label_unique.items()))
    return reverse_dict

def load_torch_record():
    record_list = os.listdir('effei_b3')
    return record_list

def stacking_predict():
    record_list = load_torch_record()
    test_df = pd.read_csv(config.test_dir)
    submit_data = []

    for record in record_list:
        model = torch.load(os.path.join('effei_b3',record))
        test_trans = test_transform()

        test_ds = mvtectestDataset(dataframe = test_df, root_dir = 'open/test', transforms= test_trans)
        test_dl = DataLoader(test_ds, batch_size= config.BATCH_SIZE, shuffle = True)

        model.eval()
        stacking_predict = []
        with torch.no_grad():
            for data in test_dl:
                image = data['image'].to(config.DEVICE)
                outputs = model(image)
                stacking_predict += outputs.cpu()
        
        submit_data.append(stacking_predict)
    
    prep = torch.zeros(2154, 88)
    for submit in submit_data:
        for jp, en in enumerate(submit):
            prep[jp] += en
    
    prep = prep / len(submit_data)
    _, predicted = torch.max(prep, 1)

    submission = pd.DataFrame({
        "index" : np.arange(2154),
        "label" : predicted
    })

    reverse_dict = label_to_key()
    submission['label'] = submission['label'].map(reverse_dict)
    submission.to_csv('submit.csv', index = False)

stacking_predict()