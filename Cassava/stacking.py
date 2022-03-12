from inspect import stack
import os
from sklearn import ensemble
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from Data_Loader import CassavaDataset
from  model import CassavaNet, SnapMixLosss
from metric import accuracy_metric, print_kaggle_scores, accuracy_metric_cpu
from Snap_Mix.SnapMix import snapmix
from setting import setting, checkpoint
from transforms import train_transform, test_transform
from model import CassavaNet
config = setting

# 21397, 2(5개의 라벨 존재)
def load_torch_record():
    record_list = os.listdir('torch_records')
    return record_list

def stacking_predict():
    record_list = load_torch_record()
    predict_df = pd.read_csv(config.train_dir)
    stacking_ensemble_data = []
    for record in record_list:
        model = CassavaNet()
        print(record)
        model.load_state_dict(torch.load('torch_records/'+record)['state_dict'])
        model.to(config.DEVICE)
        test_trans = test_transform()
        predict_set = CassavaDataset(dataframe=predict_df, root_dir = 'train_images',
                                    transforms=test_trans)
        predict_dl = DataLoader(predict_set, batch_size=config.bs, shuffle = True,
                                 drop_last= True, pin_memory= True)
        
        model.eval()
        val_criterion = nn.CrossEntropyLoss().to(config.DEVICE)
        stacking_predict = []
        with torch.no_grad():
            for data in predict_dl:
                image, label = data.values()
                image, label = image.to(config.DEVICE).float(), label.to(config.DEVICE).long()
                outputs, _ = model(image)
                stacking_predict += F.softmax(outputs).cpu()
                
        stacking_ensemble_data.append(stacking_predict)

    stacking_submit = torch.zeros(21397, 5)
    for ensem in stacking_ensemble_data:
        for cn, en in enumerate(ensem):
            stacking_submit[cn] += en
    
    stacking_submit = stacking_submit / len(stacking_ensemble_data)
    _, predict = torch.max(stacking_submit, 1)
    scores = accuracy_metric_cpu(predict, predict_df['label'])
    print_kaggle_scores(scores)
            
stacking_predict()