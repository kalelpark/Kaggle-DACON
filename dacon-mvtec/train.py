import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from setting import config , label_to_unique
import warnings
from pre_model import MvtecNet
from sklearn.model_selection import StratifiedKFold
from transforms import train_transform, test_transform
from Data_Loader import mvtecDataset
from cs_metric import f1_function
warnings.filterwarnings('ignore')

def defined_folds(df):
    folds = StratifiedKFold(n_splits = config.NUM_FOLDS, shuffle = True, 
                            random_state=config.RANDON_STATE).split(np.arange(df.shape[0]), df['label'].values)
    return folds

def train_valid(train_df, label_dict):
    folds = defined_folds(train_df)

    for fold_num, (train_split, valid_split) in enumerate(folds):
        train_set = train_df.iloc[train_split].reset_index(drop = True)
        valid_set = train_df.iloc[valid_split].reset_index(drop = True)
        
        train_trans = train_transform()
        valid_trans = test_transform()

        train_ds = mvtecDataset(dataframe = train_set, root_dir = 'open/train', transforms = train_trans)
        valid_ds = mvtecDataset(dataframe = valid_set, root_dir = 'open/train', transforms = valid_trans)

        train_dl = DataLoader(train_ds, batch_size = config.BATCH_SIZE, shuffle= True, drop_last= True, pin_memory= True)
        valid_dl = DataLoader(valid_ds, batch_size = config.BATCH_SIZE, shuffle= True, drop_last= True, pin_memory= True)

        batches = len(train_dl)
        model = MvtecNet().to(config.DEVICE)
        
        optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
        criterion = nn.CrossEntropyLoss()
        scaler = torch.cuda.amp.GradScaler()
        
        for epoch in range(config.EPOCHS):
            train_loss = 0
            train_pred = []
            train_y = []
            train_correct = 0
            train_count = 0
            for data in train_dl:
                optimizer.zero_grad()
                train_da, train_tr = data['image'].to(config.DEVICE), data['label'].to(config.DEVICE)
                with torch.cuda.amp.autocast():
                    pred = model(train_da)
                
                loss = criterion(pred, train_tr)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                train_count += len(train_tr)
                train_correct += (pred.argmax(1) == train_tr).type(torch.float).sum().item()
                train_loss += loss.item() / batches
                train_pred += pred.argmax(1).detach().cpu().numpy().tolist()
                train_y += train_tr.detach().cpu().numpy().tolist()
            train_correct /= train_count
            train_f1 = f1_function(train_y, train_pred)
            print(f'Fold_NUM : {fold_num+1} EPOCH {epoch+1}/{config.EPOCHS}')
            print(f'Train Loss : {train_loss} Train F1 : {train_f1}')
            print(f'Train Accuracy : {train_correct*100}% / 100%')

        # Validation Setting 

        valid_pred = []
        valid_y = []
        model.eval()
        valid_correct = 0
        valid_count = 0
        with torch.no_grad():
            for data in valid_dl:
                valid_da, valid_tr = data['image'].to(config.DEVICE), data['label'].to(config.DEVICE)
                pred = model(valid_da)
                valid_correct += (pred.argmax(1) == valid_tr).type(torch.float).sum().item()
                valid_count += len(valid_tr)
                valid_pred += pred.argmax(1).detach().cpu().numpy().tolist()
                valid_y += valid_tr.detach().cpu().numpy().tolist()    
        valid_correct /= valid_count
        valid_f1 = f1_function(valid_y, valid_pred)
        print(f'Valid_{fold_num+1} :   Valid F1 Score : {valid_f1}')
        print(f'Valid Accuracy {valid_correct * 100}% / 100%')
        torch.save(model, f'torch_model_effic3_{fold_num+1}.pth')
        torch.save(model.state_dict(), f'torch_model_effic3_state_dict_{fold_num+1}.pth')

def start_train():
    train_df = pd.read_csv(config.train_dir)
    label_dict, reverse_dict = label_to_unique(train_df)
    train_valid(train_df, label_dict)

start_train()