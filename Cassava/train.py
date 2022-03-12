import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from Data_Loader import CassavaDataset
from  model import CassavaNet, SnapMixLosss
from metric import accuracy_metric, print_kaggle_scores
from Snap_Mix.SnapMix import snapmix
from setting import setting, checkpoint
from transforms import train_transform, test_transform
import warnings
warnings.filterwarnings("ignore")
config = setting

def defined_folds(df):
    folds = StratifiedKFold(n_splits= config.NUM_FOLDS, 
                            shuffle= True, random_state= config.SEED).split(np.arange(df.shape[0]), df['label'].values)
    return folds


def train_valid(train_df):
    folds = defined_folds(train_df)

    for fold_num, (train_split, valid_split) in enumerate(folds):
        train_set = train_df.iloc[train_split].reset_index(drop = True)
        valid_set = train_df.iloc[valid_split].reset_index(drop = True)

        train_trans = train_transform()
        valid_trans = test_transform()
        
        train_ds = CassavaDataset(dataframe = train_set, root_dir='train_images', 
                                transforms = train_trans)
        valid_ds = CassavaDataset(dataframe= valid_set, root_dir = 'train_images',
                                  transforms = valid_trans)

        train_dl = DataLoader(train_ds, batch_size=config.bs, shuffle= True, 
                              drop_last= True, pin_memory= True)
        valid_dl = DataLoader(valid_ds, batch_size=config.bs, shuffle= True,
                              drop_last= True, pin_memory= True)

        batches = len(train_dl)
        val_batches = len(valid_dl)
        best_metric = 0

        model = CassavaNet().to(config.DEVICE)
        criterion = nn.CrossEntropyLoss(reduction = 'none').to(config.DEVICE)
        val_criterion = nn.CrossEntropyLoss().to(config.DEVICE)
        snap_mix_criterion = SnapMixLosss().to(config.DEVICE)

        param_groups = [
            {'params' : model.backbone.parameters(), 'lr' : 1e-2},
            {'params' : model.classifier.parameters()}
        ]
        optimizer = optim.SGD(param_groups, lr = 1e-1, momentum= 0.9,
                              weight_decay= 1e-4, nesterov= True)
        
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1, 20, 40], gamma= 0.1,
                                                   last_epoch = -1, verbose = True)
        scaler = GradScaler()

        for epoch in range(config.EPOCHS):
            # --------------- Training  ---------------
            train_loss = 0
            progress = tqdm(enumerate(train_dl), desc = "Loss : ", total = batches)
            
            model.train()
            for i, data in progress:
                image, label = data.values()
                X, y = image.to(config.DEVICE).float(), label.to(config.DEVICE).long()

                with autocast():

                    rand = np.random.rand()
                    if rand > (1.0 - config.SNAPMIX_PCT):
                        X, ya, yb, lam_a, lam_b = snapmix(X, y, config.SNAPMIX_ALPHA, model)
                        outputs, _ = model(X)
                        loss = snap_mix_criterion(criterion, outputs, ya, yb, lam_a, lam_b)
                    else:
                        outputs, _ = model(X)
                        loss = torch.mean(criterion(outputs, y))

                scaler.scale(loss).backward()

                if ((i + 1) % config.GRAD_ACCM_STEPS == 0) or ((i + 1) == len(train_dl)):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            
                train_loss += loss.item()
                cur_step = i+1
                trn_epoch_result = dict()
                trn_epoch_result['Epoch'] = epoch + 1
                trn_epoch_result['Train_loss'] = round(train_loss / cur_step, 4)

                progress.set_description(str(trn_epoch_result))

            scheduler.step()                                    
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
            # ----------------- Validation Loss ----------------------
            val_loss = 0
            scores = []

            model.eval()
            with torch.no_grad():
                for i, data in enumerate(valid_dl):
                    image, label = data.values()
                    X, y = image.to(config.DEVICE), label.to(config.DEVICE)
                    outputs, _ = model(X)
                    l = val_criterion(outputs, y)
                    val_loss += l.item()    

                    preds = F.softmax(outputs).argmax(axis = 1)
                    scores.append(accuracy_metric(preds, y))
            
            epoch_result = dict()
            epoch_result['Epoch'] = epoch + 1
            epoch_result['train_loss'] = round(train_loss / batches, 4)
            epoch_result['val_loss'] = round(val_loss / val_batches, 4)

            print(epoch_result)
            
            current_metric = print_kaggle_scores(scores)

            if current_metric > best_metric:

                checkpoint(model, optimizer, epoch + 1 , current_metric, best_metric, fold_num)
                best_metric = current_metric
        
        del model, optimizer, train_dl, valid_dl, scaler, scheduler
        torch.cuda.empty_cache()

def start_train():
    train_df = pd.read_csv(config.train_dir)
    train_valid(train_df)

start_train()
