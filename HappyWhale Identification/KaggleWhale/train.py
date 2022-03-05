from sched import scheduler
from custom_data import all_custom_setting
from preprocess import Data_preprocess
from model import efficient_model
from model import CutMixCrossEntropyLoss
from sklearn.model_selection import KFold
import torch
import torch.nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, SubsetRandomSampler

def train(fold_num = 5, seed = 128, epochs = 100, lr = 1e-3):
    train_df, test_df, target, label_encoder, label_to_id, id_to_label = Data_preprocess()
    trainset, validset, testset = all_custom_setting()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    kfold = KFold(n_splits = fold_num, shuffle = True, random_state= 128)
    
    for fold, (train_idx, valid_idx) in enumerate(kfold.split(trainset)):
        print(f'Fold: {fold}/{fold_num} is Starting!')
        
        train_subsampler = SubsetRandomSampler(train_idx)
        valid_subsampler = SubsetRandomSampler(valid_idx)
        
        train_DataLoader = DataLoader(trainset, batch_size = 32, pin_memory=True, sampler = train_subsampler)
        valid_DataLoader = DataLoader(validset, batch_size = 32, pin_memory=True, sampler = valid_subsampler)
        model = efficient_model(num_class= len(label_to_id), model_name = 'b0').to(device)
        
        criterion = CutMixCrossEntropyLoss(True)
        optimizer = optim.Adam(model.parameters(), lr = lr)
        scheduler = CosineAnnealingLR(optimizer = optimizer, T_max = epochs)
        
        for epoch in range(epochs):
            training_loss = 0.0
            validation_loss = 0.0
            total = 0
            validation_accuracy = 0.0

            # Training data
            for data in train_DataLoader:
                accuracy = 0.0
                images, labels = data[0].to(device), data[1].to(device)
                optimizer.zero_grad()
                output = model(images)
                loss = criterion(output, labels)
                loss.backward()
                training_loss += loss.item()
                optimizer.step()
            scheduler.step()
            
            train_loss_value = training_loss / len(train_DataLoader)
            
            # Validation data
            with torch.no_grad():
                model.eval()
                for data in valid_DataLoader:
                    images, labels = data[0].to(device), data[1].to(device)
                    outputs = model(images)
                    val_loss = criterion(outputs, labels)

                    _, predicted = torch.max(outputs, 1)
                    _, labels = torch.max(labels, 1)
                    validation_loss += val_loss.item()
                    total += labels.size(0)
                    validation_accuracy += (predicted == labels).sum().item()
            
            val_loss_value = validation_loss / len(valid_DataLoader)
            accuracy = (100 * validation_accuracy / total)

            print(f'{epoch}/{epochs} Train_Loss: {train_loss_value}')
            print(f' Validation_Loss: {val_loss_value} Validation_Accuracy : {validation_accuracy}')

        torch.save(model.state_dict(), f'model_record/eff-b7_{fold}')

train(fold_num= 5)
