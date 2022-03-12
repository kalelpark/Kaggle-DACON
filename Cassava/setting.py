import os
import random
import torch
import numpy as np
import pandas as pd


class setting:
    train_dir = 'files/train.csv'
    test_dir = None
    sample_dir = 'files/sample_submission.csv'

    SEED = 1234                 # SEED
    NUM_FOLDS = 5
    bs = 32                     #  BatchSIZE
    EPOCHS = 40
    sz = 512                    #  ImageSIZE
    SNAPMIX_ALPHA = 5.0         # Review use 5.0 (else, don't affect improve)
    Normalize_mean = [0.485, 0.456, 0.406]    # MEAN 
    Normalize_std = [0.229, 0.224, 0.225]      # STD        
    SNAPMIX_PCT  = 0.5
    GRAD_ACCM_STEPS = 1
    mpl = 255.0                 # max_pixel_value
    TIMM_MODEL = 'resnet50'
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def seed_everything(seed):                                                                # SETTING SEED
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def checkpoint(model, optimizer, epoch, current_metric, best_metric, fold):                # SAVE MODEL
    print("Metric improved from %f to %f, Saving Model at Epoch #%d" %(best_metric, current_metric, epoch))
    ckpt = {
        'model' : model,
        'state_dict' : model.state_dict(),
        'metric' : current_metric
    }
    torch.save(ckpt, 'ckpt_%s-%d-%d.pth' % (setting.TIMM_MODEL, setting.sz, fold))