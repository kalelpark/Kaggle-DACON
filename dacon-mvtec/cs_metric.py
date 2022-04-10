from sklearn.metrics import f1_score
import torch

def f1_function(real, pred):
    score = f1_score(real, pred, average="macro")
    return score