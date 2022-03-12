import numpy as np
from sklearn.metrics import accuracy_score

def accuracy_metric(inputs, targs):
    return accuracy_score(targs.cpu(), inputs.cpu())

def print_kaggle_scores(scores):
    kaggle_metric = np.average(scores)
    print("Kaggle Metric : %f" % (kaggle_metric))
    
    return kaggle_metric 

def accuracy_metric_cpu(inputs, targs):
    return accuracy_score(targs, inputs)