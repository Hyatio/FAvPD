import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

def compute_metric(args, golds, preds):
    assert len(golds) == len(preds)
    if args.task_name == 'MedQAUSMLE':
        return MedQAUSMLE_metric(golds, preds)
    else:
        raise ValueError("task name error")

def MedQAUSMLE_metric(golds, preds):
    accuracy = accuracy_score(golds, preds)
    result = {"accuracy": accuracy}
    return result