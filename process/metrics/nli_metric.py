import numpy as np
from sklearn.preprocessing import Binarizer
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score, precision_recall_fscore_support



def RTE_metric(args, golds, preds):
    accuracy = accuracy_score(golds, preds)
    result = {"accuracy": accuracy}
    return result

def compute_metric(args, golds, preds):
    if args.task_name == "RTE":
        return RTE_metric(args, golds, preds)
    else:
        raise ValueError("wrong task name")