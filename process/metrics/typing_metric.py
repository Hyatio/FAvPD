import numpy as np
from sklearn.preprocessing import Binarizer
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score, precision_recall_fscore_support

def accuracy(threshold, l, out):
    cnt = 0
    y1 = []
    y2 = []
    for x1, x2 in zip(out, l):
        yy1 = []
        yy2 = []
        top = max(x1)
        for i in range(len(x1)):
            if x1[i] > threshold or x1[i] == top:
                yy1.append(i)
            if x2[i] > 0:
                yy2.append(i)
        y1.append(yy1)
        y2.append(yy2)
        cnt += set(yy1) == set(yy2)
    return cnt, y2, y1

def f1(p, r):
    if r == 0.:
        return 0.
    return 2 * p * r / float( p + r )

def loose_macro(true, pred):
    num_entities = len(true)
    p = 0.
    r = 0.
    for true_labels, predicted_labels in zip(true, pred):
        if len(predicted_labels) > 0:
            p += len(set(predicted_labels).intersection(set(true_labels))) / float(len(predicted_labels))
        if len(true_labels) > 0:
            r += len(set(predicted_labels).intersection(set(true_labels))) / float(len(true_labels))
    precision = p / num_entities
    recall = r / num_entities
    return precision, recall, f1( precision, recall)

def loose_micro(true, pred):
    num_predicted_labels = 0.
    num_true_labels = 0.
    num_correct_labels = 0.
    for true_labels, predicted_labels in zip(true, pred):
        num_predicted_labels += len(predicted_labels)
        num_true_labels += len(true_labels)
        num_correct_labels += len(set(predicted_labels).intersection(set(true_labels))) 
    if num_predicted_labels > 0:
        precision = num_correct_labels / num_predicted_labels
    else:
        precision = 0.
    recall = num_correct_labels / num_true_labels
    return precision, recall, f1( precision, recall)

def OpenEntity_metric(args, golds, preds):
    binarizer = Binarizer(threshold=args.threshold)
    preds = binarizer.transform(preds)
    acc = accuracy_score(golds, preds)
    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(golds, preds, average='micro', zero_division=0)
    result = {
        "accuracy": acc,
        "micro-p": micro_p,
        "micro-r": micro_r,
        "micro-f1": micro_f1
    }
    return result

def FIGER_metric(args, golds, preds):
    acc, golds, preds = accuracy(args.threshold, golds, preds)
    acc = acc / float(len(golds))
    _, _, macro = loose_macro(golds, preds)
    _, _, micro = loose_micro(golds, preds)
    result = {
        "accuracy": acc,
        "loose-macro": macro,
        "loose-micro": micro
    }
    return result

def compute_metric(args, golds, preds):
    if args.task_name == "OpenEntity":
        return OpenEntity_metric(args, golds, preds)
    elif args.task_name == "FIGER":
        return FIGER_metric(args, golds, preds)
    else:
        raise ValueError("wrong task name")