import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

def compute_metric(args, golds, preds):
    assert len(golds) == len(preds)
    if args.task_name == 'TACRED':
        return tacred_metric(args, golds, preds)
    elif args.task_name == 'FewRel':
        return fewrel_metric(golds, preds)
    elif args.task_name == 'CHEMPROT':
        return chemprot_metric(golds, preds)
    else:
        raise ValueError("task name error")

def fewrel_metric(golds, preds):
    accuracy = accuracy_score(golds, preds)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(golds, preds, average='macro', zero_division=0)
    result = {
        "accuracy": accuracy,
        "macro-p": macro_p,
        "macro-r": macro_r,
        "macro-f1": macro_f1
    }
    return result

def tacred_metric(args, golds, preds):
    total = len(golds)
    na_label = 'no_relation'
    na_id = args.label_list.index(na_label)
    accuracy = accuracy_score(golds, preds)
    conf_mt = confusion_matrix(golds, preds)

    # na_id refers to class "no relation" which is always regarded as a negative class
    # micro: ignore na id
    tp = np.diagonal(conf_mt).sum() - np.diagonal(conf_mt)[na_id]
    fp = total - conf_mt[:,0].sum()-tp
    fn = total - conf_mt[0].sum()-tp
    micro_p_igna = float(tp) / (float(tp+fp) + 0.00000001)
    micro_r_igna = float(tp) / (float(tp+fn) + 0.00000001)
    try:
        micro_f1_igna = 2 * micro_p_igna * micro_r_igna / (micro_p_igna + micro_r_igna)
    except:
        micro_f1_igna = 0
    # macro: ignore na id
    p_all = np.delete(np.diagonal(conf_mt)/(np.sum(conf_mt, axis=0)+0.00000001), na_id)
    r_all = np.delete(np.diagonal(conf_mt)/(np.sum(conf_mt, axis=0)+0.00000001), na_id)
    macro_p_igna = p_all.mean()
    macro_r_igna = r_all.mean()
    macro_f1_igna = ((2*p_all*r_all)/(p_all+r_all+0.00000001)).mean()
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(golds, preds, average='macro', zero_division=0)
    # result = {
    #     "accuracy": accuracy,
    #     "micro-p": micro_p_igna,
    #     "micro-r": micro_r_igna,
    #     "micro-f1": micro_f1_igna,
    #     "macro-p": macro_p_igna,
    #     "macro-r": macro_r_igna,
    #     "macro-f1": macro_f1_igna,
    #     "macro-p(origin)": macro_p,
    #     "macro-r(origin)": macro_r,
    #     "macro-f1(origin)": macro_f1
    # }
    result = {
        "accuracy": accuracy,
        "micro-p": micro_p_igna,
        "micro-r": micro_r_igna,
        "micro-f1": micro_f1_igna,
    }
    return result

def chemprot_metric(golds, preds):
    accuracy = accuracy_score(golds, preds)
    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(golds, preds, average='micro', zero_division=0)
    result = {
        "accuracy": accuracy,
        "micro-p": micro_p,
        "micro-r": micro_r,
        "micro-f1": micro_f1
    }
    return result

def bad_case_study(golds, preds):
    bad_case_index = []
    for i, (gold, pred) in enumerate(zip(golds, preds)):
        if gold != pred:
            bad_case_index.append(i)
    return bad_case_index